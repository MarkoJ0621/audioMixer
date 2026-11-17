#!/usr/bin/env python3
"""
gallery_mix.py - layer a folder of videos and/or audio files into one
gallery-style video.

Audio, per input:
  1. high-pass (removes rumble) + denoise, by default (auto) with
       voice: DeepFilterNet3, a neural speech enhancer, on speech and singing;
         it handles fans, typing, cafes, traffic and wind
       spectral (see denoise.py) on everything else: a noise profile learned
         from the track's quietest moments, removed with a decision-directed
         Wiener filter; safe for music
     The voice model tells them apart: it takes a few dB out of speech but
     20-30 dB out of instruments.
  2. loudness-match to a common LUFS target, gentle compression so no single
     track dominates the layered mix, then re-measure and trim to target.
     A track with no speech that's far quieter than the people talking (a
     participant who never speaks) keeps its own level instead.
Mix:
  3. sum every track, normalize the master to the output LUFS target and
     peak-limit it
Video:
  4. tile every video into a centered grid (audio-only files are heard in
     the mix but get no tile); HDR phone video is tone-mapped to SDR

Needs ffmpeg + ffprobe on PATH (tested with ffmpeg 7.1) and numpy; the voice
denoiser also needs the deep-filter binary (on PATH or in ./bin), without it
every track gets the spectral denoiser.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import denoise

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".ts", ".mts", ".m2ts",
              ".mpg", ".mpeg", ".vob", ".wmv", ".flv", ".f4v", ".3gp", ".3g2", ".mxf",
              ".ogv", ".dv"}
AUDIO_EXTS = {".wav", ".w64", ".rf64", ".bwf", ".aif", ".aiff", ".aifc", ".flac", ".mp3",
              ".mp2", ".m4a", ".aac", ".ac3", ".ogg", ".oga", ".opus", ".wma", ".caf",
              ".mka", ".wv", ".ape", ".amr"}
OUTPUT_EXTS = {".mp4", ".mov", ".m4v", ".mkv"}
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}

SAMPLE_RATE = 48000
RAW_IN = ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2"]   # our headerless intermediates
TRACK_LUFS = -23.0       # every track is matched to this before mixing
SILENT_LUFS = -60.0      # quieter than this counts as silence (no gain applied)
MASTER_PEAK_DB = -1.0    # limiter ceiling on the final mix
DEFAULT_DENOISE_DB = {"spectral": 15.0, "voice": 30.0}
# loudness the voice model may take out of a track that is still speech. Measured:
# speech loses 0-7.4 dB (fan, keyboard, cafe, traffic, wind, far laptop mic,
# 0 to -5 dB SNR, phone AAC); instruments and bands lose 13-30 dB
NOT_SPEECH_DB = 10.0
# a speechless track this much quieter than the median speaker is background
BACKGROUND_DB = 12.0
DEEP_FILTER = shutil.which("deep-filter") or str(Path(__file__).resolve().parent / "bin" / "deep-filter")
DEEP_FILTER_URL = ("https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/"
                   "deep-filter-0.5.6-aarch64-apple-darwin")


@dataclass
class Media:
    path: Path
    has_video: bool
    has_audio: bool
    duration: float
    aspect: float | None = None   # display aspect ratio of the video stream
    sar: float = 1.0              # sample (pixel) aspect ratio of the video stream
    hdr: str = ""                 # "arib-std-b67" (HLG) or "smpte2084" (PQ) for HDR video
    channels: int = 0
    layout: str = ""              # ffprobe channel layout, "" or "unknown" if unspecified
    wav: Path | None = None       # cleaned + balanced audio
    gain_db: float = 0.0          # residual gain applied at mix time
    lufs: float | None = None     # loudness measured after denoising


@dataclass
class Layout:
    cols: int
    rows: int
    tile_w: int
    tile_h: int
    positions: list[tuple[int, int]]
    grid_w: int
    grid_h: int


def fail(msg: str):
    sys.exit(f"error: {msg}")


def even(v: float) -> int:
    return int(v) // 2 * 2


def ffmpeg(args: list[str], cwd: Path | None = None, progress: bool = False) -> str:
    """Run ffmpeg and return its stderr (where ebur128 reports loudness)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    cmd += ["-loglevel", "error", "-stats"] if progress else ["-nostats"]
    if progress:    # our own messages first, then ffmpeg's (on the same stderr/stdout)
        sys.stdout.flush()
    proc = subprocess.run(cmd + args, cwd=cwd, encoding="utf-8", errors="replace",
                          stderr=None if progress else subprocess.PIPE)
    if proc.returncode != 0:
        tail = "" if progress else "\n" + "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}{tail}")
    return proc.stderr or ""


def measured_lufs(stderr: str) -> float:
    hits = re.findall(r"I:\s+(-?inf|-?\d+(?:\.\d+)?) LUFS", stderr)
    if not hits:
        raise RuntimeError("could not read loudness from ffmpeg output")
    return float(hits[-1])


def measured_peak(stderr: str) -> float:
    hits = re.findall(r"Peak:\s+(-?inf|-?\d+(?:\.\d+)?) dBFS", stderr)
    if not hits:
        raise RuntimeError("could not read sample peak from ffmpeg output")
    return float(hits[-1])


# ---------------------------------------------------------------- discovery

def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _ratio(v) -> float | None:
    try:
        a, b = (int(x) for x in str(v).split(":"))
        return a / b if a and b else None
    except ValueError:
        return None


def _rotation(stream: dict) -> float:
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return _num(sd["rotation"])
    return _num(stream.get("tags", {}).get("rotate"))


def probe(path: Path) -> Media | None:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                           "-of", "json", str(path)],
                          capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return None
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    # cover art embedded in mp3/m4a shows up as a video stream; ignore it
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video and not audio:
        return None

    duration = (_num(info.get("format", {}).get("duration"))
                or max((_num(s.get("duration")) for s in streams), default=0.0))
    m = Media(path, has_video=bool(video), has_audio=bool(audio), duration=duration)
    if audio:
        m.channels = int(_num(audio.get("channels")))
        m.layout = audio.get("channel_layout", "")
    if video and video.get("width") and video.get("height"):
        m.sar = _ratio(video.get("sample_aspect_ratio")) or 1.0
        if video.get("color_transfer") in HDR_TRANSFERS:   # iPhones film HLG by default
            m.hdr = video["color_transfer"]
        dw = video["width"] * m.sar
        dh = video["height"]
        if abs(_rotation(video)) % 180 == 90:
            dw, dh = dh, dw
        m.aspect = dw / dh
    return m


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def discover(folder: Path, recursive: bool, exclude: set[Path]) -> list[Media]:
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    # skip hidden files and folders (.git, .thumbnails, macOS ._ resource forks)
    files = sorted((p for p in candidates
                    if p.is_file()
                    and not any(part.startswith(".") for part in p.relative_to(folder).parts)
                    and p.suffix.lower() in VIDEO_EXTS | AUDIO_EXTS
                    and p.resolve() not in exclude),
                   key=lambda p: natural_key(p.relative_to(folder)))
    media = []
    for p in files:
        m = probe(p)
        if m is None:
            print(f"  skipping {p.name}: no readable audio or video")
        else:
            media.append(m)
    return media


# ------------------------------------------------------------------- layout

def plan_layout(n: int, width: int, height: int, gap: int, aspect: float,
                cols: int | None = None) -> Layout:
    """Grid of equal tiles at `aspect`, sized as large as the canvas allows."""
    def fit(c: int):
        r = math.ceil(n / c)
        cell_w = (width - gap * (c + 1)) / c
        cell_h = (height - gap * (r + 1)) / r
        tw = min(cell_w, cell_h * aspect)
        return tw, tw / aspect, r

    if cols is None:
        # biggest tiles win; on a tie prefer the wider grid
        cols = max(range(1, n + 1), key=lambda c: (round(fit(c)[0], 3), c))
    cols = max(1, min(cols, n))
    tw, th, rows = fit(cols)
    tw, th = even(tw), even(th)
    if tw < 16 or th < 16:
        fail(f"{n} tiles don't fit on a {width}x{height} canvas; raise --size or lower --gap")

    positions = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        x0 = (cols - in_row) * (tw + gap) // 2    # center a partial last row
        positions.append((even(x0 + c * (tw + gap)), even(r * (th + gap))))
    return Layout(cols, rows, tw, th, positions,
                  grid_w=cols * tw + (cols - 1) * gap,
                  grid_h=rows * th + (rows - 1) * gap)


# -------------------------------------------------------------------- audio

@dataclass
class Cleaned:
    """One track after denoising, before balancing."""
    path: Path            # denoised audio
    fmt: list[str]        # ffmpeg input options to read it
    lufs: float           # its loudness at the source's own level
    pregain: float        # dB the file sits above the source level (deep-filter input gain)
    seconds: float        # decoded length
    source: float = 0.0   # loudness before denoising
    kind: str = ""        # "speech" or "other" when the voice model judged it, else ""
    removed: float = 0.0  # dB of loudness the voice model took out
    denoiser: str = ""    # which denoiser's output this is
    voice_alt: "Cleaned | None" = None   # auto, no speech: the voice model's version too


def clean_audio(m: Media, idx: int, tmp: Path, args) -> Cleaned:
    """Decode one input to 48 kHz stereo and denoise it (runs in a worker).

    With the voice model available (auto/voice), every track goes through it
    first: on speech it removes a few dB at most, even at 0 dB SNR or in a
    cafe, while on instruments it removes 20-30 dB, the music itself. That
    tells speech from everything else; in auto mode the rest is redone with
    the spectral denoiser, which keeps music intact.
    """
    try:
        limit = ["-t", str(args.max_duration)] if args.max_duration else []
        mode = args.denoiser if args.denoise > 0 else "off"
        raw = tmp / f"{idx:03d}_raw.f32"
        chain = []
        if m.channels > 2 and m.layout in ("", "unknown"):
            # e.g. a multitrack field recorder: ffmpeg can't guess a downmix, so average
            every = "+".join(f"c{c}" for c in range(m.channels))
            chain.append(f"pan=stereo|c0<{every}|c1<{every}")
        # keep the file's own timeline: an audio stream that starts after the
        # video (camera files, lossless trims) is padded so it stays in sync
        chain += [f"aresample={SAMPLE_RATE}:async=1:first_pts=0",
                  "aformat=sample_fmts=flt:channel_layouts=stereo"]
        if args.highpass > 0:
            chain.append(f"highpass=f={args.highpass}")
        chain.append("ebur128=peak=sample:framelog=quiet")
        err = ffmpeg([*limit, "-i", str(m.path), "-map", "0:a:0", "-af", ",".join(chain),
                      "-f", "f32le", str(raw)])
        # exact, unlike the container's duration, which for raw streams (ADTS
        # .aac, some .mp3) is only estimated from the bitrate
        seconds = raw.stat().st_size / (8 * SAMPLE_RATE)
        before = measured_lufs(err)
        if before < SILENT_LUFS:
            mode = "off"

        if mode in ("auto", "voice"):
            dfw, pregain = deep_filter(raw, measured_peak(err), tmp / f"df_{idx:03d}", args.voice_db)
            lufs = loudness([], dfw) - pregain
            removed = before - lufs
            kind = "speech" if removed < NOT_SPEECH_DB else "other"
            if mode == "voice" or kind == "speech":
                raw.unlink()
                return Cleaned(dfw, [], lufs, pregain, seconds, before, kind, removed, "voice")
            # kept: if this turns out to be a silent participant, it's the cleaner one
            alt = Cleaned(dfw, [], lufs, pregain, seconds, before, kind, removed, "voice")
            c = Cleaned(raw, RAW_IN, before, 0.0, seconds, before, kind, removed, voice_alt=alt)
        else:
            c = Cleaned(raw, RAW_IN, before, 0.0, seconds, before)
        if mode in ("auto", "spectral"):
            c.path = tmp / f"{idx:03d}_clean.f32"
            denoise.denoise(raw, c.path, channels=2, max_reduction_db=args.spectral_db)
            raw.unlink()
            c.lufs, c.denoiser = loudness(RAW_IN, c.path), "spectral"
        return c
    except RuntimeError as e:
        raise RuntimeError(f"{m.path.name}: {e}") from None


def balance_audio(c: Cleaned, idx: int, tmp: Path, args, keep_level: bool = False,
                  suffix: str = "") -> tuple[float, Path]:
    """Gain a cleaned track to TRACK_LUFS and compress it, or with keep_level
    (and for silence) leave it at its own level. Returns (residual gain, wav)."""
    try:
        wav = tmp / f"{idx:03d}{suffix}.wav"
        quiet = keep_level or c.lufs < SILENT_LUFS
        file_lufs = c.lufs + c.pregain
        chain = [f"volume={-c.pregain if quiet else TRACK_LUFS - file_lufs:.2f}dB"]
        if args.compress > 1 and not quiet:
            # only the passages that poke well above the track's average get squashed
            chain.append(f"acompressor=threshold={TRACK_LUFS + 3:.0f}dB:ratio={args.compress}"
                         ":attack=20:release=250:knee=4")
        chain.append("ebur128=framelog=quiet")
        err = ffmpeg([*c.fmt, "-i", str(c.path), "-af", ",".join(chain),
                      "-c:a", "pcm_f32le", "-rf64", "auto", str(wav)])
        c.path.unlink()
        return 0.0 if quiet else TRACK_LUFS - measured_lufs(err), wav
    except RuntimeError as e:
        raise RuntimeError(f"track {idx}: {e}") from None


def process_audio(m: Media, idx: int, tmp: Path, args, keep_level: bool = False):
    """clean_audio + balance_audio for one track on its own.
    Returns (LUFS after denoising, residual gain dB, wav, Cleaned)."""
    c = clean_audio(m, idx, tmp, args)
    gain, wav = balance_audio(c, idx, tmp, args, keep_level)
    return c.lufs, gain, wav, c


def loudness(fmt: list[str], path: Path) -> float:
    return measured_lufs(ffmpeg([*fmt, "-i", str(path), "-af", "ebur128=framelog=quiet",
                                 "-f", "null", "-"]))


def deep_filter(raw: Path, peak_db: float, outdir: Path, max_reduction_db: float):
    """Run DeepFilterNet3 on a raw track; returns (denoised wav, pre-gain in dB)."""
    # deep-filter writes 16-bit files: peak-normalize first so quiet sources
    # keep their resolution (loudness is re-measured afterwards anyway)
    pregain = -1.0 - peak_db if peak_db > -120 else 0.0
    src = outdir.with_suffix(".wav")
    ffmpeg([*RAW_IN, "-i", str(raw), "-af", f"volume={pregain:.2f}dB",
            "-c:a", "pcm_f32le", str(src)])
    # -D trims the model's 30 ms latency so the audio stays in sync with the video
    proc = subprocess.run([DEEP_FILTER, "-D", "-a", f"{max_reduction_db:g}", "-o", str(outdir),
                           str(src)], capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"deep-filter exited with code {proc.returncode}\n{tail}")
    src.unlink()
    return outdir / src.name, pregain


def pan_filter(p: float) -> str:
    """Constant-power pan of the mono fold-down; p in [-1, 1]."""
    theta = (p + 1) * math.pi / 4
    gl, gr = math.cos(theta) / math.sqrt(2), math.sin(theta) / math.sqrt(2)
    return f"pan=stereo|c0={gl:.4f}*c0+{gl:.4f}*c1|c1={gr:.4f}*c0+{gr:.4f}*c1"


def mix_and_master(tracks: list[tuple[Media, float | None]], tmp: Path, target_lufs: float):
    # scale the sum to about TRACK_LUFS: ffmpeg's ebur128 can't measure above
    # +10 LUFS, which many correlated tracks (everyone playing the same part) reach
    headroom = 10 * math.log10(len(tracks))
    inputs, chains = [], []
    for k, (m, pan) in enumerate(tracks):
        inputs += ["-i", str(m.wav)]
        chain = f"[{k}:a]volume={m.gain_db - headroom:.2f}dB"
        if pan is not None:
            chain += "," + pan_filter(pan)
        chains.append(chain + f"[a{k}]")
    n = len(tracks)
    graph = ";".join(chains) + ";" + "".join(f"[a{k}]" for k in range(n)) + \
        f"amix=inputs={n}:duration=longest:normalize=0:dropout_transition=0," \
        "ebur128=framelog=quiet[mix]"
    mix = tmp / "mix.wav"
    err = ffmpeg(inputs + ["-filter_complex", graph, "-map", "[mix]",
                           "-c:a", "pcm_f32le", "-rf64", "auto", str(mix)])
    mix_lufs = measured_lufs(err)

    master = tmp / "master.wav"
    gain = target_lufs - mix_lufs if mix_lufs > SILENT_LUFS else 0.0
    err = ffmpeg(["-i", str(mix), "-af",
                  f"volume={gain:.2f}dB,"
                  f"alimiter=limit={MASTER_PEAK_DB}dB:level=disabled:latency=1,"
                  "ebur128=framelog=quiet",
                  "-c:a", "pcm_s24le", "-rf64", "auto", str(master)])
    mix.unlink()
    return master, mix_lufs + headroom, measured_lufs(err)


def settle_background(tracks: list[Media], cleaned: dict, tmp: Path, args
                      ) -> list[tuple[Media, str]]:
    """Decide what to do with tracks the voice model found no speech in.

    Next to people talking, a track with no speech that is also far quieter
    than them is a participant who never talks: their fan, typing or street.
    Balancing would raise it to full level, so it stays at its own level, and
    gets the voice model's version, which removes 20+ dB of such noise where
    the spectral denoiser manages 0-12 (it adds up over many participants).
    Other speechless tracks are music (or loud noise): spectral-cleaned in
    auto mode and balanced like the rest. Returns messages to print.
    """
    speech = [cleaned[id(m)].source for m in tracks if cleaned[id(m)].kind == "speech"]
    notes = []
    for i, m in enumerate(tracks):
        c = cleaned[id(m)]
        alt, c.voice_alt = c.voice_alt, None
        background = (c.kind == "other" and speech and m.lufs >= SILENT_LUFS
                      and statistics.median(speech) - c.source > BACKGROUND_DB)
        if background:
            below = statistics.median(speech) - c.source
            if alt:
                m.wav.unlink()
                m.gain_db, m.wav = balance_audio(alt, i, tmp, args, keep_level=True, suffix="_bg")
                m.lufs = alt.lufs
            else:
                m.gain_db += c.lufs - TRACK_LUFS      # undo the balancing gain
            notes.append((m, f"no speech and {below:.0f} dB quieter than the speakers: "
                             "background, voice-denoised and kept at its own level"))
            continue
        if alt:
            alt.path.unlink()
        if c.kind == "other" and m.lufs >= SILENT_LUFS and args.denoiser == "voice":
            notes.append((m, f"warning: the voice model removed {c.removed:.0f} dB, so this "
                             "probably isn't speech; --denoiser auto would keep it"))
    return notes


# -------------------------------------------------------------------- video

def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return cp >= 0x1F000 or 0x2600 <= cp <= 0x27BF or cp in (0x200D, 0xFE0E, 0xFE0F)


def label_text_and_font(text: str, font: str | None) -> tuple[str, str | None]:
    """Label text and the font file to draw it with (None = ffmpeg's default).

    ffmpeg's default font has no CJK, Arabic, Hebrew, ... glyphs and draws boxes
    instead, so for such names ask fontconfig for a font that covers them.
    Emoji are dropped: drawtext can't render color emoji fonts.
    """
    text = "".join(ch for ch in text if not _is_emoji(ch)).strip() or text
    if font or all(ord(ch) < 0x250 for ch in text):   # Latin is in every font
        return text, font
    return text, _fontconfig_file("".join(sorted(set(text))))


@functools.lru_cache(maxsize=None)
def _fontconfig_file(chars: str) -> str | None:
    fc = shutil.which("fc-match")
    if not fc:
        return None
    pattern = ":charset=" + " ".join(f"{ord(ch):x}" for ch in chars if not ch.isspace())
    proc = subprocess.run([fc, "-f", "%{file}", pattern], capture_output=True,
                          encoding="utf-8", errors="replace")
    found = proc.stdout.strip()
    return found if proc.returncode == 0 and found and Path(found).is_file() else None


def tonemap_filter(transfer: str) -> str:
    """HDR (HLG or PQ, BT.2020) to SDR BT.709. Diffuse white sits at 203 nits in
    both (ITU-R BT.2408), so that maps to SDR white; brighter highlights roll
    off smoothly instead of clipping, and mid-tones keep their brightness."""
    return (f"zscale=tin={transfer}:pin=bt2020:min=bt2020nc:t=linear:npl=203,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=tonemap=mobius:param=0.7:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


@functools.lru_cache(maxsize=None)
def have_filters(*names: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                         encoding="utf-8", errors="replace").stdout
    listed = {line.split()[1] for line in out.splitlines() if len(line.split()) > 2}
    return all(n in listed for n in names)


def render(videos: list[Media], layout: Layout, master: Path | None, duration: float,
           tmp: Path, out: Path, args) -> None:
    width, height = args.size
    tw, th = layout.tile_w, layout.tile_h
    d = f"{duration:.3f}"
    inputs, chains = [], []
    for i, m in enumerate(videos):
        inputs += ["-i", str(m.path)]
        if args.fit == "cover":
            steps = [f"scale={tw}:{th}:force_original_aspect_ratio=increase", f"crop={tw}:{th}"]
        else:
            steps = [f"scale={tw}:{th}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                     f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color={args.bg}"]
        if m.hdr and have_filters("zscale", "tonemap"):
            # at tile size (cheap; zscale needs the even sizes crop/scale give here)
            # and before the background color is padded in
            steps.insert(2 if args.fit == "cover" else 1, tonemap_filter(m.hdr))
        if abs(m.sar - 1) > 1e-3:   # anamorphic: square the pixels before fitting
            steps.insert(0, "scale=trunc(iw*sar/2)*2:ih,setsar=1")
        fit = ",".join(steps)
        hold = "stop_mode=clone" if args.hold == "freeze" else f"stop_mode=add:color={args.bg}"
        # start_time=0 keeps the file's own timeline (same as the audio): a video
        # stream that starts late holds its first frame until then
        chain = (f"[{i}:v:0]fps={args.fps}:start_time=0,{fit},setsar=1,format=yuv420p,"
                 f"tpad={hold}:stop_duration={d},trim=duration={d}")
        if args.labels:
            # ffmpeg runs with cwd=tmp, so bare filenames need no filtergraph escaping
            text, font = label_text_and_font(m.path.stem, args.font)
            (tmp / f"label_{i:03d}.txt").write_text(text, encoding="utf-8")
            fontopt = ""
            if font:
                link = tmp / f"font_{i:03d}{Path(font).suffix}"
                link.symlink_to(Path(font).resolve())
                fontopt = f":fontfile={link.name}"
            fs = max(12, th // 14)
            chain += (f",drawtext=textfile=label_{i:03d}.txt{fontopt}:expansion=none:fontcolor=white"
                      f":fontsize={fs}:box=1:boxcolor=black@0.55:boxborderw={fs // 3}"
                      f":x={fs // 2}:y=h-th-{fs // 2}")
        chains.append(chain + f"[t{i}]")

    canvas = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={args.bg},setsar=1[vout]"
    if len(videos) == 1:
        chains.append(f"[t0]{canvas}")
    else:
        grid = "|".join(f"{x}_{y}" for x, y in layout.positions)
        chains.append("".join(f"[t{i}]" for i in range(len(videos))) +
                      f"xstack=inputs={len(videos)}:layout={grid}:fill={args.bg},{canvas}")

    audio_out = []
    if master:
        audio_out = ["-map", f"{len(videos)}:a", "-c:a", "aac", "-b:a", "256k"]
        inputs += ["-i", str(master)]
    ffmpeg(inputs + ["-filter_complex", ";".join(chains), "-map", "[vout]", *audio_out,
                     "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                     "-pix_fmt", "yuv420p", "-t", d, "-movflags", "+faststart", str(out)],
           cwd=tmp, progress=True)


# --------------------------------------------------------------------- main

def parse_size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)[x:](\d+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1080")
    return even(int(m[1])), even(int(m[2]))


def main():
    ap = argparse.ArgumentParser(
        description="Clean up, balance and layer every video/audio file in a folder, "
                    "rendered as a gallery of the videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder of video and/or audio files")
    ap.add_argument("-o", "--output", type=Path,
                    help="output video (default: ./<folder>_gallery.mp4)")
    ap.add_argument("-r", "--recursive", action="store_true", help="include subfolders")

    g = ap.add_argument_group("audio")
    g.add_argument("--lufs", type=float, default=-16.0,
                   help="loudness target for the final mix (-14 for streaming, -23 for broadcast)")
    g.add_argument("--denoiser", choices=["auto", "voice", "spectral"], default="auto",
                   help="auto: the voice model (DeepFilterNet3) on speech and singing, the "
                        "spectral denoiser on everything else; voice: the voice model on "
                        "every track (it strips instruments); spectral: safe for any material")
    g.add_argument("--denoise", type=float, metavar="DB", default=argparse.SUPPRESS,
                   help="maximum noise reduction in dB, 0 = off "
                        "(default: 30 for the voice model, 15 for spectral)")
    g.add_argument("--highpass", type=float, default=80.0, help="rumble filter cutoff in Hz (0 = off)")
    g.add_argument("--compress", type=float, default=3.0,
                   help="per-track compression ratio for balancing dynamics (1 = off)")
    g.add_argument("--pan", type=float, default=0.0, metavar="SPREAD",
                   help="pan each video's audio by its gallery column, 0 (off) to 1 (hard L/R)")
    g.add_argument("--save-audio", action="store_true",
                   help="also write the mastered mix as a WAV next to the video")

    g = ap.add_argument_group("video")
    g.add_argument("--size", type=parse_size, default=(1920, 1080), help="output resolution")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--cols", type=int, help="force the number of grid columns")
    g.add_argument("--gap", type=int, default=8, help="pixels between tiles")
    g.add_argument("--bg", default="black", help="background color (name or 0xRRGGBB)")
    g.add_argument("--fit", choices=["cover", "contain"], default="cover",
                   help="crop clips to fill their tile, or letterbox them")
    g.add_argument("--hold", choices=["freeze", "black"], default="freeze",
                   help="what a tile shows after its clip ends")
    g.add_argument("--labels", action="store_true", help="show file names on tiles")
    g.add_argument("--font", help="font file for --labels (default: ffmpeg's, or one that "
                                  "covers the names' characters)")
    g.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    g.add_argument("--preset", default="medium", help="x264 speed preset")

    g = ap.add_argument_group("misc")
    g.add_argument("--max-duration", type=float, help="cut the output at this many seconds")
    g.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="files to process in parallel")
    g.add_argument("--keep-temp", action="store_true", help="keep intermediate files")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fail(f"{tool} not found on PATH (macOS: brew install ffmpeg)")
    install = ("From the project folder run:\n"
               f"  mkdir -p bin && curl -L -o bin/deep-filter {DEEP_FILTER_URL} "
               "&& chmod +x bin/deep-filter")
    if args.denoiser != "spectral" and not os.access(DEEP_FILTER, os.X_OK):
        if args.denoiser == "voice" and getattr(args, "denoise", 1) > 0:
            fail(f"--denoiser voice needs the deep-filter binary. {install}")
        if args.denoiser == "auto":
            print("note: the voice denoiser isn't installed, so every track gets the spectral "
                  f"one, which is weaker on speech. {install}\n")
            args.denoiser = "spectral"
    # one --denoise applies to both denoisers; otherwise each has its own default
    given = getattr(args, "denoise", None)
    args.voice_db = DEFAULT_DENOISE_DB["voice"] if given is None else given
    args.spectral_db = DEFAULT_DENOISE_DB["spectral"] if given is None else given
    args.denoise = 1.0 if given is None else given      # > 0 means on
    if args.font:
        args.font = str(Path(args.font).expanduser().resolve())
        if not Path(args.font).is_file():
            fail(f"--font: {args.font} is not a file")
    folder = args.input_dir.expanduser().resolve()
    if not folder.is_dir():
        fail(f"{folder} is not a folder")
    out = (args.output or Path.cwd() / f"{folder.name}_gallery.mp4").expanduser().resolve()
    if out.suffix.lower() not in OUTPUT_EXTS:   # H.264 + AAC only fits these
        fail(f"output must be one of {', '.join(sorted(OUTPUT_EXTS))}, not {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {folder}")
    # our own output (and the --save-audio mix next to it) from an earlier run
    media = discover(folder, args.recursive, exclude={out, out.with_suffix(".wav")})
    videos = [m for m in media if m.has_video]
    if not videos:
        fail("no video files found; the gallery needs at least one video")
    print(f"Found {len(videos)} videos and {len(media) - len(videos)} audio-only files "
          f"({sum(m.has_audio for m in media)} with audio in total)")

    if any(m.hdr for m in videos) and not have_filters("zscale", "tonemap"):
        print("warning: this ffmpeg has no zscale filter, so HDR clips (most iPhone video) "
              "will look washed out; install an ffmpeg built with libzimg")
    aspects = [m.aspect for m in videos if m.aspect]
    layout = plan_layout(len(videos), *args.size, args.gap,
                         statistics.median(aspects) if aspects else 16 / 9, args.cols)
    print(f"Layout: {layout.cols}x{layout.rows} grid of {layout.tile_w}x{layout.tile_h} "
          f"tiles on {args.size[0]}x{args.size[1]}")

    tmp = Path(tempfile.mkdtemp(prefix="gallery_mix_"))
    try:
        master = None
        with_audio = [m for m in media if m.has_audio]
        if with_audio:
            how = {"auto": f"voice model on speech (up to {args.voice_db:g} dB), spectral on "
                           f"the rest (up to {args.spectral_db:g} dB)",
                   "voice": f"voice model, up to {args.voice_db:g} dB",
                   "spectral": f"spectral denoiser, up to {args.spectral_db:g} dB"}[args.denoiser]
            if args.denoise <= 0:
                how = "no denoising"
            print(f"\n[1/3] Cleaning and balancing {len(with_audio)} audio tracks ({how})")
            cleaned = {}
            with ProcessPoolExecutor(max(1, args.jobs)) as pool:
                futures = {pool.submit(process_audio, m, i, tmp, args): m
                           for i, m in enumerate(with_audio)}
                try:
                    for done, fut in enumerate(as_completed(futures), 1):
                        m = futures[fut]
                        m.lufs, m.gain_db, m.wav, c = fut.result()
                        cleaned[id(m)] = c
                        if not m.has_video:   # the decoded length beats the container's guess
                            m.duration = c.seconds
                        what = {"speech": "speech, ", "other": "no speech, "}.get(c.kind, "")
                        level = ("silent, left as-is" if m.lufs < SILENT_LUFS
                                 else f"{what}{m.lufs:.1f} LUFS -> {TRACK_LUFS:.0f} LUFS")
                        print(f"  [{done}/{len(with_audio)}] {m.path.name}: {level}")
                except BaseException:
                    pool.shutdown(cancel_futures=True)
                    raise
            for m, msg in settle_background(with_audio, cleaned, tmp, args):
                print(f"  {m.path.name}: {msg}")

            print("\n[2/3] Mixing and mastering")
            tracks = []
            for m in with_audio:
                pan = None
                if args.pan > 0 and m.has_video:   # audio-only files have no column: stay centered
                    x, _ = layout.positions[videos.index(m)]
                    pan = ((x + layout.tile_w / 2) / layout.grid_w * 2 - 1) * args.pan
                tracks.append((m, pan))
            master, mix_lufs, final_lufs = mix_and_master(tracks, tmp, args.lufs)
            print(f"  summed mix {mix_lufs:.1f} LUFS -> master {final_lufs:.1f} LUFS, "
                  f"peaks limited to {MASTER_PEAK_DB:.0f} dBFS")
            if mix_lufs > SILENT_LUFS and final_lufs < args.lufs - 1:
                print(f"  warning: {args.lufs - final_lufs:.1f} dB short of --lufs {args.lufs:g}: "
                      "the mix is so peaky (claps, clicks, hits) that the peak limiter "
                      "pulls it down")
        else:
            print("\nNo audio in any input; the video will be silent.")

        duration = max(m.duration for m in media)
        if args.max_duration:
            duration = min(duration, args.max_duration)
        if duration <= 0:
            fail("could not determine the length of any input")
        print(f"\n[3/3] Rendering {duration:.1f}s gallery of {len(videos)} videos")
        render(videos, layout, master, duration, tmp, out, args)
        print(f"\nDone: {out}")
        if args.save_audio and master:
            shutil.copyfile(master, out.with_suffix(".wav"))
            print(f"Mix:  {out.with_suffix('.wav')}")
    except RuntimeError as e:
        fail(str(e))
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        if args.keep_temp:
            print(f"Intermediate files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
