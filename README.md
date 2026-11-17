# gallery-mixer

Point it at a folder of videos and/or audio files. It cleans up and balances
every track, layers them all into one mix, and renders the videos as a gallery
grid. Audio-only files are part of the mix but don't get a tile.

```bash
python3 gallery_mix.py path/to/folder                 # -> ./folder_gallery.mp4
python3 gallery_mix.py path/to/folder -o out.mp4 --labels --pan 0.5 --save-audio
```

Requires **ffmpeg + ffprobe** on your PATH (`brew install ffmpeg`), Python 3.9+,
and numpy (`pip install -r requirements.txt`).

Install the DeepFilterNet3 voice denoiser into `bin/` too. Without it, every
track gets the spectral denoiser, which is much weaker on laptop and phone
speech (the script says so when it runs):

```bash
mkdir -p bin && curl -L -o bin/deep-filter https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-aarch64-apple-darwin && chmod +x bin/deep-filter
```

That build is for Apple Silicon Macs. Other platforms take their binary from the
same [release](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6).

## What it does

**Per track (run in parallel):**
1. Decode to 48 kHz stereo float. Mono is upmixed; surround is downmixed;
   multitrack files with no channel layout (field recorders, ambisonics) are
   averaged. Each file keeps its own timeline, so an audio stream that starts
   later than its video (camera files, lossless trims) stays in sync.
2. High-pass at 80 Hz to remove rumble and handling noise (`--highpass`).
3. Denoise, up to `--denoise` dB of reduction. Both denoisers add no delay, so
   audio stays in sync with the picture.
   - **`--denoiser auto`** (default) uses the voice denoiser below on speech
     and singing and the spectral one on everything else. The voice model
     tells them apart itself. On speech it removes only a few dB of loudness,
     even next to a fan, typing, café chatter, traffic or wind, through a far
     laptop mic, or with noise louder than the voice (0.2–7.4 dB measured).
     On instruments it removes 13–30 dB, the music itself, so those tracks are
     redone with the spectral denoiser. On an all-music folder, auto gives
     exactly the spectral result.
   - **`--denoiser voice`** (30 dB). [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet),
     a neural speech enhancer, on every track. It's much cleaner on speech
     and also handles non-steady noise like chatter, typing and traffic. It
     treats instruments as noise and strips them, so the script warns about
     tracks it removed more than 10 dB from.
   - **`--denoiser spectral`** (15 dB, `denoise.py`) on every track. Safe for any
     material, including instruments. A noise profile is learned per frequency
     from the track's quietest moments, such as room tone before or after
     playing and pauses between phrases. Narrow spectral peaks are kept out of
     the profile, so sustained notes aren't mistaken for noise (a note held
     through the whole file loses at most about 2 dB). The noise
     is then removed with a decision-directed Wiener filter (Ephraim–Malah),
     which avoids the "watery" artifacts of plain spectral subtraction.

   One blind spot in auto: a voice *with* an instrument at about the same
   level (a singer playing guitar, for example) counts as speech, and the
   voice model removes the guitar. Use `--denoiser spectral` for that.
4. Measure EBU R128 loudness and gain the track to −23 LUFS. The exception
   is someone in a call who never speaks: with auto or voice, a track with no
   speech that's more than 12 dB quieter than the people talking keeps its
   own level and gets the voice model's cleanup, which takes 20+ dB off a
   fan or typing. Otherwise their noise would be raised to full speaking
   volume; in a test call that made the mix noisier than not processing at
   all. A musician among speakers is about as loud as they are, so still gets
   balanced.
5. Gentle compression (3:1, threshold just above the track's average) so loud
   moments don't bury the other tracks (`--compress`).
6. Re-measure and correct the residual gain, so every track lands at the same loudness.

**Mix:** sum all tracks in 32-bit float, which can't clip. Gain the sum to `--lufs`
(default −16), then peak-limit to −1 dBFS. With `--pan`, each video's audio is
placed left-to-right by its gallery column.

**Video:** tiles every video into a centered grid. The grid is sized for the
biggest tiles and uses the median clip aspect ratio. A partial last row is
centered. The output runs as long as the longest input, video or audio. When
a clip ends earlier, its tile freezes on the last frame (or goes blank with
`--hold black`). HDR video (HLG, which iPhones film by default, or PQ/HDR10)
is tone-mapped to SDR. BT.2408 reference white maps to SDR white and brighter
highlights roll off smoothly; without this, iPhone clips come out washed out
and grey. Output is H.264 + AAC.

## Denoiser benchmark

SI-SDR in dB against the clean signal (higher is better). The noise is pink
hiss plus 60 Hz hum. Each denoiser runs at its default strength. "ffmpeg" is
ffmpeg's built-in `afftdn`, which the first version of this script used.

| signal | input SNR | noisy | ffmpeg | spectral | voice |
|---|---|---|---|---|---|
| speech with pauses | 5 dB | 7.6 | 8.1 | 11.2 | **14.0** |
| speech with pauses | 15 dB | 17.9 | 18.5 | 20.5 | **21.4** |
| music, 1 s room tone first | 5 dB | 7.7 | 8.0 | **10.8** | 1.8 |
| music, 1 s room tone first | 15 dB | 17.7 | 18.5 | **19.6** | −2.2 |
| music, never silent + constant drone | 5 dB | 7.6 | 7.8 | **8.0** | −3.7 |
| music, never silent + constant drone | 15 dB | 17.7 | **18.4** | 16.6 | −4.7 |

In speech pauses, the spectral denoiser removes about 9 dB of noise. ffmpeg's
`afftdn` removed 0.7 dB.

Notes on the spectral denoiser:
- **Never-quiet music** is its worst case. With no quiet moment to learn
  from, a constant drone is indistinguishable from noise, so leave a second of
  room tone on each clip if you can.
- **Mains hum** (narrow 50/60 Hz harmonics) is deliberately left in. It looks
  just like a sustained note, so the same rule that protects sustained notes
  keeps it.
- **Low, rumbly noise** (air conditioning, traffic, brown-ish room noise) is
  reduced less than `--denoise` asks for, because the profile errs low where
  the noise spectrum is steep: at the default 15 dB, white noise loses about
  14 dB, pink noise about 12 dB, brown noise 6–8 dB. Raising `--denoise` past
  ~20 dB doesn't help with these. For speech, auto already uses the voice
  denoiser, which does.

## Useful options

| option | default | |
|---|---|---|
| `--lufs` | -16 | final loudness (−14 streaming, −23 broadcast) |
| `--denoiser` | auto | `auto` (voice model on speech, spectral on the rest), `voice` (every track), `spectral` (every track) |
| `--denoise` | 30 / 15 | maximum noise reduction in dB for the voice / spectral denoiser; one value sets both; `0` = off |
| `--compress` | 3 | per-track ratio, `1` = off |
| `--pan` | 0 | stereo spread by grid column, 0–1 |
| `--size` | 1920x1080 | output resolution |
| `--cols` | auto | force grid columns |
| `--fit` | cover | `cover` crops to fill tiles, `contain` letterboxes |
| `--gap` / `--bg` | 8 / black | spacing and background color |
| `--labels` | off | file name on each tile (any script; emoji are dropped) |
| `--font` | auto | font file for the labels |
| `--max-duration` | — | cut the output at N seconds (handy for quick previews) |
| `--save-audio` | off | also write the mastered mix as a 24-bit WAV |
| `-r` | off | include subfolders |

Run `python3 gallery_mix.py -h` for everything.

Files are ordered by name, using natural sort (`take 2` comes before `take 10`),
and all start at t=0. Hidden files and folders are skipped, as is the
script's own output from an earlier run. The output must be `.mp4`, `.mov`,
`.m4v` or `.mkv`.

## Examples

`examples/renders/` has finished galleries made from synthetic but badly
recorded inputs (`examples/inputs/`): a home-recorded ensemble, a noisy video
call with both denoisers, a laptop/phone class call with silent participants
and an HDR iPhone clip, a phasing grid with tiles going dark, every aspect
ratio letterboxed, and a messy folder. See `examples/README.md`. Rebuild them
with `python3 examples/make_examples.py` (macOS, for the `say` voices).

## Tests

```bash
python3 -m unittest discover tests          # ~3 minutes, needs ffmpeg
GM_KEEP=1 python3 -m unittest discover tests   # keep the generated media
```

`tests/test_units.py` covers the denoiser DSP, the grid layout and probing.
`tests/test_pipeline.py` runs the script end to end on generated media:
40+ container/codec combinations, channel layouts and sample rates, loudness
targets, A/V sync (flash + click) through every processing path, the auto
denoiser on laptop/phone speech and on instruments, silent participants, HDR
tone mapping, tile placement, rotation, anamorphic video, panning, junk files
and bad arguments.
The voice-denoiser tests are skipped when `bin/deep-filter` is missing.
