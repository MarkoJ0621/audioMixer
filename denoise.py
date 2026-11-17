"""
Spectral noise reduction for recordings of any length (speech or music).

1. Learn a noise profile per channel and frequency bin from the quietest
   moments of the recording: a low percentile of STFT power over time,
   bias-corrected, so it works even without a silent lead-in.
2. Filter with a decision-directed Wiener gain (Ephraim & Malah), which keeps
   "musical noise" artifacts low, smoothed across frequency and floored at the
   requested maximum reduction.

Audio is streamed through memory-mapped raw float32 files (interleaved), so an
hour-long input never has to fit in RAM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_FFT = 2048             # 43 ms at 48 kHz
HOP = N_FFT // 4
PAD = N_FFT - HOP
PERCENTILE = 10          # per-bin percentile of power used as the noise estimate
QUIET_FRAMES = 5         # percent of quietest frames averaged as a noise sample
PEAK_WINDOW = 9          # bins (~210 Hz) for the spectral floor that tonal peaks are capped to
ALPHA = 0.98             # decision-directed smoothing: higher = fewer artifacts
PROFILE_FRAMES = 6000    # frames sampled (spread over the file) to learn the profile
RUN = 64                 # ...taken in contiguous runs of this many frames
BLOCK = 1024             # frames per vectorized processing block

_WIN = np.sqrt(np.hanning(N_FFT + 1)[:-1]).astype(np.float32)   # periodic sqrt-Hann
_OLA_NORM = sum((_WIN ** 2)[j * HOP:(j + 1) * HOP] for j in range(N_FFT // HOP))


def _stft(x: np.ndarray, f0: int, f1: int) -> np.ndarray:
    """STFT frames [f0, f1) of x (channels x samples); frame k starts at k*HOP - PAD."""
    start, stop = f0 * HOP - PAD, f1 * HOP
    seg = np.zeros((x.shape[0], stop - start), np.float32)
    lo, hi = max(start, 0), min(stop, x.shape[1])
    if hi > lo:
        seg[:, lo - start:hi - start] = x[:, lo:hi]
    frames = np.lib.stride_tricks.sliding_window_view(seg, N_FFT, axis=1)[:, ::HOP]
    return np.fft.rfft(frames * _WIN, axis=-1)


def _noise_psd(x: np.ndarray, n_frames: int) -> np.ndarray | None:
    """Noise power per (channel, bin). Errs low: leftover noise beats eaten notes."""
    if n_frames <= PROFILE_FRAMES:
        spans = [(f, min(f + BLOCK, n_frames)) for f in range(0, n_frames, BLOCK)]
    else:   # first and last runs always included: lead-in/out room tone is the best sample
        starts = np.linspace(0, n_frames - RUN, PROFILE_FRAMES // RUN).astype(int)
        spans = [(s, s + RUN) for s in starts]
    power = np.concatenate([np.abs(_stft(x, a, b)) ** 2 for a, b in spans], axis=1)
    power = power[:, power.sum(axis=(0, 2)) > 1e-12]    # ignore digital silence
    if power.shape[1] == 0:
        return None

    # (a) average spectrum of the quietest frames: exact when the recording has
    #     any room tone (lead-in, pauses), too high when it never goes quiet
    energy = power[:, :, 4:342].sum(axis=(0, 2))       # ~100 Hz - 8 kHz
    quiet = power[:, energy <= np.percentile(energy, QUIET_FRAMES)].mean(axis=1)
    # (b) low percentile per bin: stationary noise power is exponentially
    #     distributed, so its p-th percentile sits at -ln(1 - p) times the mean;
    #     too high in bins a sustained note occupies most of the time
    per_bin = np.percentile(power, PERCENTILE, axis=1) / -np.log1p(-PERCENTILE / 100)
    lam = np.minimum(quiet, per_bin)
    # (c) noise is spectrally smooth, sustained notes are narrow peaks: cap the
    #     profile at 3 dB over its local spectral floor
    half = PEAK_WINDOW // 2
    padded = np.pad(lam, ((0, 0), (half, half)), mode="edge")
    floor = np.percentile(np.lib.stride_tricks.sliding_window_view(padded, PEAK_WINDOW, axis=1),
                          30, axis=-1)
    return np.minimum(lam, 2 * floor)


def denoise(src: Path, dst: Path, channels: int, max_reduction_db: float) -> None:
    """Denoise raw interleaved float32 audio from `src` into `dst`."""
    if src.stat().st_size == 0:      # memmap can't map an empty file
        dst.write_bytes(b"")
        return
    x = np.memmap(src, np.float32, "r").reshape(-1, channels).T
    n = x.shape[1]
    y = np.memmap(dst, np.float32, "w+", shape=(n, channels)).T
    n_frames = (n - 1 + PAD) // HOP + 1
    lam = _noise_psd(x, n_frames)
    if lam is None:      # nothing but digital silence
        return
    lam = np.maximum(lam, 1e-20).astype(np.float32)

    floor = np.float32(10 ** (-max_reduction_db / 20))
    g_prev = np.ones_like(lam)
    gamma_prev = np.ones_like(lam)
    for f0 in range(0, n_frames, BLOCK):
        f1 = min(f0 + BLOCK, n_frames)
        spec = _stft(x, f0, f1)                           # (channels, frames, bins)
        gamma = (spec.real ** 2 + spec.imag ** 2) / lam[:, None, :]
        gains = np.empty_like(gamma)
        for b in range(f1 - f0):
            xi = ALPHA * g_prev ** 2 * gamma_prev + (1 - ALPHA) * np.maximum(gamma[:, b] - 1, 0)
            g_prev = np.maximum(xi / (1 + xi), floor)
            gamma_prev = gamma[:, b]
            gains[:, b] = g_prev
        gains[..., 1:-1] = 0.25 * gains[..., :-2] + 0.5 * gains[..., 1:-1] + 0.25 * gains[..., 2:]

        frames = np.fft.irfft(spec * gains, n=N_FFT, axis=-1).astype(np.float32) * _WIN
        rows = np.zeros((channels, f1 - f0 + N_FFT // HOP - 1, HOP), np.float32)
        for j in range(N_FFT // HOP):
            rows[:, j:j + f1 - f0] += frames[:, :, j * HOP:(j + 1) * HOP]
        out = (rows / _OLA_NORM).reshape(channels, -1)    # samples [f0*HOP - PAD, f1*HOP)
        start = f0 * HOP - PAD
        lo, hi = max(start, 0), min(f1 * HOP, n)
        y[:, lo:hi] += out[:, lo - start:hi - start]
    y.base.flush()
