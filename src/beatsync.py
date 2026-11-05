"""Use an existing song as the soundtrack, with every scene cut landing on a downbeat.

analyse(song)            -> tempo, beat grid, downbeat phase and where the music actually ends
plan_durations(...)      -> scene durations snapped to whole bars, filling the song's length
prepare_audio(...)       -> the song trimmed to the video with a gentle fade-in and a long fade-out
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from audio import SR
from video_export import FFMPEG


def decode(song, out_wav):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(song), "-map", "0:a:0", "-ar", str(SR), "-ac", "2",
                    str(out_wav)], check=True)
    sr, x = wavfile.read(out_wav)
    return x.astype(np.float32) / 32768.0


def analyse(x):
    mono = x.mean(1)
    hop = 256
    f, _, Z = stft(mono, SR, nperseg=2048, noverlap=2048 - hop)
    S = np.log1p(np.abs(Z) * 100)
    flux = np.maximum(0, np.diff(S, axis=1)).sum(0)
    flux = (flux - flux.mean()) / flux.std()
    fps = SR / hop
    dur = len(mono) / SR
    env = np.array([np.sqrt((mono[i:i + SR // 10] ** 2).mean()) for i in range(0, len(mono), SR // 10)])
    loud = np.nonzero(env > 0.01)[0]
    start, end = loud.min() / 10, loud.max() / 10
    t0, t1 = start + 3, min(end - 3, dur - 1)

    def score(bpm, step=0.005):
        per = 60 / bpm
        best = (-1e9, 0.0)
        for ph in np.arange(0, per, step):
            ts = np.arange(t0 + ph, t1, per)
            v = flux[(ts * fps).astype(int)].mean()
            if v > best[0]:
                best = (v, ph)
        return best

    coarse = max((score(b, 0.01)[0], b) for b in np.arange(90, 180, 0.5))[1]
    bpm = max((score(b)[0], b) for b in np.arange(coarse - 0.6, coarse + 0.6, 0.05))[1]
    per = 60 / bpm
    ph = score(bpm)[1]
    b0 = (t0 + ph) % per
    beats = np.arange(b0, dur, per)
    acc = np.array([flux[max(0, i - 3):i + 4].max() if i < len(flux) else 0 for i in (beats * fps).astype(int)])
    k = int(np.argmax([acc[j::4].mean() for j in range(4)]))  # strongest beat of the 4-cycle = downbeat
    downbeats = beats[k::4]
    return dict(bpm=float(bpm), beat=per, bar=4 * per, beats=beats, downbeats=downbeats, start=start, end=end)


def plan_durations(scenes, grid, extra_priority=(9, 2, 7, 9, 1, 3, 6)):
    """Durations so that each cut's midpoint (where the transition is half done) sits on a downbeat."""
    bar = grid["bar"]
    td = scenes[1].td
    first_mid = scenes[0].duration - td / 2
    D = grid["downbeats"]
    m0 = D[D >= first_mid - 0.15][0]
    durs = [m0 + td / 2] + [max(3, round(s.duration / bar)) * bar for s in scenes[1:]]
    limit = grid["end"] - 0.5
    i = 0
    while sum(durs) + bar <= limit and i < 50:
        durs[extra_priority[i % len(extra_priority)]] += bar
        i += 1
    return durs


def prepare_audio(x, total, out_wav, fade_in=1.5, fade_out=5.0):
    n = int(round(total * SR))
    y = np.zeros((n, 2), np.float32)
    y[:min(n, len(x))] = x[:n]
    a = int(fade_in * SR)
    y[:a] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a)))[:, None]
    b = int(fade_out * SR)
    y[-b:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b)))[:, None]
    wavfile.write(out_wav, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
