"""Original procedurally-synthesised wedding music (no samples, no third-party recordings).

Video A: tanpura drone, shehnai melody (Khamaj-flavoured phrases), bansuri flute, sitar, tabla keherwa,
         dhol for the baraat, temple bells and manjira on every scene transition.
Video B: orchestral strings, choir pad, santoor/harp arpeggios, bansuri melody, light tabla,
         timpani swells, whooshes and celesta chimes synchronised to transitions and text reveals.
Video C: light and airy – bansuri in Bhupali (pentatonic), santoor arpeggios, soft string pad, gentle
         tabla and manjira (no dhol), and glass chimes on every cut and text reveal.
All timings come from the video Timeline so cuts land on musical accents.
"""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter

SR = 48000
BEAT = 0.5  # seconds per beat (120 bpm grid; keherwa matra)
DEG = {"S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "m": 5, "M": 6, "P": 7, "d": 8, "D": 9, "n": 10, "N": 11}


def hz(sa, name):
    oct_ = name.count("'") - name.count(".")
    return sa * 2 ** ((DEG[name.strip(".'")] + 12 * oct_) / 12)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR
        self.L = np.zeros(self.n, np.float64)
        self.R = np.zeros(self.n, np.float64)

    def add(self, t0, sig, gain=1.0, pan=0.0):
        i = int(t0 * SR)
        if i >= self.n or len(sig) == 0:
            return
        if i < 0:
            sig, i = sig[-i:], 0
        sig = sig[: self.n - i]
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.L[i:i + len(sig)] += sig * gain * gl * 1.414
        self.R[i:i + len(sig)] += sig * gain * gr * 1.414

    def mix(self, other, gain=1.0):
        self.L += other.L * gain
        self.R += other.R * gain


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


rng = np.random.default_rng(7)


def noise(n):
    return rng.standard_normal(n)


def onepole(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


# ---------------------------------------------------------------------------- instruments
def bell(f0=520.0, dur=4.0, bright=1.0):
    t = tvec(dur)
    parts = [(1.0, 1.0, 1.0), (2.0, 0.5, 0.8), (2.76, 0.7, 0.6), (3.3, 0.25, 0.5), (5.4, 0.35 * bright, 0.35),
             (6.8, 0.15 * bright, 0.25), (8.9, 0.1 * bright, 0.18)]
    s = sum(a * np.sin(2 * np.pi * f0 * r * t + r) * np.exp(-t / (dur * d * 0.45)) for r, a, d in parts)
    s *= 1 + 0.15 * np.sin(2 * np.pi * 3.2 * t)  # beating shimmer
    s[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return s / 2.5


def manjira(dur=1.8):
    t = tvec(dur)
    s = sum(np.sin(2 * np.pi * f * t + f) for f in (2350, 3120, 4410, 5230, 6650)) * np.exp(-t / 0.5)
    return s / 5 + noise(len(t)) * np.exp(-t / 0.03) * 0.15


def tanpura_pluck(f, dur=4.0):
    t = tvec(dur)
    out = np.zeros_like(t)
    for h in range(1, 22):
        if f * h > 12000:
            break
        bloom = 1 + 1.2 * np.sin(np.pi * np.clip(t / (0.6 + 0.12 * h), 0, 1))  # jawari brightness sweep
        out += (1 / h) * np.sin(2 * np.pi * f * h * t + h * 0.3) * np.exp(-t / (2.5 - 0.06 * h)) * bloom
    out *= np.minimum(1, t / 0.02)
    return out / 4


def pluck(f, dur=1.6, bright=1.0, decay=0.5, buzz=0.0):
    t = tvec(dur)
    out = np.zeros_like(t)
    bend = 1 + 0.01 * np.exp(-t / 0.05)
    for h in range(1, 16):
        if f * h > 14000:
            break
        a = (1 / h ** (1.3 - 0.4 * bright)) * np.exp(-t * h ** 0.6 / decay)
        out += a * np.sin(2 * np.pi * f * h * bend * t)
    if buzz:
        out = np.tanh(out * (1 + 4 * buzz)) / (1 + buzz)
    out *= np.minimum(1, t / 0.002)
    return out / 2


def tabla_dayan(f=293.7, dur=0.6, soft=False):
    t = tvec(dur)
    s = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2 * t) + 0.3 * np.sin(2 * np.pi * f * 3.01 * t)
         + 0.2 * np.sin(2 * np.pi * f * 4.2 * t)) * np.exp(-t / (0.12 if soft else 0.22))
    s += noise(len(t)) * np.exp(-t / 0.006) * 0.6
    return s / 2


def tabla_bayan(dur=0.8):
    t = tvec(dur)
    f = 95 * (1 + 0.35 * np.exp(-t / 0.08))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) + 0.2 * np.sin(2 * ph)) * np.exp(-t / 0.3) * np.minimum(1, t / 0.004)


def tabla_ka(dur=0.12):
    t = tvec(dur)
    return onepole(noise(len(t)), 2500) * np.exp(-t / 0.02) * 1.5


def dhol_boom(dur=0.7):
    t = tvec(dur)
    f = 62 * (1 + 0.6 * np.exp(-t / 0.05))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28) * np.minimum(1, t / 0.003)


def dhol_slap(dur=0.25):
    t = tvec(dur)
    s = np.sin(2 * np.pi * 430 * t) * np.exp(-t / 0.05) + onepole(noise(len(t)), 5000) * np.exp(-t / 0.03)
    return s * 0.7


def timpani(f=73.4, dur=2.5):
    t = tvec(dur)
    s = sum(a * np.sin(2 * np.pi * f * r * t) for r, a in ((1, 1), (1.5, 0.5), (1.98, 0.35), (2.44, 0.2)))
    return s * np.exp(-t / 0.9) * np.minimum(1, t / 0.005) / 1.8


def timpani_roll(f=73.4, dur=1.6):
    n = int(dur * SR)
    out = np.zeros(n + SR)
    k = 0.0
    while k < dur:
        a = (k / dur) ** 2
        s = timpani(f, 0.6) * (0.15 + 0.85 * a)
        i = int(k * SR)
        out[i:i + len(s)] += s[: len(out) - i]
        k += 0.045
    return out[:n]


def whoosh(dur=1.2, rise=True):
    t = tvec(dur)
    x = noise(len(t))
    env = np.sin(np.pi * t / dur) ** 2
    lo = onepole(x, 1800)
    hi = x - onepole(x, 600)
    mixk = t / dur if rise else 1 - t / dur
    return (lo * (1 - mixk) + onepole(hi, 6000) * mixk) * env * 0.5


def chime(f=1568.0, dur=2.5):
    t = tvec(dur)
    s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 4.0 * t) * np.exp(-t / 0.2))
    return s * np.exp(-t / 0.8) * np.minimum(1, t / 0.002) * 0.6


def boom(dur=3.0):
    t = tvec(dur)
    f = 45 * (1 + 0.8 * np.exp(-t / 0.15))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9) + onepole(noise(len(t)), 200) * np.exp(-t / 0.4)


def reed_or_flute(events, total, kind="shehnai", porta=0.06, vib_hz=5.5):
    """Continuous monophonic voice with meend (glides) and vibrato.
    events: list of (t0, dur, freq). Returns mono signal for the whole timeline."""
    n = int(total * SR) + SR
    f_tgt = np.zeros(n)
    amp = np.zeros(n)
    long_ = np.zeros(n)
    for t0, d, f in events:
        i0, i1 = int(t0 * SR), int((t0 + d) * SR)
        f_tgt[i0:i1] = f
        L = i1 - i0
        a = np.ones(L)
        att, rel = min(int(0.05 * SR), L // 3), min(int(0.08 * SR), L // 3)
        a[:att] = np.linspace(0.35, 1, att)
        a[L - rel:] = np.linspace(1, 0.55, rel)
        amp[i0:i1] = np.maximum(amp[i0:i1], a)
        ramp = np.clip((np.arange(L) / SR - 0.35) / 0.4, 0, 1)
        long_[i0:i1] = ramp
    # fill tiny gaps in frequency so glides go between notes
    idx = np.where(f_tgt > 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    f_fill = f_tgt[idx]
    f_fill[f_fill == 0] = events[0][2] if events else 440
    f = onepole(f_fill, 1 / (2 * np.pi * porta))
    amp = onepole(amp, 30)
    t = np.arange(n) / SR
    f = f * (1 + 0.007 * long_ * np.sin(2 * np.pi * vib_hz * t + 0.3 * np.sin(2 * np.pi * 0.4 * t)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(n)
    if kind == "shehnai":
        for h in range(1, 16):
            fh = f * h
            w = (1 / h ** 0.65) * (1 + 2.2 * np.exp(-((fh - 1300) / 450) ** 2) + 1.4 * np.exp(-((fh - 2800) / 700) ** 2))
            w *= fh < 15000
            out += w * np.sin(h * ph)
        out = out / 6 + onepole(noise(n), 3000) * 0.02
    else:  # bansuri
        out = np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
        breath = onepole(noise(n), 2500) - onepole(noise(n), 400)
        out = out / 1.3 + breath * 0.06
    return out * amp


def string_pad(notes, dur, attack=0.7, release=1.0, bright=0.6):
    """Ensemble strings: detuned additive sawtooth voices with slow envelope."""
    t = tvec(dur + release)
    out = np.zeros_like(t)
    env = np.minimum(1, t / attack) * np.where(t > dur, np.exp(-(t - dur) / (release * 0.4)), 1.0)
    for m in notes:
        f = midi_hz(m)
        for det in (-0.12, 0.0, 0.11):
            fd = f * 2 ** (det / 12)
            vib = 1 + 0.003 * np.sin(2 * np.pi * (5.0 + det) * t + m)
            ph = 2 * np.pi * fd * np.cumsum(vib) / SR
            for h in range(1, 10):
                if fd * h > 9000:
                    break
                out += (1 / h) * np.exp(-h * (1 - bright) * 0.5) * np.sin(h * ph + h * det * 10)
    return out * env / (len(notes) * 6)


def choir(notes, dur, attack=1.0):
    t = tvec(dur + 1.2)
    env = np.minimum(1, t / attack) * np.where(t > dur, np.exp(-(t - dur) / 0.5), 1.0)
    out = np.zeros_like(t)
    for m in notes:
        f = midi_hz(m)
        ph = 2 * np.pi * f * np.cumsum(1 + 0.004 * np.sin(2 * np.pi * 4.6 * t + m)) / SR
        for h in range(1, 14):
            fh = f * h
            if fh > 8000:
                break
            w = np.exp(-((fh - 700) / 260) ** 2) + 0.6 * np.exp(-((fh - 1150) / 300) ** 2) + 0.2 / h
            out += w * np.sin(h * ph)
    return out * env / (len(notes) * 3)


def reverb(L, R, wet=0.25, length=2.4, damp=0.45):
    t = tvec(length)
    irs = []
    for s in (1, 2):
        ir = np.random.default_rng(s).standard_normal(len(t)) * np.exp(-t / damp)
        ir = onepole(ir, 5000)
        ir[: int(0.012 * SR)] = 0
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    wl = fftconvolve(L, irs[0])[: len(L)]
    wr = fftconvolve(R, irs[1])[: len(R)]
    return L * (1 - wet) + wl * wet * 2.2, R * (1 - wet) + wr * wet * 2.2


def master(L, R, total, fade_in=1.2, fade_out=3.0):
    n = int(total * SR)
    L, R = L[:n], R[:n]
    t = np.arange(n) / SR
    g = np.minimum(1, t / fade_in) * np.minimum(1, np.maximum(0, (total - t) / fade_out))
    L, R = L * g, R * g
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-9)
    L, R = L / peak * 1.25, R / peak * 1.25
    L, R = np.tanh(L) * 0.93, np.tanh(R) * 0.93  # soft limiter
    return np.stack([L, R], 1)


# ---------------------------------------------------------------------------- composition helpers
PHRASES = [
    [("G", 1), ("m", 1), ("P", 2), ("D", 1), ("P", 1), ("m", 1), ("G", 1), ("R", 1), ("G", 1), ("m", 2), ("G", 1),
     ("R", 1), ("S", 2)],
    [("P", 1), ("D", 1), ("N", 1), ("S'", 3), ("N", 1), ("D", 1), ("P", 2), ("m", 1), ("P", 1), ("D", 1), ("n", 1),
     ("D", 1), ("P", 1)],
    [("S'", 2), ("N", 1), ("S'", 1), ("R'", 2), ("S'", 1), ("N", 1), ("D", 1), ("P", 1), ("D", 1), ("N", 1),
     ("S'", 4)],
    [("G", 1), ("G", 1), ("m", 1), ("P", 1), ("m", 2), ("G", 1), ("R", 1), ("S", 1), ("R", 1), ("G", 2), (".N", 1),
     (".D", 1), ("S", 2)],
]


def phrase_events(sa, start, order, beat=BEAT, end=None, legato=0.96, bank=None):
    ev, t = [], start
    for pi in order:
        for name, b in (bank or PHRASES)[pi]:
            if end is not None and t >= end:
                return ev
            d = b * beat
            if name != "-":
                ev.append((t, d * legato, hz(sa, name)))
            t += d
    return ev


def scene_windows(tl):
    """Global (start, end) for each scene's fully-visible span."""
    return [(s0, s0 + sc.duration) for s0, sc in zip(tl.starts, tl.scenes)]


def transition_mids(tl):
    return [tl.starts[i] - tl.scenes[i].td / 2 for i in range(1, len(tl.scenes))]


# ---------------------------------------------------------------------------- VIDEO A
def compose_traditional(tl, out_wav):
    total = tl.total
    SA = 293.66  # D4
    W = scene_windows(tl)
    mids = transition_mids(tl)
    grid0 = mids[0]  # beat grid starts on the first cut
    drone, perc, mel, fx = Track(total), Track(total), Track(total), Track(total)

    # tanpura cycle P S' S' S (every 4 s), whole piece
    t = 0.0
    while t < total:
        for k, (name, dt_) in enumerate(((".P", 0.0), ("S", 1.0), ("S", 1.6), (".S", 2.4))):
            f = hz(SA, name) / 2
            drone.add(t + dt_, tanpura_pluck(f, 4.5), 0.35, pan=-0.3 + 0.2 * k)
        t += 4.0

    # Ganesha invocation: temple bells synced to the animated bells, slow flute alaap
    for bt in (0.4, 3.6):
        fx.add(bt, bell(520, 5), 0.55, pan=-0.4)
        fx.add(bt + 0.5, bell(780, 4), 0.35, pan=0.4)
    alaap = [(0.8, 1.6, hz(SA * 2, "S")), (2.4, 0.8, hz(SA * 2, "R")), (3.2, 1.6, hz(SA * 2, "G")),
             (4.8, 0.6, hz(SA * 2, "m")), (5.4, 0.6, hz(SA * 2, "G")), (6.0, 0.8, hz(SA * 2, "R"))]
    mel.add(0, reed_or_flute(alaap, total, "bansuri"), 0.5, pan=0.1)

    # every scene cut: bell + manjira accent exactly at the transition midpoint
    for m in mids:
        fx.add(m, bell(660, 3.5, 0.8), 0.35, pan=0.3)
        fx.add(m, manjira(), 0.3, pan=-0.3)

    # tabla keherwa from the first cut to the final cadence (not during Gauna's soft section)
    baraat = W[6]
    gauna = W[8]
    end_perc = total - 3.0
    bol = ["dha", "ge", "na", "ti", "na", "ka", "dhi", "na"]
    k = 0
    t = grid0
    while t < end_perc:
        in_baraat = baraat[0] - 0.4 <= t < baraat[1] - 0.4
        in_gauna = gauna[0] - 0.4 <= t < gauna[1] - 0.4
        b = bol[k % 8]
        if in_baraat:  # dhol chaal: boom on 1 & 3, slaps on offbeats
            if k % 2 == 0:
                perc.add(t, dhol_boom(), 0.9)
            perc.add(t + BEAT / 2, dhol_slap(), 0.45, pan=0.2)
            if k % 4 == 3:
                perc.add(t + BEAT * 0.75, dhol_slap(), 0.35, pan=-0.2)
        elif not in_gauna:
            g = 0.55
            if b in ("dha", "dhi", "ge"):
                perc.add(t, tabla_bayan(), g * 0.9, pan=-0.15)
            if b in ("dha", "na", "dhi"):
                perc.add(t, tabla_dayan(SA), g * 0.7, pan=0.15)
            if b == "ti":
                perc.add(t, tabla_dayan(SA * 2, soft=True), g * 0.4, pan=0.15)
            if b == "ka":
                perc.add(t, tabla_ka(), g * 0.5)
            # little fill into each cut
            if any(abs((t + BEAT) - m) < 1e-3 for m in mids):
                perc.add(t + BEAT / 2, tabla_dayan(SA, soft=True), g * 0.5, pan=0.2)
                perc.add(t + BEAT * 0.75, tabla_dayan(SA, soft=True), g * 0.45, pan=0.2)
        t += BEAT
        k += 1

    # shehnai: phrases across the ceremony section
    shehnai_ev = phrase_events(SA * 2, grid0 + 0.0, [0, 1, 0, 3, 2, 1, 3, 0, 2, 1], end=gauna[0] - 0.4)
    shehnai_ev = [e for e in shehnai_ev if not (W[3][0] <= e[0] < W[5][1])]  # sitar takes Tilak..Haldi
    shehnai_ev += phrase_events(SA * 2, W[9][0] - 0.4, [3, 2], end=total - 4.5)
    shehnai_ev.append((total - 4.6, 3.6, hz(SA * 2, "S")))  # final long Sa
    mel.add(0, reed_or_flute(shehnai_ev, total, "shehnai"), 0.42, pan=0.05)

    # sitar: arpeggiated phrases for Tilak, Yagyopaveet, Haldi + melody
    t = W[3][0] - 0.4
    arp = ["S", "G", "P", "S'", "P", "G", "m", "R"]
    k = 0
    while t < W[5][1] - 0.4:
        nm = arp[k % 8]
        mel.add(t, pluck(hz(SA, nm), 1.4, bright=0.9, decay=0.45, buzz=0.35), 0.38, pan=-0.25)
        t += BEAT / 2
        k += 1
    sit_mel = phrase_events(SA * 2, W[3][0] - 0.4, [2, 3, 1], end=W[5][1] - 0.4)
    for t0, d, f in sit_mel:
        mel.add(t0, pluck(f, min(d + 0.8, 2.0), bright=1.0, decay=0.7, buzz=0.5), 0.42, pan=0.2)

    # Gauna: tender bansuri (bidaai feel)
    flute_ev = phrase_events(SA * 2, gauna[0] - 0.4, [3], beat=BEAT * 0.75, end=gauna[1] - 0.4)
    mel.add(0, reed_or_flute(flute_ev, total, "bansuri"), 0.45)

    # closing bells (synced with the closing scene's bell animation) and final chime
    close_local0 = tl.starts[9] - tl.scenes[9].td
    for bt in (0.3, 11.0):
        fx.add(close_local0 + bt, bell(520, 5), 0.5, pan=-0.4)
    fx.add(total - 4.6, bell(392, 6), 0.45)

    mix = Track(total)
    mix.mix(drone, 0.55)
    mix.mix(perc, 0.8)
    mix.mix(mel, 1.0)
    mix.mix(fx, 0.9)
    L, R = reverb(mix.L, mix.R, wet=0.22, length=2.6)
    wavfile.write(out_wav, SR, (master(L, R, total) * 32767).astype(np.int16))


# ---------------------------------------------------------------------------- VIDEO B
def compose_luxury(tl, out_wav):
    total = tl.total
    W = scene_windows(tl)
    mids = transition_mids(tl)
    grid0 = mids[0]
    strings, choirT, harp, mel, perc, fx = (Track(total) for _ in range(6))

    # D major orchestral progression, one chord per 2 s, aligned to the cut grid
    prog = [[50, 57, 62, 66, 69], [47, 54, 62, 66, 71], [43, 55, 62, 67, 71], [45, 57, 61, 64, 69],
            [40, 55, 59, 64, 67], [43, 55, 62, 67, 71], [45, 57, 61, 64, 69], [50, 57, 62, 66, 74]]
    # intro: sustained D pad under the door opening
    strings.add(0.0, string_pad([38, 50, 57, 62], grid0 + 0.5, attack=2.0, release=1.2, bright=0.4), 0.9)
    t, k = grid0, 0
    while t < total - 3.0:
        ch = prog[k % len(prog)]
        strings.add(t, string_pad(ch, 2.05, attack=0.35, release=0.9), 0.9)
        strings.add(t, string_pad([ch[0] - 12], 2.05, attack=0.2, release=0.6, bright=0.3), 0.7)  # basses
        # santoor/harp arpeggio, 8th notes, through the chord tones
        tones = sorted(ch[1:]) + [ch[2] + 12]
        for j in range(8):
            m = tones[[0, 1, 2, 3, 4, 3, 2, 1][j]] + 12
            harp.add(t + j * BEAT / 2, pluck(midi_hz(m), 1.6, bright=1.0, decay=0.35), 0.28,
                     pan=-0.5 + j / 8)
        t += 2.0
        k += 1
    strings.add(total - 3.2, string_pad([38, 50, 57, 62, 66, 69], 2.6, attack=0.4, release=1.6), 1.0)

    # choir on the couple, vivaah and closing scenes
    for si in (2, 7, 9):
        s0, s1 = W[si]
        choirT.add(s0 - 0.4, choir([62, 66, 69, 74], s1 - s0 - 0.6, attack=1.5), 0.6)

    # bansuri melody (same raga phrases as video A for family resemblance), Sa = D5
    ev = phrase_events(587.33, grid0 + 4.0, [0, 1, 2, 3, 0, 1, 2, 3, 1, 2], end=total - 6)
    ev.append((total - 5.6, 3.0, 587.33))
    mel.add(0, reed_or_flute(ev, total, "bansuri", porta=0.05, vib_hz=5.0), 0.5, pan=0.1)

    # light tabla dadra-feel in ceremony scenes
    t, k = grid0, 0
    while t < total - 3.0:
        si = max(i for i, (a, _) in enumerate(W) if a - 0.4 <= t) if t >= W[0][0] else 0
        if 3 <= si <= 8 and si != 8:
            if k % 4 == 0:
                perc.add(t, tabla_bayan(), 0.35)
            perc.add(t, tabla_dayan(293.66, soft=k % 2 == 1), 0.25, pan=0.2)
            if si == 6:  # baraat: add dhol energy
                if k % 2 == 0:
                    perc.add(t, dhol_boom(), 0.6)
                perc.add(t + BEAT / 2, dhol_slap(), 0.3)
        t += BEAT
        k += 1

    # cinematic punctuation
    fx.add(0.45, boom(), 0.7)                      # doors start to open
    fx.add(0.4, whoosh(2.4, rise=True), 0.4)
    fx.add(2.4, chime(1174.7, 3), 0.4)
    for m in mids:
        fx.add(m - 1.5, timpani_roll(73.4, 1.5), 0.35)
        fx.add(m, timpani(73.4, 2.5), 0.6)
        fx.add(m - 0.5, whoosh(1.0, rise=True), 0.45)
        fx.add(m, chime(1760.0, 2.5), 0.25, pan=0.4)
    # sparkle chimes on text reveals (pentatonic, quiet)
    pent = [74, 76, 78, 81, 83, 86]
    for i, (kind, et, _) in enumerate(tl.events()):
        if kind == "text":
            fx.add(et, chime(midi_hz(pent[i % len(pent)] + 12), 1.5), 0.07, pan=(-0.5 + (i % 5) / 4))
    fx.add(total - 3.2, timpani(73.4, 3), 0.6)
    fx.add(total - 3.2, chime(1174.7, 3.5), 0.4)

    mix = Track(total)
    mix.mix(strings, 0.85)
    mix.mix(choirT, 0.7)
    mix.mix(harp, 0.8)
    mix.mix(mel, 1.0)
    mix.mix(perc, 0.7)
    mix.mix(fx, 0.85)
    L, R = reverb(mix.L, mix.R, wet=0.32, length=3.2, damp=0.7)
    wavfile.write(out_wav, SR, (master(L, R, total, fade_in=0.4) * 32767).astype(np.int16))


# ---------------------------------------------------------------------------- VIDEO C
BHUPALI = [  # S R G P D only – bright, festive pentatonic; 16 beats each
    [("G", 1), ("R", 1), ("S", 2), ("R", 1), ("G", 1), ("P", 2), ("D", 1), ("P", 1), ("G", 1), ("R", 1), ("G", 2),
     ("R", 1), ("S", 1)],
    [("P", 1), ("D", 1), ("S'", 2), ("D", 1), ("P", 1), ("G", 2), ("R", 1), ("G", 1), ("P", 1), ("D", 1), ("P", 2),
     ("G", 1), ("R", 1)],
    [("S'", 1), ("R'", 1), ("G'", 2), ("R'", 1), ("S'", 1), ("D", 2), ("P", 1), ("D", 1), ("S'", 1), ("D", 1),
     ("P", 2), ("G", 1), ("P", 1)],
    [("G", 1), ("P", 1), ("D", 1), ("P", 1), ("G", 2), ("R", 1), ("S", 1), (".D", 1), ("S", 1), ("R", 1), ("G", 2),
     ("R", 1), ("S", 2)],
]


def compose_floral(tl, out_wav):
    total = tl.total
    SA = 392.0  # G4
    W = scene_windows(tl)
    mids = transition_mids(tl)
    grid0 = mids[0]
    pad, santoor, mel, perc, fx = (Track(total) for _ in range(5))

    # soft string pad, G – Em – C – D – G – C – D – G, one chord per 2 s on the cut grid
    prog = [[43, 55, 59, 62, 67], [40, 52, 59, 64, 67], [36, 48, 55, 60, 64], [38, 50, 57, 62, 66],
            [43, 55, 59, 62, 67], [36, 48, 55, 60, 64], [38, 50, 57, 62, 66], [43, 55, 59, 62, 71]]
    pad.add(0.0, string_pad([55, 59, 62, 67], grid0 + 0.5, attack=1.8, release=1.0, bright=0.35), 0.6)
    t, k = grid0, 0
    while t < total - 3.0:
        ch = prog[k % len(prog)]
        pad.add(t, string_pad(ch[1:], 2.05, attack=0.5, release=0.9, bright=0.35), 0.55)
        # santoor: rippling 8ths through the chord tones, an octave up
        tones = sorted(ch[1:]) + [ch[1] + 12]
        for j in range(8):
            m = tones[[0, 2, 1, 3, 2, 4, 3, 1][j]] + 12
            santoor.add(t + j * BEAT / 2, pluck(midi_hz(m), 1.3, bright=1.2, decay=0.3), 0.24,
                        pan=-0.45 + 0.9 * (j % 4) / 3)
        t += 2.0
        k += 1
    pad.add(total - 3.2, string_pad([43, 55, 59, 62, 67, 71], 2.8, attack=0.4, release=1.8, bright=0.35), 0.8)

    # opening: music-box santoor glissando while the toran drops, then a slow flute alaap
    for j, m in enumerate([67, 69, 71, 74, 76, 79, 81, 83]):
        santoor.add(0.25 + j * 0.11, pluck(midi_hz(m + 12), 1.8, bright=1.3, decay=0.4), 0.22, pan=-0.6 + j * 0.15)
    alaap = [(1.2, 1.4, hz(SA * 2, "S")), (2.6, 0.7, hz(SA * 2, "R")), (3.3, 1.4, hz(SA * 2, "G")),
             (4.7, 0.6, hz(SA * 2, "P")), (5.3, 0.9, hz(SA * 2, "G"))]
    mel.add(0, reed_or_flute(alaap, total, "bansuri", vib_hz=5.0), 0.42, pan=0.1)

    # bansuri melody over the rest of the film (rests during Gauna, then the closing phrase)
    gauna = W[8]
    ev = phrase_events(SA * 2, grid0, [0, 1, 3, 2, 0, 1, 2, 3, 1, 0], end=gauna[0] - 0.4, bank=BHUPALI)
    ev += phrase_events(SA * 2, W[9][0] - 0.4, [1, 3], end=total - 4.5, bank=BHUPALI)
    ev.append((total - 4.6, 3.6, hz(SA * 2, "S")))
    mel.add(0, reed_or_flute(ev, total, "bansuri", porta=0.05, vib_hz=5.0), 0.45, pan=0.05)
    # Gauna: santoor alone carries a slower phrase
    for t0, d, f in phrase_events(SA * 2, gauna[0] - 0.4, [3], beat=BEAT * 0.75, end=gauna[1] - 0.4, bank=BHUPALI):
        mel.add(t0, pluck(f, min(d + 1.0, 2.2), bright=1.1, decay=0.6), 0.35, pan=-0.15)

    # gentle percussion: soft tabla on 1 and 3, manjira every 2 beats; a livelier pattern for Baraat
    baraat = W[6]
    t, k = grid0, 0
    while t < total - 3.0:
        in_baraat = baraat[0] - 0.4 <= t < baraat[1] - 0.4
        in_gauna = gauna[0] - 0.4 <= t < gauna[1] - 0.4
        if not in_gauna:
            if k % 4 == 0:
                perc.add(t, tabla_bayan(), 0.3, pan=-0.1)
            if k % 2 == 0:
                perc.add(t, tabla_dayan(SA, soft=True), 0.28, pan=0.15)
                perc.add(t, manjira(1.0), 0.08, pan=-0.35)
            if in_baraat:
                perc.add(t + BEAT / 2, tabla_dayan(SA, soft=True), 0.24, pan=0.15)
                perc.add(t + BEAT * 0.75, tabla_ka(), 0.22)
        t += BEAT
        k += 1

    # chimes: a bright pentatonic sweep into every cut, quiet sparkles on text reveals
    pent = [67, 69, 71, 74, 76, 79]
    for m in mids:
        for j in range(5):
            fx.add(m - 0.3 + j * 0.06, chime(midi_hz(pent[j] + 12), 1.8), 0.09, pan=-0.4 + j * 0.2)
        fx.add(m, chime(midi_hz(91), 2.4), 0.12, pan=0.3)
    for i, (kind, et, _) in enumerate(tl.events()):
        if kind == "text":
            fx.add(et, chime(midi_hz(pent[i % len(pent)] + 24), 1.2), 0.05, pan=(-0.5 + (i % 5) / 4))
    fx.add(total - 4.6, chime(midi_hz(79), 4.0), 0.3)
    fx.add(total - 4.6, bell(784, 5, 0.6), 0.2)

    mix = Track(total)
    mix.mix(pad, 0.7)
    mix.mix(santoor, 0.85)
    mix.mix(mel, 1.0)
    mix.mix(perc, 0.7)
    mix.mix(fx, 0.9)
    L, R = reverb(mix.L, mix.R, wet=0.3, length=2.8, damp=0.6)
    wavfile.write(out_wav, SR, (master(L, R, total, fade_in=0.3) * 32767).astype(np.int16))
