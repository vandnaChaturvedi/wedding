"""Procedural animation: easing, particles, text reveals, swaying, transitions.

Every effect is a pure function of time `t` so frames can be rendered in any order / in parallel.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from compositor import H, W, ken_burns, place
import illustrations as I


def spark_sprite_cache(color):
    return I.spark_sprite(48, color)


# ----------------------------------------------------------------------------- easing
def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.4):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def prog(t, start, dur):
    return clamp01((t - start) / max(dur, 1e-6))


# ----------------------------------------------------------------------------- particles
class PetalField:
    """Falling petals tumbling in 3D (y-scale flips) with sway."""

    def __init__(self, kind="mixed", n=34, seed=1, speed=(90, 170), size=(0.5, 1.0), start_t=0.0):
        rng = np.random.default_rng(seed)
        self.sprites = I.petal_sprites(kind)
        self.n = n
        self.x = rng.uniform(-40, W + 40, n)
        self.y0 = rng.uniform(-H, H, n)
        self.v = rng.uniform(*speed, n)
        self.amp = rng.uniform(20, 70, n)
        self.f = rng.uniform(0.2, 0.5, n)
        self.ph = rng.uniform(0, math.tau, n)
        self.rot0 = rng.uniform(0, 360, n)
        self.rv = rng.uniform(-90, 90, n)
        self.flip = rng.uniform(0.4, 1.2, n)
        self.s = rng.uniform(*size, n)
        self.idx = rng.integers(0, len(self.sprites), n)
        self.start_t = start_t

    def draw(self, frame, t, alpha=1.0):
        tt = t - self.start_t
        span = H + 240
        for i in range(self.n):
            y = (self.y0[i] + self.v[i] * tt) % span - 120
            x = self.x[i] + self.amp[i] * math.sin(math.tau * self.f[i] * tt + self.ph[i])
            fl = math.cos(math.tau * self.flip[i] * tt * 0.5 + self.ph[i])
            place(frame, self.sprites[self.idx[i]], x, y, scale=self.s[i], rot=self.rot0[i] + self.rv[i] * tt,
                  sy=max(0.15, abs(fl)), alpha=alpha * 0.95)


class Sparkles:
    """Rising, twinkling gold dust (additive)."""

    def __init__(self, n=70, seed=2, region=(0, 0, W, H), color="#fff1c1", size=(0.25, 0.9), rise=(15, 50)):
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = region
        self.region = region
        self.n = n
        self.x = rng.uniform(x0, x1, n)
        self.y = rng.uniform(y0, y1, n)
        self.v = rng.uniform(*rise, n)
        self.tw = rng.uniform(0.6, 2.2, n)
        self.ph = rng.uniform(0, math.tau, n)
        self.s = rng.uniform(*size, n)
        self.drift = rng.uniform(-12, 12, n)
        self.spr = spark_sprite_cache(color)
        self.star = I.star_sprite(64, color)
        self.is_star = rng.random(n) < 0.12

    def draw(self, frame, t, alpha=1.0):
        x0, y0, x1, y1 = self.region
        hgt = y1 - y0
        for i in range(self.n):
            y = y0 + (self.y[i] - y0 - self.v[i] * t) % hgt
            x = self.x[i] + self.drift[i] * math.sin(t * 0.7 + self.ph[i])
            a = (0.5 + 0.5 * math.sin(math.tau * self.tw[i] * t + self.ph[i])) ** 2
            spr = self.star if self.is_star[i] else self.spr
            place(frame, spr, x, y, scale=self.s[i] * (1.3 if self.is_star[i] else 1), alpha=alpha * a, add=True,
                  rot=t * 20 if self.is_star[i] else 0)


class Bokeh:
    """Large soft out-of-focus lights drifting slowly (additive)."""

    def __init__(self, n=16, seed=3, colors=("#f3d98b", "#e7b7a3", "#ffffff"), size=(0.5, 1.6)):
        rng = np.random.default_rng(seed)
        self.n = n
        self.x = rng.uniform(0, W, n)
        self.y = rng.uniform(0, H, n)
        self.vx = rng.uniform(-10, 10, n)
        self.vy = rng.uniform(-22, -6, n)
        self.s = rng.uniform(*size, n)
        self.ph = rng.uniform(0, math.tau, n)
        self.spr = [I.bokeh_sprite(160, c) for c in colors]
        self.c = rng.integers(0, len(colors), n)

    def draw(self, frame, t, alpha=1.0):
        for i in range(self.n):
            x = (self.x[i] + self.vx[i] * t) % (W + 200) - 100
            y = (self.y[i] + self.vy[i] * t) % (H + 200) - 100
            a = 0.55 + 0.45 * math.sin(t * 0.8 + self.ph[i])
            place(frame, self.spr[self.c[i]], x, y, scale=self.s[i], alpha=alpha * a * 0.42, add=True)


class Fireworks:
    """Gentle sparkle bursts for festive night scenes (additive)."""

    def __init__(self, seed=9, region=(80, 180, W - 80, 700), every=1.3, colors=("#ffd27a", "#ff8fab", "#fff1c1")):
        self.rng = np.random.default_rng(seed)
        self.region, self.every = region, every
        self.bursts = []
        x0, y0, x1, y1 = region
        for k in range(40):
            self.bursts.append((k * every + self.rng.uniform(0, every * 0.6), self.rng.uniform(x0, x1),
                                self.rng.uniform(y0, y1), colors[k % len(colors)], self.rng.uniform(0, 1)))
        self.cache = {c: spark_sprite_cache(c) for c in colors}

    def draw(self, frame, t, alpha=1.0):
        for t0, x, y, c, ph in self.bursts:
            age = t - t0
            if 0 <= age < 1.6:
                r = 160 * ease_out(age / 1.2)
                a = (1 - age / 1.6) ** 1.5
                for j in range(18):
                    ang = math.tau * j / 18 + ph
                    place(frame, self.cache[c], x + math.cos(ang) * r, y + math.sin(ang) * r + 30 * age ** 2,
                          scale=0.45, alpha=a * alpha, add=True)


# ----------------------------------------------------------------------------- flames & sway
def draw_flame(frame, x, y, t, scale=1.0, seed=0, glow=True, alpha=1.0):
    frames = I.flame_frames()
    k = int((t * 18 + seed * 5)) % len(frames)
    flick = 1 + 0.06 * math.sin(t * 23 + seed) + 0.04 * math.sin(t * 37 + seed * 2)
    if glow:
        place(frame, glow_sprite(), x, y - 40 * scale, scale=scale * flick * 1.2, alpha=0.5 * alpha, add=True)
    place(frame, frames[k], x, y, scale=scale * flick, anchor=(0.5, 0.95), alpha=alpha)


_GLOW = None


def glow_sprite():
    global _GLOW
    if _GLOW is None:
        _GLOW = I.spark_sprite(260, "#ffb347")
    return _GLOW


def bell_angle(t, rings=(0.0,), amp=9.0, freq=0.9, decay=0.35):
    a = 2.0 * math.sin(t * 1.3)
    for r in rings:
        if t >= r:
            a += amp * math.sin(math.tau * freq * (t - r)) * math.exp(-(t - r) * decay)
    return a


def sway_sprite(spr, t, amp=6.0, freq=0.35, axis_top=True):
    """Wind sway: horizontal displacement increasing toward the hanging end."""
    h, w = spr.shape[:2]
    ys = np.arange(h, dtype=np.float32)
    k = (ys / h) if axis_top else (1 - ys / h)
    shift = amp * k ** 1.5 * math.sin(math.tau * freq * t) + amp * 0.3 * k * np.sin(ys / 60 + t * 2.1)
    mapx = (np.arange(w, dtype=np.float32)[None, :] - shift[:, None]).astype(np.float32)
    mapy = np.repeat(ys[:, None], w, 1)
    return cv2.remap(spr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ----------------------------------------------------------------------------- text reveals
def wipe(spr, p, soft=0.18):
    """Left-to-right soft reveal (keeps Devanagari clusters intact)."""
    if p >= 1:
        return spr
    w = spr.shape[1]
    xs = np.linspace(0, 1, w, dtype=np.float32)
    m = np.clip((p * (1 + soft) - xs) / soft, 0, 1)
    return spr * m[None, :, None]


def shine(spr, p, width=0.12, strength=0.7):
    """Diagonal specular sweep across a sprite (additive highlight masked by alpha)."""
    if p <= 0 or p >= 1:
        return spr
    h, w = spr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / w) + (yy / h) * 0.35
    c = -0.2 + p * 1.6
    band = np.exp(-((u - c) / width) ** 2) * strength
    out = spr.copy()
    out[..., :3] += band[..., None] * spr[..., 3:4]
    return out


class TextItem:
    def __init__(self, spr, x, y, start, style="rise", dur=0.9, end=None, shine_at=None, anchor=(0.5, 0.0)):
        self.spr, self.x, self.y, self.start, self.style, self.dur = spr, x, y, start, style, dur
        self.end, self.shine_at, self.anchor = end, shine_at, anchor

    def bbox(self):
        h, w = self.spr.shape[:2]
        x0 = self.x - self.anchor[0] * w
        y0 = self.y - self.anchor[1] * h
        return (x0, y0, x0 + w, y0 + h)

    def draw(self, frame, t, alpha=1.0):
        p = prog(t, self.start, self.dur)
        if p <= 0:
            return
        a = alpha
        if self.end is not None:
            a *= 1 - prog(t, self.end, 0.6)
            if a <= 0:
                return
        spr, dy, sc = self.spr, 0.0, 1.0
        if self.style == "rise":
            a *= ease_out(p)
            dy = 34 * (1 - ease_out(p))
        elif self.style == "wipe":
            spr = wipe(spr, ease_in_out(p))
        elif self.style == "zoom":
            a *= ease_out(p)
            sc = 1.18 - 0.18 * ease_out(p)
        elif self.style == "fade":
            a *= ease_in_out(p)
        elif self.style == "pop":
            a *= ease_out(p * 1.5)
            sc = 0.6 + 0.4 * ease_out_back(p)
        if self.shine_at is not None:
            spr = shine(spr, prog(t, self.shine_at, 1.4))
        place(frame, spr, self.x, self.y + dy, scale=sc, alpha=a, anchor=self.anchor)


# ----------------------------------------------------------------------------- transitions
def transition(kind, a, b, p):
    """Blend frame a (outgoing) into frame b (incoming) with progress p in [0, 1]."""
    e = ease_in_out(p)
    if kind == "fade":
        return a * (1 - e) + b * e
    if kind == "slide":
        off = int(W * e)
        out = np.empty_like(a)
        par = int(off * 0.35)  # outgoing moves slower (parallax)
        out[:, :W - par] = a[:, par:]
        out[:, W - par:] = a[:, W - par:]  # always covered by b below (off >= par)
        if off:
            out[:, W - off:] = b[:, :off]
            # soft shadow at the leading edge
            sh = min(60, W - off)
            if sh > 0:
                ramp = np.linspace(0.0, 0.45, sh, dtype=np.float32)[None, :, None]
                out[:, W - off - sh:W - off] *= 1 - ramp
        return out
    if kind == "iris":
        yy, xx = _grid()
        r = e * 1250
        d = np.sqrt((xx - W / 2) ** 2 + (yy - H * 0.45) ** 2)
        m = np.clip((r - d) / 70.0, 0, 1)[..., None]
        out = a * (1 - m) + b * m
        ring = np.exp(-((d - r) / 18.0) ** 2)[..., None] * (1 - e) * 0.9
        out += ring * np.array([1.0, 0.85, 0.45], np.float32)
        return out
    if kind == "flash":
        g = math.sin(math.pi * p) ** 2
        base = a * (1 - e) + b * e
        return base + g * 0.85 * np.array([1.0, 0.9, 0.7], np.float32) * (1 - base * 0.4)
    if kind == "wipe":
        yy, xx = _grid()
        u = (xx / W + (yy / H) * 0.6) / 1.6
        m = np.clip((e * 1.3 - u) / 0.3, 0, 1)[..., None]
        return a * (1 - m) + b * m
    if kind == "zoom":
        za = 1 + 0.25 * e
        aa = ken_burns(a, za)
        bb = ken_burns(b, 1.25 - 0.25 * e)
        return aa * (1 - e) + bb * e
    raise ValueError(kind)


_G = None


def _grid():
    global _G
    if _G is None:
        _G = np.mgrid[0:H, 0:W].astype(np.float32)
    return _G
