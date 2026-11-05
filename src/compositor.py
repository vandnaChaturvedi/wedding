"""Low-level frame compositing.

Frames are float32 RGB arrays in [0, 1], shape (H, W, 3).
Sprites are float32 *premultiplied* RGBA arrays in [0, 1], shape (h, w, 4).
"""
from __future__ import annotations

import math

import cairocffi as cairo
import cv2
import numpy as np

W, H = 1080, 1920
FPS = 30


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# --------------------------------------------------------------------------- cairo bridge
def cairo_canvas(w: int, h: int):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(w), int(h))
    return surf, cairo.Context(surf)


def surface_to_sprite(surf) -> np.ndarray:
    surf.flush()
    w, h, stride = surf.get_width(), surf.get_height(), surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(h, stride // 4, 4)[:, :w]
    # cairo ARGB32 is premultiplied BGRA in little-endian memory
    return buf[..., [2, 1, 0, 3]].astype(np.float32) / 255.0


def pil_to_sprite(img) -> np.ndarray:
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def solid(color, alpha_mask: np.ndarray) -> np.ndarray:
    """Premultiplied sprite of one colour shaped by an alpha mask (h, w)."""
    r, g, b = hex_rgb(color) if isinstance(color, str) else color
    out = np.empty(alpha_mask.shape + (4,), np.float32)
    out[..., 0] = r * alpha_mask
    out[..., 1] = g * alpha_mask
    out[..., 2] = b * alpha_mask
    out[..., 3] = alpha_mask
    return out


# --------------------------------------------------------------------------- sprite ops
def blur(spr: np.ndarray, sigma: float) -> np.ndarray:
    pad = int(sigma * 3) + 2
    p = cv2.copyMakeBorder(spr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.GaussianBlur(p, (0, 0), sigma)


def tint(spr: np.ndarray, color, strength: float = 1.0) -> np.ndarray:
    r, g, b = hex_rgb(color) if isinstance(color, str) else color
    out = spr.copy()
    a = spr[..., 3:4]
    col = np.array([r, g, b], np.float32) * a
    out[..., :3] = spr[..., :3] * (1 - strength) + col * strength
    return out


def gold_map(spr: np.ndarray, dark="#5a3a0a", mid="#c9a24a", light="#fff1c1") -> np.ndarray:
    """Re-colour a sprite as polished gold using its luminance."""
    a = spr[..., 3:4]
    rgb = np.divide(spr[..., :3], a, out=np.zeros_like(spr[..., :3]), where=a > 1e-4)
    lum = (0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2])[..., None]
    lum = np.clip(lum * 1.15, 0, 1)
    d, m, l = (np.array(hex_rgb(c), np.float32) for c in (dark, mid, light))
    col = np.where(lum < 0.5, d + (m - d) * (lum / 0.5), m + (l - m) * ((lum - 0.5) / 0.5))
    out = np.empty_like(spr)
    out[..., :3] = col * a
    out[..., 3:] = a
    return out


def scale_sprite(spr: np.ndarray, s: float) -> np.ndarray:
    h, w = spr.shape[:2]
    return cv2.resize(spr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- placing
def place(frame: np.ndarray, spr: np.ndarray, x: float, y: float, scale: float = 1.0,
          rot: float = 0.0, alpha: float = 1.0, anchor=(0.5, 0.5), add: bool = False,
          sx: float | None = None, sy: float | None = None) -> None:
    """Composite `spr` onto `frame` in place. (x, y) is where the anchor lands.
    rot in degrees (counter-clockwise). sx/sy allow non-uniform scaling."""
    if alpha <= 0.002:
        return
    h, w = spr.shape[:2]
    ax, ay = anchor[0] * w, anchor[1] * h
    sx = scale if sx is None else sx * scale
    sy = scale if sy is None else sy * scale
    FH, FW = frame.shape[:2]
    if rot == 0 and abs(sx - 1) < 1e-4 and abs(sy - 1) < 1e-4:
        x0, y0 = int(round(x - ax)), int(round(y - ay))
        src = spr
    else:
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        A = np.array([[c * sx, s * sy], [-s * sx, c * sy]], np.float32)
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32) - [ax, ay]
        tc = corners @ A.T + [x, y]
        x0, y0 = int(math.floor(tc[:, 0].min())), int(math.floor(tc[:, 1].min()))
        x1, y1 = int(math.ceil(tc[:, 0].max())), int(math.ceil(tc[:, 1].max()))
        # clip early to the frame to avoid warping huge off-screen areas
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, FW), min(y1, FH)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        t = np.array([x, y], np.float32) - A @ np.array([ax, ay], np.float32) - [cx0, cy0]
        M = np.hstack([A, t[:, None]])
        src = cv2.warpAffine(spr, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        x0, y0 = cx0, cy0
    sh, sw = src.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + sw, FW), min(y0 + sh, FH)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = src[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    d = frame[fy0:fy1, fx0:fx1]
    if add:
        d += s[..., :3] * alpha
    else:
        d *= 1.0 - s[..., 3:4] * alpha
        d += s[..., :3] * alpha


def over(frame: np.ndarray, layer: np.ndarray, alpha: float = 1.0) -> None:
    """Full-frame premultiplied layer over frame."""
    if alpha <= 0.002:
        return
    frame *= 1.0 - layer[..., 3:4] * alpha
    frame += layer[..., :3] * alpha


def ken_burns(img: np.ndarray, zoom: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Zoom an (H, W, C) image about its centre, with a pixel pan offset."""
    h, w = img.shape[:2]
    M = np.array([[zoom, 0, (1 - zoom) * w / 2 + dx], [0, zoom, (1 - zoom) * h / 2 + dy]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def vertical_gradient(w, h, stops) -> np.ndarray:
    """stops: list of (pos 0..1, hex). Returns (h, w, 3) float32."""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    pos = [p for p, _ in stops]
    cols = np.array([hex_rgb(c) for _, c in stops], np.float32)
    col = np.stack([np.interp(ys, pos, cols[:, i]) for i in range(3)], -1)
    return np.repeat(col[:, None, :], w, axis=1)


def radial_mask(w, h, cx, cy, r0, r1) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r1 - d) / max(r1 - r0, 1e-3), 0, 1)


def vignette(frame: np.ndarray, strength: float = 0.45) -> None:
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = np.clip(d, 0, 1.4)[..., None] ** 2.2
    frame *= 1.0 - strength * np.clip(_VIG, 0, 1)


_VIG = None


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
