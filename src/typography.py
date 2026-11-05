"""Text rendering with correct Devanagari shaping (Pillow + libraqm) and gold-foil effects."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from compositor import hex_rgb

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# key -> (file, variable-font weight or None)
FONTS = {
    "deva": ("NotoSerifDevanagari-Medium.ttf", None),
    "deva_semi": ("NotoSerifDevanagari-SemiBold.ttf", None),
    "deva_bold": ("NotoSerifDevanagari-Bold.ttf", None),
    "deva_sans": ("NotoSansDevanagari-Medium.ttf", None),
    "tiro": ("TiroDevanagariHindi-Regular.ttf", None),
    "tiro_sa": ("TiroDevanagariSanskrit-Regular.ttf", None),
    "cinzel": ("Cinzel.ttf", 500),
    "cinzel_bold": ("Cinzel.ttf", 700),
    "cinzel_deco": ("CinzelDecorative-Bold.ttf", None),
    "cormorant": ("CormorantGaramond.ttf", 500),
    "cormorant_semi": ("CormorantGaramond.ttf", 600),
    "cormorant_bold": ("CormorantGaramond.ttf", 700),
    "cormorant_it": ("CormorantGaramond-Italic.ttf", 500),
    "playfair": ("PlayfairDisplay.ttf", 500),
    "playfair_bold": ("PlayfairDisplay.ttf", 700),
    "playfair_it": ("PlayfairDisplay-Italic.ttf", 500),
    "mont": ("Montserrat.ttf", 500),
    "mont_semi": ("Montserrat.ttf", 600),
    "mont_it": ("Montserrat-Italic.ttf", 500),
    "script": ("GreatVibes-Regular.ttf", None),
    "pinyon": ("PinyonScript-Regular.ttf", None),
}


def check_fonts() -> list[str]:
    """Return a list of problems (empty if all fonts + complex shaping are available)."""
    problems = []
    if not features.check("raqm"):
        problems.append("Pillow was built without libraqm: Devanagari conjuncts/matras would break.")
    for key, (fn, _) in FONTS.items():
        if not (FONT_DIR / fn).exists():
            problems.append(f"missing font {fn} ({key})")
    return problems


@lru_cache(None)
def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    fn, wght = FONTS[key]
    f = ImageFont.truetype(str(FONT_DIR / fn), size, layout_engine=ImageFont.Layout.RAQM)
    if wght is not None:
        try:
            axes = f.get_variation_axes()
            f.set_variation_by_axes([wght if a.get("name", b"") in (b"Weight", "Weight") or i == 0 else a["default"]
                                     for i, a in enumerate(axes)])
        except OSError:
            pass
    return f


def _is_deva(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def text_mask(text: str, key: str, size: int, tracking: float = 0.0) -> np.ndarray:
    """Antialiased coverage mask (float32 0..1), tightly cropped with a small margin."""
    f = font(key, size)
    lang = "hi" if _is_deva(text) else None
    pad = int(size * 0.6)
    if tracking and not _is_deva(text):
        # per-glyph tracking only for Latin (Devanagari clusters must never be split)
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + tracking * size * (len(text) - 1)
        img = Image.new("L", (int(total + 2 * pad), int(size * 2.2 + 2 * pad)), 0)
        d = ImageDraw.Draw(img)
        x = pad
        for ch, w in zip(text, widths):
            d.text((x, pad + size * 0.2), ch, font=f, fill=255)
            x += w + tracking * size
    else:
        l, t, r, b = f.getbbox(text, language=lang)
        img = Image.new("L", (int(r - l + 2 * pad), int(b - t + 2 * pad)), 0)
        ImageDraw.Draw(img).text((pad - l, pad - t), text, font=f, fill=255, language=lang)
    a = np.asarray(img, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.003)
    if len(ys) == 0:
        return np.zeros((2, 2), np.float32)
    m = 6
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, a.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, a.shape[1])
    return a[y0:y1, x0:x1]


GOLD_FOIL = [(0.0, "#fff4c9"), (0.35, "#e9c46a"), (0.55, "#b8860b"), (0.75, "#f6d77d"), (1.0, "#9c6b12")]
CHAMPAGNE = [(0.0, "#fffaf0"), (0.4, "#f1dcb0"), (0.6, "#c9a46a"), (0.8, "#f5e3bd"), (1.0, "#a67c3d")]
ROSE_GOLD = [(0.0, "#fff0ea"), (0.4, "#f0c3b0"), (0.65, "#b76e59"), (1.0, "#f3cdbd")]


def _gradient(h, w, stops):
    ys = np.linspace(0, 1, h, dtype=np.float32)
    cols = np.array([hex_rgb(c) for _, c in stops], np.float32)
    pos = [p for p, _ in stops]
    g = np.stack([np.interp(ys, pos, cols[:, i]) for i in range(3)], -1)
    return np.repeat(g[:, None, :], w, 1)


def text_sprite(text: str, key: str, size: int, color="#6b0f1a", gradient=None, shadow=None,
                stroke=None, glow=None, tracking=0.0, max_width=None) -> np.ndarray:
    """Premultiplied RGBA sprite for one line of text.

    shadow: (hex, alpha, blur, dy)   stroke: (hex, px)   glow: (hex, alpha, blur)
    max_width: shrink the font until the line fits (guards against clipped text).
    """
    while True:
        m = text_mask(text, key, size, tracking)
        if max_width is None or m.shape[1] <= max_width or size <= 12:
            break
        size = int(size * 0.95)
    pad = 0
    effects = [e for e in (shadow, glow) if e]
    if effects:
        pad = int(max(e[2] for e in effects) * 3 + (shadow[3] if shadow else 0)) + 4
    if stroke:
        pad = max(pad, stroke[1] + 2)
    h, w = m.shape[0] + 2 * pad, m.shape[1] + 2 * pad
    mask = np.zeros((h, w), np.float32)
    mask[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m
    out = np.zeros((h, w, 4), np.float32)

    def layer(rgb, a):
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + rgb * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a

    if glow:
        g = cv2.GaussianBlur(mask, (0, 0), glow[2]) * glow[1]
        layer(np.array(hex_rgb(glow[0]), np.float32), np.clip(g * 1.6, 0, 1))
    if shadow:
        s = cv2.GaussianBlur(np.roll(mask, int(shadow[3]), axis=0), (0, 0), max(shadow[2], 0.1)) * shadow[1]
        layer(np.array(hex_rgb(shadow[0]), np.float32), s)
    if stroke:
        k = 2 * stroke[1] + 1
        dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        layer(np.array(hex_rgb(stroke[0]), np.float32), dil)
    if gradient:
        y0 = pad
        g = _gradient(m.shape[0], w, gradient)
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[y0:y0 + m.shape[0]] = g
        layer(rgb, mask)
    else:
        layer(np.array(hex_rgb(color), np.float32), mask)
    return out  # already premultiplied (built with 'over' on straight colours into premult space)


def stack(lines, gap=18, max_width=None):
    """Lay out a list of line dicts vertically. Each dict: text, font, size, + text_sprite kwargs,
    optional 'gap' (space after). Returns list of (sprite, y_offset) and total height."""
    out, y = [], 0
    for ln in lines:
        if "sprite" in ln:
            spr = ln["sprite"]
        else:
            kw = {k: v for k, v in ln.items() if k not in ("text", "font", "size", "gap", "sprite")}
            kw.setdefault("max_width", max_width)
            spr = text_sprite(ln["text"], ln["font"], ln["size"], **kw)
        out.append((spr, y))
        y += spr.shape[0] + ln.get("gap", gap)
    return out, y - (lines[-1].get("gap", gap) if lines else 0)
