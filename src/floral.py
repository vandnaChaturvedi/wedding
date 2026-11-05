"""Original artwork for Video C – the pink floral style (white ground, pink peonies, hanging hearts,
dotted Ganesha medallion and a jharokha arch window). Drawn with cairo like illustrations.py;
nothing is traced from the reference template.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import cairocffi as cairo
import numpy as np

from compositor import H, W, cairo_canvas, hex_rgb
from illustrations import TAU, draw_leaf, ellipse, finish, lin, petal_path, rad, rgba, smooth_path

C = dict(
    magenta="#d81b60", hot="#e6007e", pink="#f06292", blush="#f8bbd0", petal="#f7c6d3",
    orange="#f26b1d", amber="#f7941d", yellow="#e8a317", teal="#4fa3a5", sage="#a9b8a0",
    dusty="#b59a8e", ink="#7a1040", white="#ffffff",
)
PEONY = [("#c2185b", "#f06292", "#ffd6e0"), ("#d81b60", "#f48fb1", "#ffe4ec"), ("#e5737f", "#f8a5b0", "#fff0f2")]


# ----------------------------------------------------------------------------- motifs
def heart_path(ctx, cx, cy, s):
    """Heart with its point at the bottom; (cx, cy) is the top cleft, s its width."""
    r = s / 4
    ctx.move_to(cx, cy + r * 0.6)
    ctx.curve_to(cx, cy - r * 0.4, cx - 2 * r, cy - r * 0.9, cx - 2 * r, cy + r * 0.5)
    ctx.curve_to(cx - 2 * r, cy + r * 1.6, cx - r * 0.6, cy + r * 2.4, cx, cy + s * 0.95)
    ctx.curve_to(cx + r * 0.6, cy + r * 2.4, cx + 2 * r, cy + r * 1.6, cx + 2 * r, cy + r * 0.5)
    ctx.curve_to(cx + 2 * r, cy - r * 0.9, cx, cy - r * 0.4, cx, cy + r * 0.6)
    ctx.close_path()


def blossom(ctx, cx, cy, r, color, centre="#c2185b", n=5, rot=0.0, a=1.0):
    """Small round five-petal blossom."""
    for i in range(n):
        ang = rot + TAU * i / n
        ctx.arc(cx + math.cos(ang) * r * 0.55, cy + math.sin(ang) * r * 0.55, r * 0.5, 0, TAU)
        rgba(ctx, color, a)
        ctx.fill()
    ctx.arc(cx, cy, r * 0.22, 0, TAU)
    rgba(ctx, centre, a)
    ctx.fill()


def draw_peony(ctx, cx, cy, r, pal=PEONY[0], rng=None):
    """Watercolour-like peony: ruffled cupped petals, dark base fading to pale edges."""
    rng = rng or random.Random(3)
    dark, mid, light = pal
    layers = [(1.0, 9, 0.0), (0.82, 8, -0.08), (0.64, 7, -0.16), (0.46, 6, -0.22), (0.3, 5, -0.26)]
    for li, (k, n, lift) in enumerate(layers):
        R = r * k
        ocy = cy + r * lift
        for j in range(n):
            a = TAU * j / n + li * 0.37 + rng.uniform(-0.12, 0.12)
            span = TAU / n * 0.85
            pts = [(cx + math.cos(a) * R * 0.15, ocy + math.sin(a) * R * 0.15 * 0.8)]
            m = 9
            for q in range(m + 1):
                u = q / m
                aa = a - span + 2 * span * u
                rr = R * (0.86 + 0.14 * math.sin(math.pi * u)) * (1 + 0.05 * math.sin(u * 17 + j))
                pts.append((cx + math.cos(aa) * rr, ocy + math.sin(aa) * rr * 0.8))
            smooth_path(ctx, pts, close=True)
            rad(ctx, cx, ocy, R * 0.1, R, [(0, dark), (0.45, mid), (1, light)])
            ctx.fill_preserve()
            rgba(ctx, dark, 0.18)
            ctx.set_line_width(max(1, r * 0.012))
            ctx.stroke()
    # centre: tight ruffles and a few stamens
    for j in range(7):
        a = TAU * j / 7 + 0.4
        petal_path(ctx, cx + math.cos(a) * r * 0.05, cy - r * 0.24 + math.sin(a) * r * 0.04, r * 0.16,
                   r * 0.1, a)
        rgba(ctx, mid if j % 2 else dark, 0.8)
        ctx.fill()
    for j in range(9):
        a = rng.uniform(0, TAU)
        ctx.arc(cx + math.cos(a) * r * 0.08, cy - r * 0.26 + math.sin(a) * r * 0.06, r * 0.022, 0, TAU)
        rgba(ctx, "#f9c74f")
        ctx.fill()
    # pale watercolour bloom over the upper half
    rad(ctx, cx - r * 0.25, cy - r * 0.35, 0, r * 0.8, [(0, "#ffffff", 0.35), (1, "#ffffff", 0.0)])
    ctx.arc(cx, cy, r, 0, TAU)
    ctx.fill()


def draw_bud(ctx, x, y, r, ang, color="#e5737f"):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ellipse(ctx, 0, 0, r * 0.7, r)
    rad(ctx, 0, r * 0.3, 0, r, [(0, "#c2185b"), (1, color)])
    ctx.fill()
    for s in (-1, 1):
        petal_path(ctx, 0, r * 0.9, r * 0.9, r * 0.35, math.pi - s * 0.5)
        rgba(ctx, "#7f9a78")
        ctx.fill()
    ctx.restore()


def sprig(ctx, x, y, length, ang, rng, color=None, flowers=True):
    """Thin stem with dusty leaves and tiny blossoms (the airy filler in the template's bouquets)."""
    color = color or rng.choice([C["sage"], C["dusty"], "#8fa58a"])
    pts = []
    for q in range(6):
        u = q / 5
        bend = math.sin(u * 2.2) * length * 0.08
        pts.append((x + math.cos(ang) * length * u - math.sin(ang) * bend,
                    y + math.sin(ang) * length * u + math.cos(ang) * bend))
    smooth_path(ctx, pts)
    rgba(ctx, "#8d7b72", 0.8)
    ctx.set_line_width(max(1.2, length * 0.012))
    ctx.stroke()
    for q in range(1, 6):
        px, py = pts[q]
        side = 1 if q % 2 else -1
        draw_leaf(ctx, px, py, length * 0.16, length * 0.06, ang + math.pi / 2 + side * 0.9 - math.pi / 2,
                  color=color, vein=False)
    if flowers:
        ex, ey = pts[-1]
        for k in range(rng.randint(2, 4)):
            blossom(ctx, ex + rng.uniform(-1, 1) * length * 0.1, ey + rng.uniform(-1, 1) * length * 0.1,
                    length * rng.uniform(0.05, 0.08), rng.choice(["#f48fb1", "#f8bbd0", "#e91e63", "#ffffff"]),
                    rot=rng.uniform(0, 1))


# ----------------------------------------------------------------------------- background
@lru_cache(None)
def floral_pattern(seed=2):
    """White ground with a soft scatter of pink blossoms and sage sprigs (full frame RGB)."""
    surf, ctx = cairo_canvas(W, H)
    rgba(ctx, "#fffafb")
    ctx.paint()
    rng = random.Random(seed)
    step = 120
    for gy in range(-1, H // step + 2):
        for gx in range(-1, W // step + 2):
            x = gx * step + (step / 2 if gy % 2 else 0) + rng.uniform(-22, 22)
            y = gy * step + rng.uniform(-22, 22)
            sprig(ctx, x + 18, y + 10, rng.uniform(40, 60), rng.uniform(0, TAU), rng, color="#d7ddd2",
                  flowers=False)
            blossom(ctx, x, y, rng.uniform(17, 24), "#f9d3dd", centre="#f3a9bd", rot=rng.uniform(0, 1))
            if rng.random() < 0.5:
                blossom(ctx, x + rng.uniform(30, 50), y + rng.uniform(-50, -30), rng.uniform(9, 12), "#fbe0e7",
                        centre="#f3a9bd")
    return finish(surf)[..., :3].copy()


# ----------------------------------------------------------------------------- heart toran
@lru_cache(None)
def heart_toran(width=1080, drop=360, seed=7):
    """Floral swag along the top with beaded strands of hearts ending in tassels."""
    surf, ctx = cairo_canvas(width, drop)
    rng = random.Random(seed)
    n = 21
    for s in range(n):
        x = (s + 0.5) * width / n + rng.uniform(-6, 6)
        L = rng.uniform(0.45, 1.0) * (drop - 110) + 40
        ctx.move_to(x, 30)
        ctx.line_to(x, L)
        rgba(ctx, "#c98fa3", 0.8)
        ctx.set_line_width(1.6)
        ctx.stroke()
        y = 55
        while y < L - 30:
            kind = rng.random()
            if kind < 0.42:
                sz = rng.uniform(18, 28)
                heart_path(ctx, x, y, sz)
                lin(ctx, x, y, x, y + sz, [(0, rng.choice(["#e91e63", "#f06292", "#c2185b", "#f48fb1"])),
                                            (1, "#ad1457")])
                ctx.fill()
                y += sz + 12
            else:
                ctx.arc(x, y, rng.uniform(3, 5), 0, TAU)
                rgba(ctx, rng.choice([C["teal"], "#f48fb1", "#c2185b", "#f9c74f"]))
                ctx.fill()
                y += 14
        # tassel
        col = rng.choice([C["teal"], "#f06292", "#e05780"])
        ctx.arc(x, L, 6, 0, TAU)
        rgba(ctx, col)
        ctx.fill()
        for k in range(9):
            dx = (k - 4) * 1.6
            ctx.move_to(x + dx * 0.4, L + 4)
            ctx.line_to(x + dx * 1.6, L + 42)
            rgba(ctx, col, 0.9)
            ctx.set_line_width(1.6)
            ctx.stroke()
    # swag of flowers along the top edge
    for i in range(int(width / 46) + 2):
        x = i * 46 + rng.uniform(-8, 8)
        draw_leaf(ctx, x, 18, 44, 14, rng.uniform(0.2, 0.9) + (0 if i % 2 else math.pi * 0.6), color="#8fa58a",
                  vein=False)
    for i in range(int(width / 70) + 2):
        x = i * 70 + rng.uniform(-10, 10)
        if i % 2:
            draw_peony(ctx, x, 18 + rng.uniform(-4, 6), rng.uniform(20, 26), rng.choice(PEONY), rng)
        else:
            blossom(ctx, x, 20, rng.uniform(13, 17), rng.choice(["#f48fb1", "#f8bbd0"]), rot=rng.uniform(0, 1))
    return finish(surf)


# ----------------------------------------------------------------------------- bouquets
@lru_cache(None)
def peony_border(width=1080, height=560, seed=4):
    """Bottom band: two big peonies, sprays and buds rising from the lower edge."""
    surf, ctx = cairo_canvas(width, height)
    rng = random.Random(seed)
    for _ in range(26):
        x = rng.uniform(0, width)
        sprig(ctx, x, height + 10, rng.uniform(150, 300), -math.pi / 2 + rng.uniform(-0.7, 0.7), rng)
    for _ in range(18):
        x = rng.uniform(0, width)
        draw_leaf(ctx, x, height - rng.uniform(40, 160), rng.uniform(70, 120), rng.uniform(26, 38),
                  rng.uniform(-2.6, -0.5), color=rng.choice(["#7f9a78", "#9aae8e", "#a89a8c"]), vein=True)
    for x, y, r, k in ((0.07, 0.86, 110, 2), (0.93, 0.86, 110, 2), (0.3, 0.78, 150, 0), (0.72, 0.8, 140, 1),
                       (0.5, 0.95, 90, 2), (0.15, 0.55, 60, 1), (0.86, 0.52, 58, 0)):
        draw_peony(ctx, x * width, y * height, r, PEONY[k], rng)
    for _ in range(9):
        draw_bud(ctx, rng.uniform(0.05, 0.95) * width, rng.uniform(0.3, 0.6) * height, rng.uniform(12, 18),
                 rng.uniform(-0.5, 0.5))
    for _ in range(16):
        x, y = rng.uniform(0, width), rng.uniform(0.45, 0.95) * height
        blossom(ctx, x, y, rng.uniform(9, 15), rng.choice(["#e6007e", "#d81b60", "#f48fb1"]), centre="#880e4f",
                rot=rng.uniform(0, 1))
    return finish(surf)


@lru_cache(None)
def peony_corner(size=560, seed=9, flip_x=False, flip_y=False):
    """Corner bouquet (drawn for the top-left corner; flips give the other three)."""
    surf, ctx = cairo_canvas(size, size)
    rng = random.Random(seed)
    for _ in range(14):
        sprig(ctx, size * 0.1, size * 0.1, rng.uniform(0.35, 0.7) * size, rng.uniform(0.0, math.pi / 2), rng)
    for _ in range(10):
        a = rng.uniform(0, math.pi / 2)
        draw_leaf(ctx, size * 0.18 + math.cos(a) * size * 0.25, size * 0.18 + math.sin(a) * size * 0.25,
                  rng.uniform(60, 100) * size / 560, rng.uniform(24, 34) * size / 560, a,
                  color=rng.choice(["#7f9a78", "#9aae8e", "#a89a8c"]))
    draw_peony(ctx, size * 0.2, size * 0.16, size * 0.24, PEONY[0], rng)
    draw_peony(ctx, size * 0.46, size * 0.1, size * 0.12, PEONY[1], rng)
    draw_peony(ctx, size * 0.1, size * 0.44, size * 0.11, PEONY[2], rng)
    for _ in range(6):
        draw_bud(ctx, rng.uniform(0.3, 0.6) * size, rng.uniform(0.25, 0.5) * size, size * 0.025,
                 rng.uniform(-1, 1))
    for _ in range(10):
        blossom(ctx, rng.uniform(0.05, 0.55) * size, rng.uniform(0.05, 0.55) * size, size * rng.uniform(0.016, 0.026),
                rng.choice(["#e6007e", "#d81b60", "#f48fb1"]), centre="#880e4f", rot=rng.uniform(0, 1))
    d = finish(surf)
    if flip_x:
        d = d[:, ::-1]
    if flip_y:
        d = d[::-1]
    return np.ascontiguousarray(d)


# ----------------------------------------------------------------------------- medallion
@lru_cache(None)
def dot_medallion(size=640):
    """Ganesha medallion in three rings (so they can counter-rotate): outer yellow ring with magenta
    dots, orange ring with white dots, and the white centre with pink and gold rings."""
    out = []
    c = size / 2
    s1, ctx = cairo_canvas(size, size)
    n = 26
    for i in range(n):
        a = TAU * i / n
        ctx.arc(c + math.cos(a) * size * 0.465, c + math.sin(a) * size * 0.465, size * 0.032, 0, TAU)
        rgba(ctx, C["magenta"])
        ctx.fill()
    ctx.arc(c, c, size * 0.44, 0, TAU)
    rgba(ctx, C["yellow"])
    ctx.fill()
    out.append(finish(s1))
    s2, ctx = cairo_canvas(size, size)
    ctx.arc(c, c, size * 0.405, 0, TAU)
    rgba(ctx, C["orange"])
    ctx.fill()
    n = 24
    for i in range(n):
        a = TAU * i / n
        ctx.arc(c + math.cos(a) * size * 0.375, c + math.sin(a) * size * 0.375, size * 0.021, 0, TAU)
        rgba(ctx, "#ffffff")
        ctx.fill()
    out.append(finish(s2))
    s3, ctx = cairo_canvas(size, size)
    ctx.arc(c, c, size * 0.325, 0, TAU)
    rgba(ctx, C["hot"])
    ctx.fill()
    ctx.arc(c, c, size * 0.305, 0, TAU)
    rgba(ctx, C["orange"])
    ctx.fill()
    ctx.arc(c, c, size * 0.29, 0, TAU)
    rgba(ctx, C["yellow"])
    ctx.fill()
    ctx.arc(c, c, size * 0.275, 0, TAU)
    rgba(ctx, "#ffffff")
    ctx.fill()
    out.append(finish(s3))
    return out


# ----------------------------------------------------------------------------- jharokha window
def _arch(ctx, x0, x1, y_bot, y_spring):
    """Round-topped arch path between x0..x1, sides from y_bot up to y_spring."""
    cx, r = (x0 + x1) / 2, (x1 - x0) / 2
    ctx.move_to(x0, y_bot)
    ctx.line_to(x0, y_spring)
    ctx.arc(cx, y_spring, r, math.pi, TAU)
    ctx.line_to(x1, y_bot)
    ctx.close_path()


@lru_cache(None)
def jharokha(w=600, h=780):
    """Arch window frame (orange band, magenta outline, scroll crest, ledge). Returns (frame sprite,
    opening mask (h, w) float32, (ox0, ox1, oy_bot, oy_spring))."""
    surf, ctx = cairo_canvas(w, h)
    ox0, ox1 = w * 0.17, w * 0.83
    oy_bot, oy_spring = h * 0.86, h * 0.47
    bx0, bx1 = w * 0.06, w * 0.94
    by_spring = oy_spring + 10
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    _arch(ctx, bx0, bx1, oy_bot + 10, by_spring)
    _arch(ctx, ox0, ox1, oy_bot, oy_spring)
    lin(ctx, 0, 0, 0, h, [(0, C["amber"]), (1, "#f58220")])
    ctx.fill()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    for (a0, a1, sp, lw, col) in ((bx0, bx1, by_spring, 7, C["hot"]), (ox0, ox1, oy_spring, 7, C["hot"])):
        _arch(ctx, a0, a1, oy_bot + 10, sp)
        rgba(ctx, col)
        ctx.set_line_width(lw)
        ctx.stroke()
    # dotted inner line
    mx0, mx1 = ox0 - 20, ox1 + 20
    cx, r = w / 2, (mx1 - mx0) / 2
    pts = [(mx0, oy_bot - 10 - k * 18) for k in range(int((oy_bot - 10 - oy_spring) / 18))]
    pts += [(cx + math.cos(math.pi + math.pi * u) * r, oy_spring + math.sin(math.pi + math.pi * u) * r)
            for u in np.linspace(0, 1, 26)]
    pts += [(mx1, oy_spring + k * 18) for k in range(int((oy_bot - 10 - oy_spring) / 18))]
    for x, y in pts:
        ctx.arc(x, y, 3.4, 0, TAU)
        rgba(ctx, C["hot"])
        ctx.fill()
    # scroll crest around the top of the band
    R = (bx1 - bx0) / 2 - 22
    for k in range(13):
        u = (k + 0.5) / 13
        a = math.pi + math.pi * u
        x, y = cx + math.cos(a) * R, by_spring + math.sin(a) * R
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(a + math.pi / 2)
        for q in range(18):
            ang = q * 0.5
            rr = 2 + q * 0.75
            px, py = math.cos(ang) * rr, math.sin(ang) * rr * 0.8
            (ctx.move_to if q == 0 else ctx.line_to)(px, py)
        rgba(ctx, C["hot"])
        ctx.set_line_width(3.2)
        ctx.stroke()
        ctx.restore()
    # crest jewel
    heart_path(ctx, cx, by_spring - (bx1 - bx0) / 2 - 2, 26)
    rgba(ctx, C["hot"])
    ctx.fill()
    # ledge
    ly = oy_bot + 6
    ctx.move_to(w * 0.0, ly)
    ctx.line_to(w * 1.0, ly)
    ctx.line_to(w * 0.94, h * 0.98)
    ctx.line_to(w * 0.06, h * 0.98)
    ctx.close_path()
    lin(ctx, 0, ly, 0, h, [(0, "#ec407a"), (1, C["magenta"])])
    ctx.fill()
    ctx.rectangle(w * 0.03, ly + 10, w * 0.94, 8)
    rgba(ctx, C["amber"])
    ctx.fill()
    for s in (-1, 1):
        ctx.move_to(cx + s * 22, ly + 34)
        ctx.line_to(cx + s * 90, ly + 34)
        rgba(ctx, C["amber"])
        ctx.set_line_width(3)
        ctx.stroke()
    heart_path(ctx, cx, ly + 26, 18)
    rgba(ctx, C["amber"])
    ctx.fill()
    frame = finish(surf)
    ms, mctx = cairo_canvas(w, h)
    _arch(mctx, ox0, ox1, oy_bot + 2, oy_spring)
    mctx.set_source_rgba(1, 1, 1, 1)
    mctx.fill()
    mask = finish(ms)[..., 3].copy()
    return frame, mask, (ox0, ox1, oy_bot, oy_spring)


@lru_cache(None)
def window_scene(w=600, h=780):
    """Soft garden-at-dusk backdrop seen through the window (RGB, full sprite size)."""
    surf, ctx = cairo_canvas(w, h)
    lin(ctx, 0, 0, 0, h, [(0, "#fde2e4"), (0.55, "#fbc4cf"), (1, "#f6a7b8")])
    ctx.paint()
    rng = random.Random(5)
    for _ in range(14):
        x, y, r = rng.uniform(0, w), rng.uniform(h * 0.45, h * 0.8), rng.uniform(40, 90)
        rad(ctx, x, y, 0, r, [(0, "#c5d6b8", 0.7), (1, "#c5d6b8", 0.0)])
        ctx.arc(x, y, r, 0, TAU)
        ctx.fill()
    for k in range(3):  # strings of fairy lights
        y0 = h * (0.24 + 0.08 * k)
        for i in range(16):
            u = i / 15
            x = w * u
            y = y0 + 40 * 4 * u * (1 - u)
            rad(ctx, x, y, 0, 10, [(0, "#fffbe6", 0.95), (1, "#fff3b0", 0.0)])
            ctx.arc(x, y, 10, 0, TAU)
            ctx.fill()
    return finish(surf)[..., :3].copy()
