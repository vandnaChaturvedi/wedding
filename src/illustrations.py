"""Original procedural vector illustrations (drawn with cairo).

Every function returns a premultiplied RGBA float32 sprite (see compositor.py).
Nothing here is traced from or copied from third-party artwork.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import cairocffi as cairo
import numpy as np

from compositor import cairo_canvas, hex_rgb, surface_to_sprite

TAU = math.tau

# ----------------------------------------------------------------------------- palette
P = dict(
    maroon="#6b0f1a", crimson="#a4161a", red="#c1121f", deep="#3d0610",
    gold="#c9a227", gold_d="#8a6a12", gold_l="#f3d98b", champagne="#e9d3a3",
    ivory="#fff8e7", cream="#f6ead0", marigold="#f28c28", yellow="#ffc72c",
    saffron="#ff9933", leaf="#3a7d44", leaf_d="#24502b", leaf_l="#6aa84f",
    pink="#f4a6b8", rose="#d6336c", lotus="#f7b2c4", lotus_d="#e0607e",
    skin="#e2a77f", skin_d="#c98a62", hair="#1f1410", white="#ffffff",
    ganesh="#f4a77a", ganesh_d="#e07f55", ganesh_l="#ffd2b0", turmeric="#f4c430",
    burgundy="#5c0a1f", rosegold="#e7b7a3",
)


def rgba(ctx, color, a=1.0):
    r, g, b = hex_rgb(P.get(color, color))
    ctx.set_source_rgba(r, g, b, a)


def lin(ctx, x0, y0, x1, y1, stops):
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for p, c, *a in stops:
        r, gg, b = hex_rgb(P.get(c, c))
        g.add_color_stop_rgba(p, r, gg, b, a[0] if a else 1.0)
    ctx.set_source(g)


def rad(ctx, cx, cy, r0, r1, stops, fx=None, fy=None):
    g = cairo.RadialGradient(cx if fx is None else fx, cy if fy is None else fy, r0, cx, cy, r1)
    for p, c, *a in stops:
        r, gg, b = hex_rgb(P.get(c, c))
        g.add_color_stop_rgba(p, r, gg, b, a[0] if a else 1.0)
    ctx.set_source(g)


def fill_stroke(ctx, fill, stroke=None, lw=2.0, fa=1.0, sa=1.0):
    if fill is not None:
        if isinstance(fill, str):
            rgba(ctx, fill, fa)
        else:
            fill(ctx)
        if stroke is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if stroke is not None:
        rgba(ctx, stroke, sa)
        ctx.set_line_width(lw)
        ctx.stroke()


def ellipse(ctx, cx, cy, rx, ry):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()


def petal_path(ctx, cx, cy, length, width, angle, tip=0.0):
    """Teardrop petal from (cx, cy) outward along angle."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.move_to(0, 0)
    ctx.curve_to(width * 0.9, -length * 0.15, width * 0.8, -length * 0.8, tip, -length)
    ctx.curve_to(-width * 0.8, -length * 0.8, -width * 0.9, -length * 0.15, 0, 0)
    ctx.close_path()
    ctx.restore()


def poly_path(ctx, pts, close=True):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def smooth_path(ctx, pts, close=False, k=0.5):
    """Catmull-Rom through pts as cubic beziers."""
    n = len(pts)
    if close:
        pts = [pts[-1]] + list(pts) + [pts[0], pts[1]]
    else:
        pts = [pts[0]] + list(pts) + [pts[-1]]
    ctx.move_to(*pts[1])
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 3, p1[1] + (p2[1] - p0[1]) * k / 3)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 3, p2[1] - (p3[1] - p1[1]) * k / 3)
        ctx.curve_to(*c1, *c2, *p2)
    if close:
        ctx.close_path()


def tapered_stroke(ctx, pts, w0, w1):
    """Filled polygon along a centre-line polyline with tapering width."""
    pts = np.asarray(pts, float)
    n = len(pts)
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    ws = np.linspace(w0, w1, n)[:, None] / 2
    left, right = pts + nrm * ws, pts - nrm * ws
    poly = list(map(tuple, left)) + list(map(tuple, right[::-1]))
    smooth_path(ctx, poly, close=True, k=0.3)


def tapered_edges(pts, w0, w1):
    pts = np.asarray(pts, float)
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    ws = np.linspace(w0, w1, len(pts))[:, None] / 2
    return pts + nrm * ws, pts - nrm * ws


def bezier_pts(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def finish(surf):
    return surface_to_sprite(surf)


# ============================================================================ FLORA
def draw_marigold(ctx, cx, cy, r, base="marigold", rng=None):
    rng = rng or random.Random(1)
    rings = 5
    col = {"marigold": ("#d9620f", "#f28c28", "#ffb347"), "yellow": ("#e09b00", "#ffc72c", "#ffe27a"),
           "red": ("#8e0e1c", "#c1121f", "#ef4f5c")}[base]
    for i in range(rings, 0, -1):
        rr = r * i / rings
        n = 10 + i * 4
        for j in range(n):
            a = TAU * j / n + rng.random() * 0.3 + i
            ctx.save()
            ctx.translate(cx + math.cos(a) * rr * 0.55, cy + math.sin(a) * rr * 0.55)
            ctx.rotate(a)
            ellipse(ctx, 0, 0, rr * 0.5, rr * 0.28)
            rgba(ctx, col[0] if j % 2 else col[1])
            ctx.fill()
            ctx.restore()
    rad(ctx, cx, cy, 0, r * 0.45, [(0, col[2]), (1, col[1], 0)])
    ctx.arc(cx, cy, r * 0.45, 0, TAU)
    ctx.fill()


def draw_leaf(ctx, x, y, length, width, angle, color="leaf", vein=True):
    petal_path(ctx, x, y, length, width, angle)
    rgba(ctx, color)
    ctx.fill()
    if vein:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(angle)
        ctx.move_to(0, 0)
        ctx.line_to(0, -length * 0.9)
        rgba(ctx, "#ffffff", 0.25)
        ctx.set_line_width(max(1, width * 0.08))
        ctx.stroke()
        ctx.restore()


def draw_lotus(ctx, cx, cy, s, open_=1.0):
    """Side-view lotus with layered petals. Base at (cx, cy)."""
    # back petals
    for layer, (n, ln, wd, spread, c1, c2) in enumerate([
        (5, 1.0, 0.36, 1.25, "lotus_d", "lotus"),
        (4, 0.92, 0.34, 0.85, "lotus", "#ffd9e2"),
        (3, 0.8, 0.3, 0.42, "#ffd9e2", "#fff2f5"),
    ]):
        for i in range(n):
            a = (i / (n - 1) - 0.5) * spread * 2 * open_ if n > 1 else 0
            petal_path(ctx, cx, cy, s * ln, s * wd, a)
            lin(ctx, cx, cy, cx + math.sin(a) * s, cy - math.cos(a) * s, [(0, c1), (1, c2)])
            ctx.fill_preserve()
            rgba(ctx, "lotus_d", 0.55)
            ctx.set_line_width(max(1, s * 0.012))
            ctx.stroke()
    # calyx
    ellipse(ctx, cx, cy + s * 0.02, s * 0.32, s * 0.08)
    rgba(ctx, "leaf_d")
    ctx.fill()


def draw_lotus_pad(ctx, cx, cy, rx, ry, color="leaf"):
    ctx.move_to(cx, cy)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, -math.pi / 2 + 0.25, -math.pi / 2 - 0.25 + TAU)
    ctx.restore()
    ctx.close_path()
    lin(ctx, cx, cy - ry, cx, cy + ry, [(0, P["leaf_l"]), (1, P[color] if color in P else color)])
    ctx.fill()


def draw_rose(ctx, cx, cy, r, c1="#7a0c24", c2="#c2185b", c3="#f06292"):
    rad(ctx, cx, cy, 0, r, [(0, c3), (0.55, c2), (1, c1)])
    ctx.arc(cx, cy, r, 0, TAU)
    ctx.fill()
    for i in range(14):
        k = 1 - i / 14
        a0 = i * 2.3
        ctx.arc(cx + math.cos(a0) * r * 0.05, cy + math.sin(a0) * r * 0.05, r * k * 0.95, a0, a0 + 2.6)
        rgba(ctx, c1, 0.55)
        ctx.set_line_width(max(1, r * 0.06))
        ctx.stroke()
        ctx.arc(cx, cy, r * k * 0.9, a0 + 0.2, a0 + 1.9)
        rgba(ctx, c3, 0.5)
        ctx.set_line_width(max(1, r * 0.035))
        ctx.stroke()


def draw_jasmine(ctx, cx, cy, r):
    for i in range(5):
        petal_path(ctx, cx, cy, r, r * 0.55, TAU * i / 5)
        rgba(ctx, "#fffdf5")
        ctx.fill()
    ctx.arc(cx, cy, r * 0.18, 0, TAU)
    rgba(ctx, "yellow")
    ctx.fill()


@lru_cache(None)
def petal_sprites(kind="mixed"):
    """Small falling-petal sprites."""
    out = []
    specs = {
        "marigold": ["#f28c28", "#ff9f1c", "#ffb703", "#e85d04"],
        "rose": ["#c1121f", "#9d0208", "#d6336c", "#e5383b"],
        "pink": ["#ffb3c6", "#ff8fab", "#fb6f92", "#ffc2d1"],
        "mixed": ["#f28c28", "#ffb703", "#c1121f", "#d6336c", "#ff8fab"],
    }[kind]
    for i, c in enumerate(specs * 2):
        s, ctx = cairo_canvas(60, 60)
        petal_path(ctx, 30, 50, 40 + (i % 3) * 4, 22 + (i % 2) * 5, 0, tip=(i % 3 - 1) * 4)
        lin(ctx, 30, 50, 30, 10, [(0, c), (1, "#ffffff", 0.0)])
        rgba(ctx, c)
        ctx.fill_preserve()
        lin(ctx, 20, 50, 40, 10, [(0, "#ffffff", 0.0), (0.6, "#ffffff", 0.35), (1, "#ffffff", 0.0)])
        ctx.fill()
        out.append(finish(s))
    return out


@lru_cache(None)
def spark_sprite(size=48, color="#fff1c1"):
    s, ctx = cairo_canvas(size, size)
    c = size / 2
    rad(ctx, c, c, 0, c, [(0, "#ffffff", 1), (0.15, color, 0.9), (0.5, color, 0.25), (1, color, 0)])
    ctx.arc(c, c, c, 0, TAU)
    ctx.fill()
    return finish(s)


@lru_cache(None)
def star_sprite(size=64, color="#fff1c1"):
    s, ctx = cairo_canvas(size, size)
    c = size / 2
    rad(ctx, c, c, 0, c * 0.5, [(0, "#ffffff", 1), (1, color, 0)])
    ctx.arc(c, c, c * 0.5, 0, TAU)
    ctx.fill()
    for a in (0, math.pi / 2):
        ctx.save()
        ctx.translate(c, c)
        ctx.rotate(a)
        lin(ctx, -c, 0, c, 0, [(0, color, 0), (0.5, "#ffffff", 0.95), (1, color, 0)])
        ellipse(ctx, 0, 0, c, c * 0.06)
        ctx.fill()
        ctx.restore()
    return finish(s)


@lru_cache(None)
def bokeh_sprite(size=160, color="#f3d98b"):
    s, ctx = cairo_canvas(size, size)
    c = size / 2
    rad(ctx, c, c, 0, c * 0.95, [(0, color, 0.22), (0.8, color, 0.3), (0.92, color, 0.5), (1, color, 0)])
    ctx.arc(c, c, c * 0.95, 0, TAU)
    ctx.fill()
    return finish(s)


# ============================================================================ GANESHA
@lru_cache(None)
def ganesha(size=900):
    """Seated Lord Ganesha – original flat devotional illustration (900 x 1000 design)."""
    Wd, Hd = 900, 1000
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    cx = 450
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    OUT = "#7a3b12"

    # --- aura
    rad(ctx, cx, 400, 0, 440, [(0, "#fff4c2", 0.95), (0.55, "#ffd36b", 0.55), (1, "#ffb300", 0.0)])
    ctx.arc(cx, 400, 440, 0, TAU)
    ctx.fill()
    for i in range(36):
        a = TAU * i / 36
        petal_path(ctx, cx + math.cos(a) * 300, 400 + math.sin(a) * 300, 60, 22, a + math.pi / 2)
        rgba(ctx, "gold", 0.45)
        ctx.fill()
    ctx.arc(cx, 400, 300, 0, TAU)
    rgba(ctx, "gold", 0.9)
    ctx.set_line_width(6)
    ctx.stroke()
    ctx.arc(cx, 400, 285, 0, TAU)
    rgba(ctx, "gold_l", 0.8)
    ctx.set_line_width(2)
    ctx.stroke()

    # --- lotus seat
    for row, (n, ln, wd, y, c1, c2) in enumerate([(11, 150, 46, 930, "lotus_d", "lotus"),
                                                    (9, 120, 44, 915, "lotus", "#ffe0e8")]):
        for i in range(n):
            a = (i / (n - 1) - 0.5) * 2.6
            petal_path(ctx, cx + math.sin(a) * 40, y, ln, wd, a)
            lin(ctx, cx, y, cx + math.sin(a) * ln, y - ln, [(0, c1), (1, c2)])
            ctx.fill_preserve()
            rgba(ctx, "lotus_d")
            ctx.set_line_width(2)
            ctx.stroke()
    ellipse(ctx, cx, 935, 260, 26)
    lin(ctx, cx - 260, 0, cx + 260, 0, [(0, "gold_d"), (0.5, "gold_l"), (1, "gold_d")])
    ctx.fill()

    # --- legs (seated, yellow dhoti)
    smooth_path(ctx, [(220, 880), (250, 790), (360, 740), (450, 760), (540, 740), (650, 790),
                      (680, 880), (560, 905), (450, 895), (340, 905)], close=True)
    lin(ctx, 0, 740, 0, 905, [(0, "#ffd54a"), (1, "#f2a900")])
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()
    for xx in (300, 380, 520, 600):  # dhoti folds
        ctx.move_to(xx, 790)
        ctx.curve_to(xx + 10, 830, xx - 10, 860, xx + 5, 890)
        rgba(ctx, "#d18a00", 0.7)
        ctx.set_line_width(3)
        ctx.stroke()
    # gold hem
    ctx.move_to(230, 875)
    ctx.curve_to(330, 905, 570, 905, 670, 875)
    rgba(ctx, "#c0392b")
    ctx.set_line_width(10)
    ctx.stroke()
    # feet
    for fx, d in ((330, -1), (570, 1)):
        ellipse(ctx, fx, 880, 50, 22)
        rgba(ctx, "ganesh")
        ctx.fill_preserve()
        rgba(ctx, OUT)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.arc(fx + d * 18, 878, 7, 0, TAU)
        rgba(ctx, "#c0392b")
        ctx.fill()

    # --- belly / torso
    ellipse(ctx, cx, 640, 175, 150)
    rad(ctx, cx - 40, 600, 10, 200, [(0, "ganesh_l"), (0.6, "ganesh"), (1, "ganesh_d")])
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()
    ctx.arc(cx, 680, 9, 0, TAU)  # navel
    rgba(ctx, "ganesh_d")
    ctx.fill()
    # red sash at waist
    ctx.move_to(275, 720)
    ctx.curve_to(350, 770, 550, 770, 625, 720)
    ctx.line_to(630, 750)
    ctx.curve_to(550, 800, 350, 800, 270, 750)
    ctx.close_path()
    rgba(ctx, "#c0392b")
    ctx.fill_preserve()
    rgba(ctx, "gold")
    ctx.set_line_width(4)
    ctx.stroke()
    # sacred thread (janeu)
    ctx.move_to(330, 520)
    ctx.curve_to(400, 600, 500, 690, 590, 720)
    rgba(ctx, "#fffdf0")
    ctx.set_line_width(5)
    ctx.stroke()

    # --- four arms
    def arm(pts, w0, w1):
        tapered_stroke(ctx, pts, w0, w1)
        lin(ctx, 0, 450, 0, 800, [(0, "ganesh"), (1, "ganesh_d")])
        ctx.fill_preserve()
        rgba(ctx, OUT)
        ctx.set_line_width(3)
        ctx.stroke()

    # lower arms
    arm(bezier_pts((300, 580), (250, 650), (270, 700), (330, 700)), 70, 55)  # holds modak bowl
    arm(bezier_pts((600, 580), (660, 640), (650, 660), (640, 600)), 70, 55)  # blessing hand
    # armbands
    for bx, by, ang in ((268, 640, -0.8), (652, 630, 0.6)):
        ctx.save()
        ctx.translate(bx, by)
        ctx.rotate(ang)
        ellipse(ctx, 0, 0, 34, 9)
        rgba(ctx, "gold")
        ctx.fill_preserve()
        rgba(ctx, "gold_d")
        ctx.set_line_width(2)
        ctx.stroke()
        ctx.restore()
    # lower-left: bowl of modaks
    ellipse(ctx, 345, 712, 70, 22)
    lin(ctx, 275, 0, 415, 0, [(0, "gold_d"), (0.5, "gold_l"), (1, "gold_d")])
    ctx.fill()
    for mx, my in ((315, 690), (345, 680), (375, 690), (330, 668), (360, 668), (345, 650)):
        ctx.move_to(mx - 16, my + 10)
        ctx.curve_to(mx - 16, my - 5, mx - 3, my - 18, mx, my - 24)
        ctx.curve_to(mx + 3, my - 18, mx + 16, my - 5, mx + 16, my + 10)
        ctx.close_path()
        lin(ctx, mx, my - 24, mx, my + 10, [(0, "#fff3d6"), (1, "#f0c27a")])
        ctx.fill_preserve()
        rgba(ctx, "#b8860b")
        ctx.set_line_width(1.5)
        ctx.stroke()
    # lower-right: abhaya mudra palm (raised, facing viewer)
    ctx.save()
    ctx.translate(640, 585)
    smooth_path(ctx, [(-30, 25), (-34, -10), (-28, -48), (-14, -60), (0, -64), (14, -60), (27, -48),
                      (34, -10), (30, 25)], close=True)
    rgba(ctx, "ganesh_l")
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()
    for fx in (-18, -6, 6, 18):
        ctx.move_to(fx, -30)
        ctx.line_to(fx, -56)
        rgba(ctx, "ganesh_d", 0.7)
        ctx.set_line_width(2)
        ctx.stroke()
    ctx.arc(0, -5, 9, 0, TAU)  # om/kumkum dot in palm
    rgba(ctx, "#c0392b")
    ctx.fill()
    ctx.restore()

    # --- necklace on chest
    for i, (rr, col, lw) in enumerate(((120, "gold", 12), (140, "#c0392b", 5), (155, "gold_l", 4))):
        ctx.arc(cx, 470, rr, 0.18 * math.pi, 0.82 * math.pi)
        rgba(ctx, col)
        ctx.set_line_width(lw)
        ctx.stroke()
    ctx.arc(cx, 600, 16, 0, TAU)
    rgba(ctx, "#2e8b57")
    ctx.fill_preserve()
    rgba(ctx, "gold")
    ctx.set_line_width(5)
    ctx.stroke()

    # --- ears (big fan shapes)
    for d in (-1, 1):
        ex = cx + d * 150
        smooth_path(ctx, [(cx + d * 75, 300), (cx + d * 160, 245), (cx + d * 235, 280), (cx + d * 248, 370),
                          (cx + d * 215, 445), (cx + d * 145, 470), (cx + d * 92, 430)], close=True)
        rad(ctx, ex, 370, 20, 160, [(0, "ganesh_d"), (0.75, "ganesh"), (1, "ganesh_l")])
        ctx.fill_preserve()
        rgba(ctx, OUT)
        ctx.set_line_width(3)
        ctx.stroke()
        # decorated ear edge
        smooth_path(ctx, [(cx + d * 165, 258), (cx + d * 228, 290), (cx + d * 238, 370), (cx + d * 207, 436),
                          (cx + d * 147, 460)])
        rgba(ctx, "gold")
        ctx.set_line_width(7)
        ctx.stroke()
        for t in np.linspace(0.1, 0.9, 6):
            a = -0.9 + t * 2.0
            ctx.arc(ex + d * math.cos(a) * 80, 362 + math.sin(a) * 88, 5, 0, TAU)
            rgba(ctx, "#c0392b")
            ctx.fill()

    # upper arms raised
    arm(bezier_pts((320, 540), (230, 520), (190, 440), (200, 360)), 70, 52)
    arm(bezier_pts((580, 540), (670, 520), (710, 440), (700, 360)), 70, 52)
    for bx, by, ang in ((222, 468, 0.5), (678, 468, -0.5)):
        ctx.save()
        ctx.translate(bx, by)
        ctx.rotate(ang)
        ellipse(ctx, 0, 0, 34, 9)
        rgba(ctx, "gold")
        ctx.fill_preserve()
        rgba(ctx, "gold_d")
        ctx.set_line_width(2)
        ctx.stroke()
        ctx.restore()
    # upper-left hand: axe (parashu)
    ctx.move_to(200, 360)
    ctx.line_to(205, 250)
    rgba(ctx, "#8d5524")
    ctx.set_line_width(10)
    ctx.stroke()
    ctx.move_to(205, 260)
    ctx.curve_to(150, 250, 130, 300, 150, 330)
    ctx.curve_to(170, 300, 190, 290, 207, 300)
    ctx.close_path()
    lin(ctx, 130, 250, 210, 330, [(0, "#e0e0e0"), (1, "#8a8a8a")])
    ctx.fill_preserve()
    rgba(ctx, "#555555")
    ctx.set_line_width(2)
    ctx.stroke()
    ellipse(ctx, 200, 362, 28, 24)
    rgba(ctx, "ganesh")
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()
    # upper-right hand: lotus
    ellipse(ctx, 700, 362, 28, 24)
    rgba(ctx, "ganesh")
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()
    ctx.move_to(700, 350)
    ctx.line_to(705, 300)
    rgba(ctx, "leaf")
    ctx.set_line_width(6)
    ctx.stroke()
    draw_lotus(ctx, 706, 305, 75)
    # --- head
    smooth_path(ctx, [(cx - 110, 330), (cx - 100, 250), (cx - 40, 215), (cx + 40, 215), (cx + 100, 250),
                      (cx + 110, 330), (cx + 90, 400), (cx + 45, 430), (cx - 45, 430), (cx - 90, 400)],
                close=True)
    rad(ctx, cx - 25, 300, 10, 170, [(0, "ganesh_l"), (0.65, "ganesh"), (1, "ganesh_d")])
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(3)
    ctx.stroke()

    # --- trunk (curling to viewer's right, toward the modak – classic vamamukhi is to his left)
    centre = np.vstack([bezier_pts((cx, 330), (cx + 5, 440), (cx - 10, 520), (cx - 60, 560), 30),
                        bezier_pts((cx - 60, 560), (cx - 100, 590), (cx - 150, 570), (cx - 140, 530), 20)[1:]])
    tapered_stroke(ctx, centre, 86, 26)
    lin(ctx, cx - 60, 360, cx + 40, 360, [(0, "ganesh_d"), (0.45, "ganesh_l"), (1, "ganesh")])
    ctx.fill()
    le, ri = tapered_edges(centre, 86, 26)
    for edge in (le[6:], ri[6:]):
        smooth_path(ctx, list(map(tuple, edge)))
        rgba(ctx, OUT)
        ctx.set_line_width(3)
        ctx.stroke()
    ctx.arc(*centre[-1], 13, 0, TAU)
    rgba(ctx, "ganesh_d")
    ctx.fill()
    for i in range(4, 26, 3):  # trunk rings
        p, q = centre[i], centre[i + 1]
        d = q - p
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        w = (86 - (86 - 26) * i / len(centre)) * 0.42
        ctx.move_to(*(p + n * w))
        ctx.curve_to(*(p + n * w * 0.3 + d * 1.5), *(p - n * w * 0.3 + d * 1.5), *(p - n * w))
        rgba(ctx, "ganesh_d", 0.8)
        ctx.set_line_width(2.5)
        ctx.stroke()
    # tusk (single, on his right)
    ctx.move_to(cx + 38, 395)
    ctx.curve_to(cx + 70, 420, cx + 85, 450, cx + 78, 470)
    ctx.curve_to(cx + 62, 450, cx + 48, 430, cx + 30, 418)
    ctx.close_path()
    rgba(ctx, "#fffaf0")
    ctx.fill_preserve()
    rgba(ctx, "#bfae8a")
    ctx.set_line_width(2)
    ctx.stroke()
    # eyes: gentle, half-closed in blessing
    for d in (-1, 1):
        ex = cx + d * 48
        ctx.move_to(ex - 22, 318)
        ctx.curve_to(ex - 10, 330, ex + 10, 330, ex + 22, 318)
        ctx.curve_to(ex + 10, 310, ex - 10, 310, ex - 22, 318)
        rgba(ctx, "#ffffff")
        ctx.fill()
        ellipse(ctx, ex + d * 2, 321, 8, 7)
        rgba(ctx, "#2b1a12")
        ctx.fill()
        ctx.move_to(ex - 24, 316)
        ctx.curve_to(ex - 10, 306, ex + 10, 306, ex + 24, 316)
        rgba(ctx, "#2b1a12")
        ctx.set_line_width(3.5)
        ctx.stroke()
        ctx.move_to(ex - 22, 296)
        ctx.curve_to(ex - 8, 288, ex + 8, 288, ex + 22, 296)
        ctx.set_line_width(3)
        ctx.stroke()
    # tilak (tripundra + red dot)
    for i, yy in enumerate((258, 268, 278)):
        ctx.move_to(cx - 45, yy)
        ctx.curve_to(cx - 15, yy - 6, cx + 15, yy - 6, cx + 45, yy)
        rgba(ctx, "#fff7e0")
        ctx.set_line_width(4)
        ctx.stroke()
    ellipse(ctx, cx, 272, 8, 12)
    rgba(ctx, "#d00000")
    ctx.fill()

    # --- crown (mukut)
    ctx.move_to(cx - 105, 240)
    ctx.curve_to(cx - 95, 170, cx - 60, 110, cx - 25, 70)
    ctx.line_to(cx, 25)
    ctx.line_to(cx + 25, 70)
    ctx.curve_to(cx + 60, 110, cx + 95, 170, cx + 105, 240)
    ctx.curve_to(cx + 40, 222, cx - 40, 222, cx - 105, 240)
    ctx.close_path()
    lin(ctx, cx - 105, 0, cx + 105, 0, [(0, "gold_d"), (0.3, "gold"), (0.5, "#fff1b8"), (0.7, "gold"),
                                        (1, "gold_d")])
    ctx.fill_preserve()
    rgba(ctx, "#6d4c06")
    ctx.set_line_width(3)
    ctx.stroke()
    for yy, rr, n in ((205, 9, 7), (160, 8, 5), (118, 7, 3)):
        span = {7: 150, 5: 105, 3: 60}[n]
        for i in range(n):
            xx = cx - span / 2 + span * i / max(n - 1, 1)
            ctx.arc(xx, yy, rr, 0, TAU)
            rgba(ctx, ("#c0392b", "#1e8449", "#2471a3")[i % 3])
            ctx.fill_preserve()
            rgba(ctx, "#fff1b8")
            ctx.set_line_width(2)
            ctx.stroke()
    # crown band
    ctx.move_to(cx - 108, 238)
    ctx.curve_to(cx - 40, 218, cx + 40, 218, cx + 108, 238)
    rgba(ctx, "#c0392b")
    ctx.set_line_width(9)
    ctx.stroke()
    # finial
    ctx.arc(cx, 22, 13, 0, TAU)
    rgba(ctx, "#fff1b8")
    ctx.fill_preserve()
    rgba(ctx, "gold_d")
    ctx.set_line_width(2)
    ctx.stroke()
    return finish(surf)


# ============================================================================ RITUAL OBJECTS
def diya_body(ctx, cx, cy, s):
    """Clay diya; flame is drawn separately (animated). (cx, cy) = wick point."""
    ctx.move_to(cx - s * 1.0, cy)
    ctx.curve_to(cx - s * 0.9, cy + s * 0.55, cx + s * 0.6, cy + s * 0.6, cx + s * 1.0, cy)
    ctx.curve_to(cx + s * 0.6, cy + s * 0.1, cx - s * 0.6, cy + s * 0.12, cx - s * 1.0, cy)
    ctx.close_path()
    lin(ctx, cx, cy, cx, cy + s * 0.6, [(0, "#e07a3f"), (1, "#8c3b13")])
    ctx.fill()
    # rim decoration
    ctx.move_to(cx - s * 0.95, cy + s * 0.04)
    ctx.curve_to(cx - s * 0.5, cy + s * 0.2, cx + s * 0.5, cy + s * 0.2, cx + s * 0.95, cy + s * 0.04)
    rgba(ctx, "gold_l", 0.9)
    ctx.set_line_width(max(1, s * 0.05))
    ctx.stroke()
    for i in range(5):
        ctx.arc(cx - s * 0.5 + i * s * 0.25, cy + s * 0.32, s * 0.04, 0, TAU)
        rgba(ctx, "yellow")
        ctx.fill()
    # oil / wick
    ellipse(ctx, cx - s * 0.75, cy + s * 0.02, s * 0.1, s * 0.05)
    rgba(ctx, "#5a2a0a")
    ctx.fill()


@lru_cache(None)
def diya(size=200):
    surf, ctx = cairo_canvas(size, size // 2)
    diya_body(ctx, size / 2 + size * 0.09, size * 0.12, size * 0.4)
    return finish(surf)


@lru_cache(None)
def flame_frames(size=120, n=16):
    """Looping flame sprites, anchored bottom-centre."""
    frames = []
    for i in range(n):
        ph = TAU * i / n
        surf, ctx = cairo_canvas(size, size * 2)
        cx, by = size / 2, size * 1.9
        rad(ctx, cx, by - size * 0.6, 0, size * 0.55, [(0, "#ffcc55", 0.55), (1, "#ff7700", 0)])
        ctx.arc(cx, by - size * 0.6, size * 0.55, 0, TAU)
        ctx.fill()
        sway = math.sin(ph) * size * 0.06 + math.sin(ph * 2 + 1) * size * 0.03
        ht = size * (1.05 + 0.08 * math.sin(ph * 3))
        for scale, c1, c2 in ((1.0, "#ff6a00", "#ffb000"), (0.7, "#ffb000", "#ffe066"), (0.4, "#fff3b0", "#ffffff")):
            w = size * 0.2 * scale
            h = ht * scale
            ctx.move_to(cx, by)
            ctx.curve_to(cx - w * 1.6, by - h * 0.2, cx - w * 0.6 + sway, by - h * 0.65, cx + sway * 1.6, by - h)
            ctx.curve_to(cx + w * 0.6 + sway, by - h * 0.65, cx + w * 1.6, by - h * 0.2, cx, by)
            lin(ctx, cx, by, cx, by - h, [(0, c1), (1, c2)])
            ctx.fill()
        frames.append(finish(surf))
    return frames


@lru_cache(None)
def bell(size=140):
    """Temple bell hanging from a chain; anchor at the top for swinging."""
    Wd, Hd = 140, 300
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    for i in range(7):  # chain links
        ellipse(ctx, 70, 10 + i * 18, 6, 10)
        rgba(ctx, "gold_d")
        ctx.set_line_width(3)
        ctx.stroke()
    ctx.move_to(70, 135)
    ctx.curve_to(40, 140, 38, 175, 35, 215)
    ctx.curve_to(30, 245, 15, 255, 10, 262)
    ctx.line_to(130, 262)
    ctx.curve_to(125, 255, 110, 245, 105, 215)
    ctx.curve_to(102, 175, 100, 140, 70, 135)
    ctx.close_path()
    lin(ctx, 10, 0, 130, 0, [(0, "gold_d"), (0.35, "#ffe9a8"), (0.6, "gold"), (1, "#5f4508")])
    ctx.fill()
    for yy, ww in ((180, 31), (232, 44)):
        ctx.move_to(70 - ww, yy)
        ctx.line_to(70 + ww, yy)
        rgba(ctx, "gold_d")
        ctx.set_line_width(4)
        ctx.stroke()
    ellipse(ctx, 70, 262, 60, 9)
    lin(ctx, 10, 0, 130, 0, [(0, "#6d4c06"), (0.5, "#ffe9a8"), (1, "#6d4c06")])
    ctx.fill()
    ctx.arc(70, 280, 11, 0, TAU)  # clapper
    rgba(ctx, "gold_d")
    ctx.fill()
    ctx.arc(70, 128, 12, 0, TAU)
    rgba(ctx, "gold")
    ctx.fill()
    return finish(surf)


@lru_cache(None)
def kalash(size=300):
    Wd, Hd = 300, 400
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    cx = 150
    # mango leaves
    for i in range(7):
        a = (i / 6 - 0.5) * 2.2
        draw_leaf(ctx, cx + math.sin(a) * 20, 150, 110, 30, a, color="leaf" if i % 2 else "leaf_d")
    # coconut
    ellipse(ctx, cx, 110, 52, 62)
    rad(ctx, cx - 15, 90, 5, 70, [(0, "#a0612b"), (1, "#5e3415")])
    ctx.fill()
    for i in range(6):
        ctx.move_to(cx - 40 + i * 16, 60)
        ctx.curve_to(cx - 30 + i * 12, 90, cx - 35 + i * 14, 130, cx - 40 + i * 16, 165)
        rgba(ctx, "#3d2210", 0.6)
        ctx.set_line_width(2)
        ctx.stroke()
    # red cloth & swastika-free mangal mark
    ctx.move_to(cx - 60, 160)
    ctx.curve_to(cx - 20, 140, cx + 20, 140, cx + 60, 160)
    ctx.line_to(cx + 55, 180)
    ctx.line_to(cx - 55, 180)
    ctx.close_path()
    rgba(ctx, "red")
    ctx.fill()
    # pot
    ctx.move_to(cx - 45, 175)
    ctx.line_to(cx + 45, 175)
    ctx.curve_to(cx + 45, 200, cx + 120, 230, cx + 120, 290)
    ctx.curve_to(cx + 120, 350, cx + 60, 375, cx, 375)
    ctx.curve_to(cx - 60, 375, cx - 120, 350, cx - 120, 290)
    ctx.curve_to(cx - 120, 230, cx - 45, 200, cx - 45, 175)
    ctx.close_path()
    lin(ctx, cx - 120, 0, cx + 120, 0, [(0, "#7a4a05"), (0.3, "#e8b830"), (0.45, "#fff0b0"), (0.7, "#c9962a"),
                                        (1, "#6d4006")])
    ctx.fill()
    ctx.move_to(cx - 112, 260)
    ctx.curve_to(cx - 40, 280, cx + 40, 280, cx + 112, 260)
    rgba(ctx, "red")
    ctx.set_line_width(12)
    ctx.stroke()
    for i in range(9):
        ctx.arc(cx - 96 + i * 24, 272 - abs(i - 4) * 2.5, 4, 0, TAU)
        rgba(ctx, "#fff0b0")
        ctx.fill()
    # kumkum mark
    ellipse(ctx, cx, 320, 11, 16)
    rgba(ctx, "red")
    ctx.fill()
    ellipse(ctx, cx, 185, 50, 9)
    rgba(ctx, "gold_d")
    ctx.fill()
    return finish(surf)


@lru_cache(None)
def tilak_thali(size=520):
    """Puja thali for the Tilak ceremony: kumkum, akshat, diya, coconut, sweets."""
    Wd, Hd = 520, 300
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    cx, cy = 260, 190
    ellipse(ctx, cx, cy + 18, 250, 80)
    rgba(ctx, "#000000", 0.18)
    ctx.fill()
    ellipse(ctx, cx, cy, 250, 80)
    lin(ctx, 10, 0, 510, 0, [(0, "#8a6a12"), (0.3, "#f3d98b"), (0.5, "#fff6d5"), (0.7, "#d8b04a"), (1, "#7a5a0a")])
    ctx.fill()
    ellipse(ctx, cx, cy, 215, 64)
    lin(ctx, 40, 0, 480, 0, [(0, "#b8901f"), (0.5, "#ffe9a8"), (1, "#a07c15")])
    ctx.fill()
    for i in range(24):  # engraved rim dots
        a = TAU * i / 24
        ctx.arc(cx + math.cos(a) * 232, cy + math.sin(a) * 72, 3.5, 0, TAU)
        rgba(ctx, "#7a5a0a")
        ctx.fill()
    # kumkum bowl
    ellipse(ctx, cx - 120, cy - 10, 48, 18)
    rgba(ctx, "#b8901f")
    ctx.fill()
    ellipse(ctx, cx - 120, cy - 14, 40, 13)
    rgba(ctx, "#d00000")
    ctx.fill()
    # rice (akshat) bowl
    ellipse(ctx, cx - 40, cy + 22, 48, 18)
    rgba(ctx, "#b8901f")
    ctx.fill()
    ellipse(ctx, cx - 40, cy + 18, 40, 13)
    rgba(ctx, "#fffaf0")
    ctx.fill()
    rng = random.Random(3)
    for _ in range(40):
        ellipse(ctx, cx - 40 + rng.uniform(-32, 32), cy + 18 + rng.uniform(-9, 9), 3, 1.6)
        rgba(ctx, "#e8dcc0")
        ctx.fill()
    # coconut wrapped in red cloth
    ellipse(ctx, cx + 70, cy - 30, 48, 40)
    rad(ctx, cx + 55, cy - 45, 5, 50, [(0, "#a0612b"), (1, "#5e3415")])
    ctx.fill()
    ctx.move_to(cx + 25, cy - 20)
    ctx.curve_to(cx + 60, cy + 10, cx + 90, cy + 5, cx + 118, cy - 25)
    ctx.line_to(cx + 112, cy - 5)
    ctx.curve_to(cx + 85, cy + 25, cx + 55, cy + 25, cx + 28, cy)
    ctx.close_path()
    rgba(ctx, "red")
    ctx.fill()
    # sweets
    for i, (mx, my) in enumerate(((cx + 140, cy + 25), (cx + 168, cy + 20), (cx + 154, cy + 5))):
        ctx.arc(mx, my, 15, 0, TAU)
        rad(ctx, mx - 4, my - 4, 2, 16, [(0, "#ffe08a"), (1, "#f0a020")])
        ctx.fill()
    # diya on the thali (flame added at runtime)
    diya_body(ctx, cx + 30, cy + 30, 42)
    # tilak mark floating above (U-shaped vaishnava + red line)
    return finish(surf)


@lru_cache(None)
def havan_kund(size=420):
    """Square havan kund with stepped tiers (flames added at runtime)."""
    Wd, Hd = 420, 220
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    tiers = [(30, 150, 360, 60, "#8c3b13", "#c8602a"), (60, 110, 300, 50, "#a54a1a", "#e07a3f"),
             (90, 75, 240, 42, "#b85a22", "#f0955a")]
    for x, y, w, h, c1, c2 in tiers:
        # front face
        ctx.rectangle(x, y, w, h)
        lin(ctx, 0, y, 0, y + h, [(0, c2), (1, c1)])
        ctx.fill()
        ctx.move_to(x, y)
        ctx.line_to(x + 30, y - 22)
        ctx.line_to(x + w - 30, y - 22)
        ctx.line_to(x + w, y)
        ctx.close_path()
        rgba(ctx, c2)
        ctx.fill()
        ctx.rectangle(x, y, w, 6)
        rgba(ctx, "#fff0b0", 0.8)
        ctx.fill()
        for i in range(int(w // 30)):  # haldi-kumkum dots
            ctx.arc(x + 15 + i * 30, y + h / 2 + 3, 5, 0, TAU)
            rgba(ctx, "#ffd000" if i % 2 else "#d00000")
            ctx.fill()
    # wood sticks inside
    for i in range(5):
        ctx.move_to(150 + i * 28, 62)
        ctx.line_to(170 + i * 22, 38)
        rgba(ctx, "#4a2a10")
        ctx.set_line_width(9)
        ctx.stroke()
    return finish(surf)


@lru_cache(None)
def janeu(size=360):
    """Sacred thread (yagyopaveet) – a loop of three twisted cotton strands with a brahma-granthi knot."""
    surf, ctx = cairo_canvas(size, size)
    c = size / 2
    for s_i, col in enumerate(("#fffdf2", "#f3e7c7", "#fff8e0")):
        pts = []
        for i in range(181):
            t = TAU * i / 180
            r = size * 0.38 + math.sin(t * 24 + s_i * 2.1) * size * 0.012
            pts.append((c + math.cos(t) * r * 0.8, c + math.sin(t) * r))
        smooth_path(ctx, pts, close=True)
        rgba(ctx, "#bfa76a")
        ctx.set_line_width(size * 0.022)
        ctx.stroke_preserve()
        rgba(ctx, col)
        ctx.set_line_width(size * 0.014)
        ctx.stroke()
    ctx.arc(c, c + size * 0.38, size * 0.04, 0, TAU)
    rgba(ctx, "#f3e7c7")
    ctx.fill_preserve()
    rgba(ctx, "#bfa76a")
    ctx.set_line_width(2)
    ctx.stroke()
    # tiny haldi marks on thread
    for t in (0.3, 1.3, 2.5, 3.6, 4.8, 5.7):
        ctx.arc(c + math.cos(t) * size * 0.38 * 0.8, c + math.sin(t) * size * 0.38, size * 0.012, 0, TAU)
        rgba(ctx, "#f4c430")
        ctx.fill()
    return finish(surf)


@lru_cache(None)
def om_symbol(size=200, color="gold"):
    """Om rendered from the Devanagari glyph via typography to guarantee correct form."""
    from typography import text_sprite
    return text_sprite("ॐ", "deva_bold", size, color=P.get(color, color))


@lru_cache(None)
def haldi_bowls(size=600):
    """Brass urli bowls heaped with turmeric paste, marigolds around."""
    Wd, Hd = 600, 300
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    rng = random.Random(7)
    for bx, by, s in ((180, 200, 1.0), (420, 205, 0.9), (300, 235, 1.1)):
        ellipse(ctx, bx, by + 50 * s, 120 * s, 22 * s)
        rgba(ctx, "#000000", 0.15)
        ctx.fill()
        # paste mound
        ctx.move_to(bx - 95 * s, by)
        ctx.curve_to(bx - 70 * s, by - 70 * s, bx + 70 * s, by - 70 * s, bx + 95 * s, by)
        ctx.close_path()
        rad(ctx, bx - 20 * s, by - 40 * s, 5, 100 * s, [(0, "#ffe680"), (0.6, "#f4c430"), (1, "#d99a00")])
        ctx.fill()
        # bowl
        ctx.move_to(bx - 110 * s, by - 5 * s)
        ctx.curve_to(bx - 100 * s, by + 55 * s, bx + 100 * s, by + 55 * s, bx + 110 * s, by - 5 * s)
        ctx.close_path()
        lin(ctx, bx - 110 * s, 0, bx + 110 * s, 0, [(0, "#8a5a0a"), (0.35, "#f3d98b"), (0.55, "#fff6d5"),
                                                    (1, "#7a4a05")])
        ctx.fill()
        ellipse(ctx, bx, by - 5 * s, 110 * s, 14 * s)
        lin(ctx, bx - 110 * s, 0, bx + 110 * s, 0, [(0, "#b8901f"), (0.5, "#fff0b0"), (1, "#a07c15")])
        ctx.set_line_width(5 * s)
        ctx.stroke()
    for _ in range(9):
        draw_marigold(ctx, rng.uniform(40, 560), rng.uniform(235, 285), rng.uniform(16, 24),
                      base=rng.choice(["marigold", "yellow"]), rng=rng)
    return finish(surf)


@lru_cache(None)
def dhol(size=260):
    Wd, Hd = 260, 200
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    ctx.move_to(40, 40)
    ctx.curve_to(110, 25, 150, 25, 220, 40)
    ctx.line_to(220, 160)
    ctx.curve_to(150, 175, 110, 175, 40, 160)
    ctx.close_path()
    lin(ctx, 0, 30, 0, 170, [(0, "#d00000"), (0.5, "#ff4d4d"), (1, "#7a0000")])
    ctx.fill()
    for i in range(8):  # ropes
        x = 50 + i * 22
        ctx.move_to(x, 42)
        ctx.line_to(x + 11, 158)
        ctx.move_to(x + 22, 42)
        ctx.line_to(x + 11, 158)
        rgba(ctx, "#ffd700")
        ctx.set_line_width(3)
        ctx.stroke()
    for x in (40, 220):
        ellipse(ctx, x, 100, 22, 62)
        lin(ctx, x - 20, 0, x + 20, 0, [(0, "#f6e7c1"), (1, "#c9a96e")])
        ctx.fill_preserve()
        rgba(ctx, "#6b3e0a")
        ctx.set_line_width(5)
        ctx.stroke()
    # sticks
    ctx.move_to(10, 10)
    ctx.curve_to(20, 40, 30, 60, 38, 80)
    ctx.move_to(250, 5)
    ctx.line_to(225, 70)
    rgba(ctx, "#6b3e0a")
    ctx.set_line_width(6)
    ctx.stroke()
    return finish(surf)


@lru_cache(None)
def horse(size=640):
    """Decorated wedding mare (ghodi), side view facing right, with jhool, bridle and plume."""
    Wd, Hd = 640, 560
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    BODY, SH, OUT, MANE = "#fbf7f0", "#e3d8c8", "#9a8a74", "#4a3426"

    def leg(top, knee, hoof, w0, w1, col_top, col_bot):
        pts = np.vstack([bezier_pts(top, (top[0], (top[1] + knee[1]) / 2), knee, knee, 14),
                         bezier_pts(knee, knee, (hoof[0], (knee[1] + hoof[1]) / 2), hoof, 14)[1:]])
        tapered_stroke(ctx, pts, w0, w1)
        lin(ctx, 0, top[1], 0, hoof[1], [(0, col_top), (1, col_bot)])
        ctx.fill_preserve()
        rgba(ctx, OUT)
        ctx.set_line_width(2.2)
        ctx.stroke()
        ctx.move_to(hoof[0] - 13, hoof[1] - 2)
        ctx.line_to(hoof[0] + 15, hoof[1] - 2)
        ctx.line_to(hoof[0] + 17, hoof[1] + 16)
        ctx.line_to(hoof[0] - 13, hoof[1] + 16)
        ctx.close_path()
        rgba(ctx, "#3b2a1e")
        ctx.fill()
        ellipse(ctx, hoof[0], hoof[1] - 14, 16, 6)  # ghungroo anklet
        rgba(ctx, "gold")
        ctx.fill()

    # far legs
    leg((205, 320), (198, 420), (205, 515), 44, 22, SH, "#cfc2ae")
    leg((440, 320), (452, 420), (440, 515), 42, 22, SH, "#cfc2ae")
    # tail
    ctx.move_to(128, 235)
    ctx.curve_to(70, 250, 55, 360, 85, 440)
    ctx.curve_to(95, 400, 105, 330, 150, 270)
    ctx.close_path()
    rgba(ctx, MANE)
    ctx.fill()
    # neck + head (one shape)
    smooth_path(ctx, [(380, 215), (420, 160), (465, 105), (510, 62), (548, 52), (578, 70), (604, 110),
                      (628, 150), (636, 176), (622, 192), (596, 190), (566, 172), (540, 160), (522, 200),
                      (510, 260), (488, 320), (430, 320)], close=True, k=0.6)
    lin(ctx, 420, 60, 620, 200, [(0, BODY), (0.7, BODY), (1, SH)])
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(2.5)
    ctx.stroke()
    # body
    smooth_path(ctx, [(135, 250), (190, 205), (300, 200), (400, 195), (470, 215), (505, 275), (480, 330),
                      (400, 350), (290, 352), (190, 345), (140, 310)], close=True, k=0.6)
    lin(ctx, 0, 195, 0, 352, [(0, BODY), (0.6, BODY), (1, SH)])
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(2.5)
    ctx.stroke()
    # mane along crest
    for i in range(9):
        t = i / 8
        x, y = 395 + t * 125, 205 - t * 145
        petal_path(ctx, x, y, 46 - t * 14, 14, -2.2 + t * 0.25)
        rgba(ctx, MANE)
        ctx.fill()
    # ear, eye, nostril
    poly_path(ctx, [(532, 60), (540, 22), (556, 58)])
    rgba(ctx, BODY)
    ctx.fill_preserve()
    rgba(ctx, OUT)
    ctx.set_line_width(2)
    ctx.stroke()
    ellipse(ctx, 566, 98, 7, 5)
    rgba(ctx, "#2b1a12")
    ctx.fill()
    ctx.arc(624, 172, 4, 0, TAU)
    rgba(ctx, "#8a7a66")
    ctx.fill()
    # near legs
    leg((235, 320), (228, 425), (238, 520), 50, 24, BODY, SH)
    leg((462, 310), (470, 425), (466, 520), 48, 24, BODY, SH)
    # bridle with gold studs
    for (x0, y0), (x1, y1) in (((548, 58), (612, 180)), ((528, 128), (626, 152)), ((540, 160), (520, 210))):
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        rgba(ctx, "#a4161a")
        ctx.set_line_width(7)
        ctx.stroke()
        for t in np.linspace(0, 1, 6):
            ctx.arc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 3.5, 0, TAU)
            rgba(ctx, "gold_l")
            ctx.fill()
    # head plume (kalgi)
    for i in range(7):
        petal_path(ctx, 542, 46, 78 - abs(i - 3) * 8, 13, -0.75 + i * 0.22)
        rgba(ctx, ("#c1121f", "#ffd700", "#d6336c", "#fff0b0", "#d6336c", "#ffd700", "#c1121f")[i])
        ctx.fill()
    ctx.arc(542, 50, 10, 0, TAU)
    rgba(ctx, "gold")
    ctx.fill()
    # forehead jewel
    ctx.arc(584, 96, 9, 0, TAU)
    rgba(ctx, "gold")
    ctx.fill()
    # jhool (saddle cloth)
    smooth_path(ctx, [(225, 210), (330, 196), (430, 202), (452, 280), (436, 372), (330, 388),
                      (220, 378), (205, 290)], close=True)
    lin(ctx, 0, 196, 0, 388, [(0, "#c1121f"), (1, "#6b0f1a")])
    ctx.fill_preserve()
    rgba(ctx, "gold")
    ctx.set_line_width(9)
    ctx.stroke()
    smooth_path(ctx, [(245, 230), (330, 220), (412, 225), (428, 285), (414, 352), (330, 364),
                      (240, 356), (228, 290)], close=True)
    rgba(ctx, "gold_l", 0.9)
    ctx.set_line_width(2.5)
    ctx.stroke()
    for i in range(13):  # tassels
        x = 228 + i * 17
        ctx.move_to(x, 380)
        ctx.line_to(x, 402)
        rgba(ctx, "gold")
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.arc(x, 405, 5, 0, TAU)
        rgba(ctx, "#ffd700")
        ctx.fill()
    for j, yy in enumerate((258, 330)):
        for i in range(7):
            ctx.arc(262 + i * 24 + (12 if j else 0), yy, 6, 0, TAU)
            rgba(ctx, "#ffd700")
            ctx.fill()
    draw_marigold(ctx, 330, 292, 24)
    # saddle
    ctx.move_to(285, 205)
    ctx.curve_to(305, 168, 365, 168, 385, 203)
    rgba(ctx, "gold_d")
    ctx.set_line_width(15)
    ctx.stroke()
    # marigold garland around neck
    for t in np.linspace(0, 1, 9):
        x = 478 + 30 * math.sin(t * math.pi) + t * 20
        y = 205 + t * 110
        draw_marigold(ctx, x, y, 11, base="marigold" if int(t * 8) % 2 else "yellow")
    return finish(surf)


@lru_cache(None)
def doli(size=560):
    """Bridal palanquin (doli) for Gauna."""
    Wd, Hd = 560, 440
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    # poles
    ctx.rectangle(0, 250, 560, 16)
    lin(ctx, 0, 250, 0, 266, [(0, "#d8b04a"), (1, "#7a5a0a")])
    ctx.fill()
    for x in (0, 544):
        ctx.rectangle(x, 246, 16, 24)
        rgba(ctx, "gold_d")
        ctx.fill()
    # body box
    ctx.rectangle(150, 150, 260, 210)
    lin(ctx, 150, 0, 410, 0, [(0, "#6b0f1a"), (0.5, "#a4161a"), (1, "#6b0f1a")])
    ctx.fill()
    ctx.rectangle(150, 150, 260, 210)
    rgba(ctx, "gold")
    ctx.set_line_width(6)
    ctx.stroke()
    # arched window with curtain
    ctx.move_to(200, 330)
    ctx.line_to(200, 220)
    ctx.curve_to(200, 180, 360, 180, 360, 220)
    ctx.line_to(360, 330)
    ctx.close_path()
    lin(ctx, 200, 0, 360, 0, [(0, "#ffb3c6"), (0.5, "#ffe0e8"), (1, "#ffb3c6")])
    ctx.fill_preserve()
    rgba(ctx, "gold")
    ctx.set_line_width(5)
    ctx.stroke()
    for i in range(6):
        ctx.move_to(212 + i * 28, 205)
        ctx.curve_to(205 + i * 28, 260, 220 + i * 28, 290, 212 + i * 28, 330)
        rgba(ctx, "#e78aa5", 0.7)
        ctx.set_line_width(3)
        ctx.stroke()
    # dome roof
    ctx.move_to(130, 152)
    ctx.curve_to(150, 60, 230, 30, 280, 20)
    ctx.curve_to(330, 30, 410, 60, 430, 152)
    ctx.close_path()
    lin(ctx, 130, 0, 430, 0, [(0, "#8a6a12"), (0.5, "#fff0b0"), (1, "#8a6a12")])
    ctx.fill()
    for i in range(9):
        x = 140 + i * 35
        ctx.arc(x, 156, 16, 0, math.pi)
        rgba(ctx, "#c0392b")
        ctx.fill()
    ctx.arc(280, 14, 12, 0, TAU)
    rgba(ctx, "gold_l")
    ctx.fill()
    # legs/base
    for x in (160, 385):
        ctx.rectangle(x, 360, 16, 50)
        rgba(ctx, "gold_d")
        ctx.fill()
    # garlands
    for x in (150, 410):
        for j in range(5):
            draw_marigold(ctx, x, 175 + j * 34, 15, base="marigold" if j % 2 else "yellow")
    return finish(surf)


# ============================================================================ ARCHITECTURE
def scallop_arch_path(ctx, x0, y_spring, x1, apex_y, lobes=7, depth=0.12):
    """Multifoil (Mughal/Rajput) arch from left spring point to right, through apex."""
    cx = (x0 + x1) / 2
    w = (x1 - x0) / 2
    pts = []
    n = 240
    for i in range(n + 1):
        t = i / n
        a = math.pi * (1 - t)
        # ogee-ish profile: circle stretched to apex with pointed top
        x = cx + math.cos(a) * w
        base = y_spring - math.sin(a) * (y_spring - apex_y) * (0.86 + 0.14 * math.sin(a) ** 6)
        lobe = abs(math.sin(t * lobes * math.pi)) * depth * w * math.sin(a) ** 0.5
        pts.append((x + math.cos(a) * lobe * 0.3, base + lobe))
    for p in pts:  # line_to: continue the caller's current path (no new sub-path)
        ctx.line_to(*p)


@lru_cache(None)
def card_frame(theme="maroon"):
    """Full-frame (1080x1920) invitation card: arch-topped ivory panel with gold borders."""
    from compositor import W, H
    surf, ctx = cairo_canvas(W, H)
    m = 64
    top = 300
    col = {"maroon": ("#6b0f1a", "#a4161a"), "saffron": ("#8a3b00", "#d35400"),
           "haldi": ("#8a5a00", "#e0a000"), "pink": ("#7a1044", "#c2185b"),
           "red": ("#5c0a10", "#b3121f"), "peach": ("#7a2e2e", "#c0605a")}[theme]

    def panel(inset):
        x0, x1 = m + inset, W - m - inset
        ctx.new_path()
        ctx.move_to(x0, H - m - inset)
        ctx.line_to(x0, top + 180)
        scallop_arch_path(ctx, x0, top + 180, x1, top - 120 + inset, lobes=9, depth=0.06)
        ctx.line_to(x1, H - m - inset)
        ctx.close_path()

    panel(0)
    rgba(ctx, "#fff8e7")
    ctx.fill()
    # paper texture gradient
    panel(0)
    rad(ctx, W / 2, H * 0.45, 100, 1100, [(0, "#fffdf6", 0.0), (1, "#f0dcb0", 0.6)])
    ctx.fill()
    for inset, lw, c in ((0, 10, "gold"), (16, 3, col[1]), (26, 2, "gold")):
        panel(inset)
        lin(ctx, 0, 0, W, H, [(0, "#8a6a12"), (0.3, "#f3d98b"), (0.5, "#c9a227"), (0.7, "#f3d98b"), (1, "#8a6a12")]) \
            if c == "gold" else rgba(ctx, c)
        ctx.set_line_width(lw)
        ctx.stroke()
    # corner ornaments at the bottom
    for sx in (1, -1):
        ctx.save()
        ctx.translate(W / 2 + sx * (W / 2 - m - 26), H - m - 26)
        ctx.scale(sx, 1)
        for i in range(3):
            petal_path(ctx, 0, 0, 70 - i * 15, 16, -math.pi / 4 + (i - 1) * 0.5)
            rgba(ctx, "gold", 0.9)
            ctx.fill()
        ctx.restore()
    return finish(surf)


@lru_cache(None)
def outer_pattern(theme="maroon"):
    """Rich background behind the card: deep colour with gold jaal pattern."""
    from compositor import W, H
    surf, ctx = cairo_canvas(W, H)
    col = {"maroon": ("#4a0a12", "#7a1420"), "saffron": ("#5e2600", "#9a4300"),
           "haldi": ("#5e3d00", "#a77400"), "pink": ("#4a0a2a", "#8a1550"),
           "red": ("#3d0610", "#8b0f1c"), "peach": ("#4d1a1a", "#8a3a3a")}[theme]
    rad(ctx, W / 2, H / 2, 100, 1300, [(0, col[1]), (1, col[0])])
    ctx.paint()
    step = 120
    for j in range(-1, H // step + 2):
        for i in range(-1, W // step + 2):
            x = i * step + (step / 2 if j % 2 else 0)
            y = j * step
            for a in range(4):
                petal_path(ctx, x, y, 34, 12, a * math.pi / 2)
                rgba(ctx, "gold", 0.16)
                ctx.fill()
            ctx.arc(x, y, 5, 0, TAU)
            rgba(ctx, "gold", 0.3)
            ctx.fill()
    return finish(surf)


@lru_cache(None)
def mandala(size=900, color="gold", alpha=1.0, rings=6):
    surf, ctx = cairo_canvas(size, size)
    c = size / 2
    R = size / 2 * 0.98
    for r_i in range(rings):
        r = R * (r_i + 1) / rings
        n = 8 + r_i * 4
        for i in range(n):
            a = TAU * i / n
            petal_path(ctx, c + math.cos(a) * r * 0.55, c + math.sin(a) * r * 0.55, r * 0.42, r * 0.11 + 6,
                       a + math.pi / 2)
            rgba(ctx, color, alpha * (0.9 - r_i * 0.08))
            ctx.set_line_width(2.2)
            ctx.stroke()
        ctx.arc(c, c, r * 0.97, 0, TAU)
        rgba(ctx, color, alpha * 0.5)
        ctx.set_line_width(1.5)
        ctx.stroke()
        for i in range(n * 2):
            a = TAU * (i + 0.5) / (n * 2)
            ctx.arc(c + math.cos(a) * r * 0.97, c + math.sin(a) * r * 0.97, 3, 0, TAU)
            rgba(ctx, color, alpha * 0.8)
            ctx.fill()
    return finish(surf)


@lru_cache(None)
def divider(width=520, color="gold"):
    """Ornamental text divider: scrolls either side of a central motif."""
    h = 60
    surf, ctx = cairo_canvas(width, h)
    c = width / 2
    for s in (-1, 1):
        ctx.move_to(c + s * 34, h / 2)
        ctx.curve_to(c + s * 90, h / 2 - 18, c + s * 130, h / 2 + 18, c + s * (width / 2 - 20), h / 2)
        rgba(ctx, color)
        ctx.set_line_width(2.5)
        ctx.stroke()
        for k in range(3):
            xx = c + s * (70 + k * 60)
            ctx.arc(xx, h / 2 - (6 if k % 2 else -6), 4, 0, TAU)
            rgba(ctx, color)
            ctx.fill()
        petal_path(ctx, c + s * (width / 2 - 18), h / 2, 16, 6, s * math.pi / 2)
        rgba(ctx, color)
        ctx.fill()
    for a in range(8):
        petal_path(ctx, c, h / 2, 22, 7, a * math.pi / 4)
        rgba(ctx, color, 0.95 if a % 2 == 0 else 0.6)
        ctx.fill()
    ctx.arc(c, h / 2, 5, 0, TAU)
    rgba(ctx, "#c0392b" if color == "gold" else "#ffffff")
    ctx.fill()
    return finish(surf)


@lru_cache(None)
def toran(width=1080, drop=260, strands=11, seed=5):
    """Marigold garland toran with mango leaves, hanging from top edge."""
    surf, ctx = cairo_canvas(width, drop + 60)
    rng = random.Random(seed)
    # top swag
    for i in range(int(width / 34) + 1):
        x = i * 34
        y = 22 + math.sin(i * 0.7) * 3
        draw_marigold(ctx, x, y, 19, base="marigold" if i % 3 else "yellow", rng=rng)
    for i in range(int(width / 70) + 1):
        draw_leaf(ctx, i * 70 + 20, 40, 52, 16, math.pi + (0.3 if i % 2 else -0.3), color="leaf")
    # hanging strands
    for s in range(strands):
        x = (s + 0.5) * width / strands
        edge = abs(x - width / 2) / (width / 2)
        L = drop * (0.18 + 0.82 * edge ** 1.6) * (0.85 + 0.15 * (s % 2))
        n = max(1, int(L / 26))
        for j in range(n):
            draw_marigold(ctx, x, 45 + j * 26, 13, base=("marigold", "yellow", "red")[(j + s) % 3], rng=rng)
        draw_rose(ctx, x, 50 + n * 26, 13)
    return finish(surf)


@lru_cache(None)
def string_lights_bulb(size=40, color="#ffd27a"):
    return spark_sprite(size, color)


@lru_cache(None)
def mandap(size=980, night=True):
    """Wedding mandap: four carved pillars, domed canopy, drapes and marigold strings."""
    Wd, Hd = 980, 900
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    # floor platform
    ctx.move_to(60, 820)
    ctx.line_to(920, 820)
    ctx.line_to(980, 900)
    ctx.line_to(0, 900)
    ctx.close_path()
    lin(ctx, 0, 820, 0, 900, [(0, "#f3d98b"), (1, "#a07c15")])
    ctx.fill()
    ctx.rectangle(60, 805, 860, 18)
    rgba(ctx, "#c0392b")
    ctx.fill()
    # back pillars (thin, darker)
    for x in (230, 750):
        ctx.rectangle(x - 14, 250, 28, 560)
        lin(ctx, x - 14, 0, x + 14, 0, [(0, "#8a6a12"), (0.5, "#d8b04a"), (1, "#6d4c06")])
        ctx.fill()
    # drapes back
    for x0, x1 in ((120, 490), (490, 860)):
        ctx.move_to(x0, 240)
        ctx.curve_to(x0 + 60, 420, x1 - 60, 420, x1, 240)
        ctx.line_to(x1, 250)
        ctx.curve_to(x1 - 70, 330, x0 + 70, 330, x0, 250)
        ctx.close_path()
        lin(ctx, 0, 240, 0, 420, [(0, "#fff0b0"), (1, "#e8b830")])
        ctx.fill()
    # front pillars
    for x in (120, 860):
        # base
        ctx.rectangle(x - 45, 740, 90, 70)
        lin(ctx, x - 45, 0, x + 45, 0, [(0, "#7a5a0a"), (0.5, "#fff0b0"), (1, "#7a5a0a")])
        ctx.fill()
        # shaft
        ctx.rectangle(x - 26, 260, 52, 480)
        lin(ctx, x - 26, 0, x + 26, 0, [(0, "#8a6a12"), (0.35, "#fff0b0"), (0.6, "#d8b04a"), (1, "#6d4c06")])
        ctx.fill()
        for yy in range(300, 740, 60):
            ctx.rectangle(x - 30, yy, 60, 10)
            rgba(ctx, "#c0392b")
            ctx.fill()
            ctx.arc(x, yy + 30, 7, 0, TAU)
            rgba(ctx, "#fff0b0")
            ctx.fill()
        # capital
        ctx.move_to(x - 50, 260)
        ctx.curve_to(x - 40, 230, x + 40, 230, x + 50, 260)
        ctx.close_path()
        rgba(ctx, "gold")
        ctx.fill()
        # banana plant at the base
        for i in range(4):
            draw_leaf(ctx, x + (40 if x < 490 else -40), 800, 170, 40, (-0.6 + i * 0.4) * (1 if x < 490 else -1),
                      color="leaf" if i % 2 else "leaf_d")
    # canopy beam
    ctx.rectangle(70, 225, 840, 34)
    lin(ctx, 0, 225, 0, 259, [(0, "#fff0b0"), (1, "#b8901f")])
    ctx.fill()
    for i in range(29):
        ctx.arc(85 + i * 29, 262, 13, 0, math.pi)
        rgba(ctx, "#c0392b")
        ctx.fill()
    # dome
    ctx.move_to(100, 228)
    ctx.curve_to(150, 90, 330, 40, 490, 30)
    ctx.curve_to(650, 40, 830, 90, 880, 228)
    ctx.close_path()
    lin(ctx, 100, 0, 880, 0, [(0, "#6b0f1a"), (0.5, "#c1121f"), (1, "#6b0f1a")])
    ctx.fill()
    for i in range(7):  # dome ribs
        t = i / 6
        ctx.move_to(100 + 780 * t, 228)
        ctx.curve_to(100 + 780 * t, 150, 490 + (t - 0.5) * 300, 60, 490, 30)
        rgba(ctx, "gold", 0.9)
        ctx.set_line_width(4)
        ctx.stroke()
    ctx.arc(490, 22, 18, 0, TAU)
    rgba(ctx, "gold_l")
    ctx.fill()
    ctx.move_to(490, 4)
    ctx.line_to(490, -20)
    ctx.stroke()
    # marigold strings between pillars
    for s in range(9):
        x = 170 + s * 80
        L = 4 + (s % 3) * 2
        for j in range(L):
            draw_marigold(ctx, x, 290 + j * 24, 11, base="marigold" if (j + s) % 2 else "yellow")
    return finish(surf)


@lru_cache(None)
def palace(width=1080, height=760, tone="ivory"):
    """Palace skyline with domes, chhatris and arched windows (lit at night)."""
    surf, ctx = cairo_canvas(width, height)
    base_c = {"ivory": ("#f7e8cf", "#e2c99d", "#c9a96e"), "dusk": ("#3a1020", "#5a1a30", "#2a0814"),
              "champagne": ("#e9d3a3", "#c9a96e", "#8a6a3a")}[tone]
    rng = random.Random(11)

    def dome(cx, by, r, h):
        ctx.move_to(cx - r, by)
        ctx.curve_to(cx - r * 1.15, by - h * 0.7, cx - r * 0.2, by - h * 0.9, cx, by - h)
        ctx.curve_to(cx + r * 0.2, by - h * 0.9, cx + r * 1.15, by - h * 0.7, cx + r, by)
        ctx.close_path()
        lin(ctx, cx - r, 0, cx + r, 0, [(0, base_c[2]), (0.4, base_c[0]), (1, base_c[1])])
        ctx.fill()
        ctx.move_to(cx, by - h)
        ctx.line_to(cx, by - h - r * 0.4)
        rgba(ctx, "gold")
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.arc(cx, by - h - r * 0.42, 5, 0, TAU)
        ctx.fill()

    # towers & main body
    blocks = [(60, 330, 160, "t"), (220, 420, 640, "m"), (860, 330, 160, "t")]
    for x, y, w, kind in blocks:
        ctx.rectangle(x, y, w, height - y)
        lin(ctx, x, 0, x + w, 0, [(0, base_c[1]), (0.5, base_c[0]), (1, base_c[2])])
        ctx.fill()
        ctx.rectangle(x - 8, y - 14, w + 16, 16)
        rgba(ctx, base_c[2])
        ctx.fill()
        for i in range(int(w // 18)):
            ctx.rectangle(x + i * 18 + 3, y - 30, 10, 16)
            rgba(ctx, base_c[2])
            ctx.fill()
    dome(540, 420, 190, 250)
    dome(140, 330, 75, 120)
    dome(940, 330, 75, 120)
    for cx in (330, 750):
        dome(cx, 420, 55, 90)
    # arched windows
    for x, y, w, kind in blocks:
        cols = int(w // 80)
        for row in range(3):
            for c in range(cols):
                wx = x + (c + 0.5) * w / cols
                wy = y + 70 + row * 110
                ctx.move_to(wx - 20, wy + 60)
                ctx.line_to(wx - 20, wy)
                ctx.curve_to(wx - 20, wy - 22, wx, wy - 34, wx, wy - 40)
                ctx.curve_to(wx, wy - 34, wx + 20, wy - 22, wx + 20, wy)
                ctx.line_to(wx + 20, wy + 60)
                ctx.close_path()
                if tone == "dusk":
                    lin(ctx, 0, wy - 40, 0, wy + 60, [(0, "#ffe9a8"), (1, "#ff9a3c")])
                else:
                    rgba(ctx, base_c[2], 0.8)
                ctx.fill()
    # grand entrance
    ctx.move_to(470, height)
    ctx.line_to(470, 560)
    ctx.curve_to(470, 500, 540, 470, 540, 450)
    ctx.curve_to(540, 470, 610, 500, 610, 560)
    ctx.line_to(610, height)
    ctx.close_path()
    lin(ctx, 0, 450, 0, height, [(0, "#ffe9a8"), (1, "#ff9a3c")] if tone == "dusk" else
        [(0, base_c[2]), (1, base_c[1])])
    ctx.fill()
    return finish(surf)


@lru_cache(None)
def grand_arch(width=1080, height=1920, depth=0, color="#e9d3a3", lw=5):
    """Gold line-art palace entrance arch (for the luxury video's parallax corridor)."""
    surf, ctx = cairo_canvas(width, height)
    x0, x1 = 40, width - 40
    spring = height * 0.42
    apex = height * 0.08
    for inset, a, w in ((0, 1.0, lw), (22, 0.7, lw * 0.5), (40, 0.45, lw * 0.4)):
        ctx.new_path()
        ctx.move_to(x0 + inset, height)
        ctx.line_to(x0 + inset, spring)
        scallop_arch_path(ctx, x0 + inset, spring, x1 - inset, apex + inset, lobes=11, depth=0.05)
        ctx.line_to(x1 - inset, height)
        rgba(ctx, color, a)
        ctx.set_line_width(w)
        ctx.stroke()
    # pillar detailing
    for x in (x0 + 20, x1 - 20):
        for y in np.arange(spring + 60, height, 140):
            ctx.arc(x, y, 10, 0, TAU)
            rgba(ctx, color, 0.7)
            ctx.set_line_width(2)
            ctx.stroke()
            petal_path(ctx, x, y - 12, 30, 9, 0)
            ctx.stroke()
    # hanging jhoomar (chandelier) at the apex
    cx = width / 2
    ctx.move_to(cx, apex + 40)
    ctx.line_to(cx, apex + 160)
    rgba(ctx, color, 0.8)
    ctx.set_line_width(2)
    ctx.stroke()
    for i, rr in enumerate((70, 50, 30)):
        y = apex + 170 + i * 40
        ellipse(ctx, cx, y, rr, rr * 0.25)
        rgba(ctx, color, 0.8)
        ctx.set_line_width(2.5)
        ctx.stroke()
        for j in range(8):
            a = TAU * j / 8
            ctx.move_to(cx + math.cos(a) * rr, y + math.sin(a) * rr * 0.25)
            ctx.line_to(cx + math.cos(a) * rr, y + math.sin(a) * rr * 0.25 + 30)
            ctx.stroke()
    return finish(surf)


# ============================================================================ CHARACTERS
@lru_cache(None)
def groom(size=420):
    """Groom: cream/gold sherwani, maroon dupatta, red-gold safa with kalgi (faceless flat style)."""
    Wd, Hd = 420, 1000
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    cx = 210
    SKIN, SKIN_D = P["skin"], P["skin_d"]
    SH, SH_D, GOLD = "#f6ead0", "#d9c49a", P["gold"]
    # shadow
    ellipse(ctx, cx, 985, 120, 14)
    rgba(ctx, "#000000", 0.18)
    ctx.fill()
    # churidar legs
    for d in (-1, 1):
        ctx.move_to(cx + d * 10, 740)
        ctx.line_to(cx + d * 70, 740)
        ctx.curve_to(cx + d * 66, 820, cx + d * 60, 900, cx + d * 56, 955)
        ctx.line_to(cx + d * 18, 955)
        ctx.curve_to(cx + d * 16, 880, cx + d * 12, 800, cx + d * 10, 740)
        ctx.close_path()
        lin(ctx, cx, 740, cx + d * 70, 740, [(0, "#efe3c8"), (1, "#d8c7a2")])
        ctx.fill()
        for yy in range(860, 950, 14):  # churidar gathers
            ctx.move_to(cx + d * 18, yy)
            ctx.curve_to(cx + d * 30, yy + 5, cx + d * 45, yy + 5, cx + d * 57, yy)
            rgba(ctx, "#bfae86", 0.6)
            ctx.set_line_width(2)
            ctx.stroke()
        # mojari
        ctx.move_to(cx + d * 14, 950)
        ctx.curve_to(cx + d * 14, 975, cx + d * 60, 980, cx + d * 90, 968)
        ctx.curve_to(cx + d * 70, 955, cx + d * 62, 948, cx + d * 58, 945)
        ctx.close_path()
        lin(ctx, cx, 945, cx + d * 90, 975, [(0, "#8a6a12"), (1, "#f3d98b")])
        ctx.fill()
    # sherwani body (long coat)
    ctx.move_to(cx - 95, 330)
    ctx.curve_to(cx - 110, 450, cx - 115, 600, cx - 125, 790)
    ctx.curve_to(cx - 40, 805, cx + 40, 805, cx + 125, 790)
    ctx.curve_to(cx + 115, 600, cx + 110, 450, cx + 95, 330)
    ctx.curve_to(cx + 60, 300, cx - 60, 300, cx - 95, 330)
    ctx.close_path()
    lin(ctx, cx - 125, 0, cx + 125, 0, [(0, SH_D), (0.35, SH), (0.65, "#fff6e3"), (1, SH_D)])
    ctx.fill()
    # gold embroidery: hem border, placket, butis
    ctx.move_to(cx - 122, 770)
    ctx.curve_to(cx - 40, 785, cx + 40, 785, cx + 122, 770)
    rgba(ctx, GOLD)
    ctx.set_line_width(14)
    ctx.stroke()
    ctx.move_to(cx, 312)
    ctx.line_to(cx, 795)
    rgba(ctx, GOLD)
    ctx.set_line_width(12)
    ctx.stroke()
    for yy in range(345, 560, 34):
        ctx.arc(cx, yy, 6, 0, TAU)
        rgba(ctx, "#fff0b0")
        ctx.fill_preserve()
        rgba(ctx, "gold_d")
        ctx.set_line_width(1.5)
        ctx.stroke()
    rng = random.Random(2)
    for _ in range(46):
        x = cx + rng.uniform(-100, 100)
        y = rng.uniform(380, 750)
        if abs(x - cx) < 20:
            continue
        for a in range(4):
            petal_path(ctx, x, y, 8, 3.5, a * math.pi / 2)
            rgba(ctx, GOLD, 0.85)
            ctx.fill()
    # maroon dupatta over right shoulder, diagonal to left hip
    ctx.move_to(cx + 70, 318)
    ctx.curve_to(cx + 40, 420, cx - 40, 540, cx - 112, 620)
    ctx.line_to(cx - 118, 680)
    ctx.curve_to(cx - 20, 580, cx + 70, 460, cx + 105, 345)
    ctx.close_path()
    lin(ctx, cx - 110, 680, cx + 100, 320, [(0, "#5c0a10"), (0.5, "#a4161a"), (1, "#6b0f1a")])
    ctx.fill()
    ctx.move_to(cx + 72, 320)
    ctx.curve_to(cx + 42, 422, cx - 38, 542, cx - 112, 622)
    rgba(ctx, GOLD)
    ctx.set_line_width(5)
    ctx.stroke()
    # hanging dupatta end with tassels
    ctx.move_to(cx + 100, 345)
    ctx.curve_to(cx + 120, 430, cx + 118, 520, cx + 112, 600)
    ctx.line_to(cx + 140, 600)
    ctx.curve_to(cx + 140, 500, cx + 135, 410, cx + 105, 335)
    ctx.close_path()
    lin(ctx, 0, 330, 0, 600, [(0, "#a4161a"), (1, "#6b0f1a")])
    ctx.fill()
    ctx.rectangle(cx + 110, 590, 32, 10)
    rgba(ctx, GOLD)
    ctx.fill()
    # arms (sleeves) – relaxed, hands joined slightly in front
    for d in (-1, 1):
        pts = bezier_pts((cx + d * 92, 340), (cx + d * 130, 430), (cx + d * 130, 540), (cx + d * 80, 600), 30)
        tapered_stroke(ctx, pts, 58, 44)
        lin(ctx, cx + d * 60, 0, cx + d * 140, 0, [(0, SH), (1, SH_D)])
        ctx.fill()
        ctx.move_to(*pts[-4])
        ctx.line_to(*pts[-1])
        rgba(ctx, GOLD)
        ctx.set_line_width(46)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        ctx.stroke()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # hands clasped
    ellipse(ctx, cx - 40, 612, 26, 22)
    ellipse(ctx, cx + 40, 612, 26, 22)
    rgba(ctx, SKIN)
    ctx.fill()
    # sword hilt hint – skip; pearl mala
    for strand, (rr, yy) in enumerate(((70, 330), (88, 335), (106, 340))):
        for i in range(17):
            a = math.pi * (0.15 + 0.7 * i / 16)
            ctx.arc(cx + math.cos(a) * rr, yy + math.sin(a) * rr * 1.4, 4.5 - strand * 0.5, 0, TAU)
            rgba(ctx, "#fffdf5")
            ctx.fill_preserve()
            rgba(ctx, "#d4c7a8")
            ctx.set_line_width(1)
            ctx.stroke()
    ctx.arc(cx, 340 + 106 * 1.4, 10, 0, TAU)
    rgba(ctx, "#1e8449")
    ctx.fill_preserve()
    rgba(ctx, GOLD)
    ctx.set_line_width(3)
    ctx.stroke()
    # bandhgala collar
    ctx.rectangle(cx - 38, 285, 76, 34)
    lin(ctx, cx - 38, 0, cx + 38, 0, [(0, "#8a6a12"), (0.5, "#f3d98b"), (1, "#8a6a12")])
    ctx.fill()
    # neck
    ctx.rectangle(cx - 24, 250, 48, 40)
    rgba(ctx, SKIN_D)
    ctx.fill()
    # face (faceless oval)
    ellipse(ctx, cx, 205, 56, 68)
    rad(ctx, cx - 15, 190, 5, 80, [(0, "#efbf98"), (1, SKIN)])
    ctx.fill()
    for d in (-1, 1):  # ears
        ellipse(ctx, cx + d * 56, 210, 9, 16)
        rgba(ctx, SKIN_D)
        ctx.fill()
    # beard-less jawline shadow
    ctx.move_to(cx - 50, 225)
    ctx.curve_to(cx - 30, 280, cx + 30, 280, cx + 50, 225)
    rgba(ctx, SKIN_D, 0.25)
    ctx.set_line_width(3)
    ctx.stroke()
    # tilak on forehead
    ctx.move_to(cx, 165)
    ctx.line_to(cx, 185)
    rgba(ctx, "#d00000")
    ctx.set_line_width(5)
    ctx.stroke()
    # safa (turban)
    ctx.move_to(cx - 70, 180)
    ctx.curve_to(cx - 85, 120, cx - 60, 70, cx, 62)
    ctx.curve_to(cx + 60, 70, cx + 90, 110, cx + 75, 180)
    ctx.curve_to(cx + 30, 158, cx - 30, 158, cx - 70, 180)
    ctx.close_path()
    lin(ctx, cx - 85, 0, cx + 90, 0, [(0, "#7a0c18"), (0.45, "#c1121f"), (1, "#6b0f1a")])
    ctx.fill()
    for i in range(5):  # diagonal folds with gold
        ctx.move_to(cx - 75 + i * 10, 172 - i * 22)
        ctx.curve_to(cx - 20, 150 - i * 20, cx + 30, 140 - i * 20, cx + 80, 150 - i * 18)
        rgba(ctx, GOLD if i % 2 == 0 else "#ff8a8a", 0.9)
        ctx.set_line_width(4 if i % 2 == 0 else 2)
        ctx.stroke()
    # turban tail (turra) behind
    ctx.move_to(cx + 70, 130)
    ctx.curve_to(cx + 120, 150, cx + 120, 230, cx + 100, 280)
    ctx.line_to(cx + 82, 270)
    ctx.curve_to(cx + 95, 220, cx + 92, 170, cx + 66, 150)
    ctx.close_path()
    rgba(ctx, "#a4161a")
    ctx.fill()
    # kalgi brooch + plume
    for i in range(5):
        petal_path(ctx, cx + 25, 95, 60 - abs(i - 2) * 8, 9, -0.35 + i * 0.12)
        rgba(ctx, "#fffdf5" if i % 2 else "#f3d98b")
        ctx.fill()
    ctx.arc(cx + 25, 100, 15, 0, TAU)
    rgba(ctx, GOLD)
    ctx.fill()
    ctx.arc(cx + 25, 100, 7, 0, TAU)
    rgba(ctx, "#1e8449")
    ctx.fill()
    for i in range(5):  # pearl drops from kalgi
        ctx.arc(cx + 10 + i * 7, 122 + abs(i - 2) * 3, 3.5, 0, TAU)
        rgba(ctx, "#fffdf5")
        ctx.fill()
    return finish(surf)


@lru_cache(None)
def bride(size=460):
    """Bride: red/maroon lehenga with gold border, dupatta over head, jewellery (faceless flat style)."""
    Wd, Hd = 460, 1000
    k = size / Wd
    surf, ctx = cairo_canvas(int(Wd * k), int(Hd * k))
    ctx.scale(k, k)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    cx = 230
    SKIN, SKIN_D = "#e6ae86", "#c98a62"
    RED, RED_D, RED_L, GOLD = "#b3121f", "#6b0f1a", "#e03a3e", P["gold"]
    ellipse(ctx, cx, 985, 190, 16)
    rgba(ctx, "#000000", 0.18)
    ctx.fill()
    # dupatta back drape (behind body, from head to floor)
    ctx.move_to(cx - 70, 150)
    ctx.curve_to(cx - 150, 300, cx - 190, 700, cx - 200, 960)
    ctx.line_to(cx + 200, 960)
    ctx.curve_to(cx + 190, 700, cx + 150, 300, cx + 70, 150)
    ctx.close_path()
    lin(ctx, cx - 200, 0, cx + 200, 0, [(0, "#8a0f1a", 0.85), (0.5, "#c1121f", 0.6), (1, "#8a0f1a", 0.85)])
    ctx.fill()
    # lehenga skirt (wide flare)
    ctx.move_to(cx - 85, 470)
    ctx.curve_to(cx - 130, 600, cx - 190, 800, cx - 210, 965)
    ctx.curve_to(cx - 80, 985, cx + 80, 985, cx + 210, 965)
    ctx.curve_to(cx + 190, 800, cx + 130, 600, cx + 85, 470)
    ctx.close_path()
    lin(ctx, cx - 210, 0, cx + 210, 0, [(0, RED_D), (0.3, RED), (0.5, RED_L), (0.7, RED), (1, RED_D)])
    ctx.fill()
    # kalis (panel lines)
    for i in range(-4, 5):
        ctx.move_to(cx + i * 18, 480)
        ctx.curve_to(cx + i * 28, 650, cx + i * 40, 800, cx + i * 48, 970)
        rgba(ctx, RED_D, 0.45)
        ctx.set_line_width(2)
        ctx.stroke()
    # gold hem borders
    for yy, ww, lw in ((945, 210, 30), (905, 198, 6), (880, 192, 3)):
        ctx.move_to(cx - ww, yy)
        ctx.curve_to(cx - 80, yy + 20, cx + 80, yy + 20, cx + ww, yy)
        lin(ctx, cx - ww, 0, cx + ww, 0, [(0, "#8a6a12"), (0.5, "#fff0b0"), (1, "#8a6a12")])
        ctx.set_line_width(lw)
        ctx.stroke()
    for i in range(15):  # scallops on hem
        t = i / 14
        x = cx - 200 + 400 * t
        y = 932 + math.sin(t * math.pi) * 14
        ctx.arc(x, y, 12, math.pi, TAU)
        rgba(ctx, "#fff0b0", 0.9)
        ctx.set_line_width(2.5)
        ctx.stroke()
    rng = random.Random(4)
    for _ in range(70):  # gold butis
        y = rng.uniform(520, 860)
        spread = 85 + (y - 470) / 500 * 120
        x = cx + rng.uniform(-spread, spread) * 0.9
        for a in range(5):
            petal_path(ctx, x, y, 9, 4, a * TAU / 5)
            rgba(ctx, GOLD, 0.9)
            ctx.fill()
    # choli
    ctx.move_to(cx - 80, 330)
    ctx.curve_to(cx - 92, 400, cx - 90, 450, cx - 85, 478)
    ctx.line_to(cx + 85, 478)
    ctx.curve_to(cx + 90, 450, cx + 92, 400, cx + 80, 330)
    ctx.curve_to(cx + 40, 305, cx - 40, 305, cx - 80, 330)
    ctx.close_path()
    lin(ctx, cx - 90, 0, cx + 90, 0, [(0, RED_D), (0.5, RED), (1, RED_D)])
    ctx.fill()
    # midriff
    ctx.rectangle(cx - 82, 440, 164, 36)
    rgba(ctx, SKIN)
    ctx.fill()
    ctx.move_to(cx - 85, 478)
    ctx.line_to(cx + 85, 478)
    rgba(ctx, GOLD)
    ctx.set_line_width(12)
    ctx.stroke()
    ctx.move_to(cx - 85, 440)
    ctx.line_to(cx + 85, 440)
    rgba(ctx, GOLD)
    ctx.set_line_width(6)
    ctx.stroke()
    # front dupatta drape across body (left shoulder to right hip)
    ctx.move_to(cx - 85, 330)
    ctx.curve_to(cx - 40, 430, cx + 60, 520, cx + 140, 600)
    ctx.curve_to(cx + 120, 700, cx + 150, 820, cx + 170, 900)
    ctx.line_to(cx + 200, 890)
    ctx.curve_to(cx + 175, 760, cx + 170, 640, cx + 150, 560)
    ctx.curve_to(cx + 70, 480, cx - 10, 390, cx - 50, 318)
    ctx.close_path()
    lin(ctx, cx - 80, 330, cx + 190, 890, [(0, "#d62828", 0.85), (1, "#8a0f1a", 0.9)])
    ctx.fill()
    ctx.move_to(cx - 85, 330)
    ctx.curve_to(cx - 40, 430, cx + 60, 520, cx + 140, 600)
    ctx.curve_to(cx + 120, 700, cx + 150, 820, cx + 170, 900)
    rgba(ctx, GOLD)
    ctx.set_line_width(6)
    ctx.stroke()
    # arms with chooda bangles & mehndi hands, hands folded at waist
    for d in (-1, 1):
        sleeve = bezier_pts((cx + d * 78, 335), (cx + d * 110, 390), (cx + d * 112, 420), (cx + d * 105, 440), 16)
        tapered_stroke(ctx, sleeve, 48, 40)
        rgba(ctx, RED)
        ctx.fill()
        fore = bezier_pts((cx + d * 105, 440), (cx + d * 100, 500), (cx + d * 70, 530), (cx + d * 30, 535), 20)
        tapered_stroke(ctx, fore, 32, 26)
        rgba(ctx, SKIN)
        ctx.fill()
        for j in range(4, 14, 2):  # bangles
            p = fore[j]
            ellipse(ctx, p[0], p[1], 9, 16)
            rgba(ctx, "#c1121f" if j % 4 else "#fffdf5")
            ctx.set_line_width(4)
            ctx.stroke()
        ctx.move_to(*sleeve[-1] + np.array([-d * 20, 0]))
        ctx.line_to(*sleeve[-1] + np.array([d * 20, 0]))
        rgba(ctx, GOLD)
        ctx.set_line_width(8)
        ctx.stroke()
    ellipse(ctx, cx, 538, 38, 22)  # folded hands
    rgba(ctx, "#c97a4a")
    ctx.fill()
    for i in range(3):
        ctx.arc(cx - 15 + i * 15, 538, 4, 0, TAU)
        rgba(ctx, "#8a3b13")
        ctx.fill()
    # neck + jewellery
    ctx.rectangle(cx - 22, 255, 44, 60)
    rgba(ctx, SKIN_D)
    ctx.fill()
    ctx.arc(cx, 290, 42, 0.1 * math.pi, 0.9 * math.pi)  # choker
    rgba(ctx, GOLD)
    ctx.set_line_width(14)
    ctx.stroke()
    for i in range(9):
        a = math.pi * (0.15 + 0.7 * i / 8)
        ctx.arc(cx + math.cos(a) * 42, 290 + math.sin(a) * 42, 5, 0, TAU)
        rgba(ctx, "#c1121f" if i % 2 else "#1e8449")
        ctx.fill()
    ctx.arc(cx, 300, 70, 0.15 * math.pi, 0.85 * math.pi)  # rani haar
    rgba(ctx, "#f3d98b")
    ctx.set_line_width(5)
    ctx.stroke()
    ctx.move_to(cx, 370)
    ctx.line_to(cx - 14, 392)
    ctx.line_to(cx, 412)
    ctx.line_to(cx + 14, 392)
    ctx.close_path()
    rgba(ctx, GOLD)
    ctx.fill()
    # hair
    ellipse(ctx, cx, 200, 64, 76)
    rgba(ctx, "hair")
    ctx.fill()
    # face
    ellipse(ctx, cx, 212, 50, 62)
    rad(ctx, cx - 12, 200, 5, 70, [(0, "#f2c3a0"), (1, SKIN)])
    ctx.fill()
    # hair parting with sindoor & maang tikka
    ctx.move_to(cx, 140)
    ctx.line_to(cx, 165)
    rgba(ctx, "#d00000")
    ctx.set_line_width(4)
    ctx.stroke()
    ctx.move_to(cx, 128)
    ctx.line_to(cx, 168)
    rgba(ctx, GOLD)
    ctx.set_line_width(2.5)
    ctx.stroke()
    ellipse(ctx, cx, 174, 9, 11)
    rgba(ctx, GOLD)
    ctx.fill()
    ctx.arc(cx, 174, 4.5, 0, TAU)
    rgba(ctx, "#c1121f")
    ctx.fill()
    ctx.arc(cx, 196, 4, 0, TAU)  # bindi
    rgba(ctx, "#d00000")
    ctx.fill()
    # nath (nose ring) with chain to ear
    ctx.arc(cx + 10, 232, 12, -0.5 * math.pi, 1.2 * math.pi)
    rgba(ctx, GOLD)
    ctx.set_line_width(2.5)
    ctx.stroke()
    ctx.move_to(cx + 20, 228)
    ctx.curve_to(cx + 35, 215, cx + 45, 205, cx + 52, 195)
    ctx.set_line_width(1.5)
    ctx.stroke()
    # jhumkas
    for d in (-1, 1):
        ex = cx + d * 52
        ctx.arc(ex, 238, 6, 0, TAU)
        rgba(ctx, GOLD)
        ctx.fill()
        ctx.move_to(ex - 12, 262)
        ctx.curve_to(ex - 12, 244, ex + 12, 244, ex + 12, 262)
        ctx.close_path()
        ctx.fill()
        for i in range(4):
            ctx.arc(ex - 9 + i * 6, 266, 2.5, 0, TAU)
            rgba(ctx, "#fffdf5")
            ctx.fill()
    # dupatta over head (front edge framing face)
    ctx.move_to(cx - 95, 330)
    ctx.curve_to(cx - 100, 220, cx - 85, 110, cx, 100)
    ctx.curve_to(cx + 85, 110, cx + 100, 220, cx + 95, 330)
    ctx.line_to(cx + 72, 320)
    ctx.curve_to(cx + 78, 230, cx + 70, 140, cx, 132)
    ctx.curve_to(cx - 70, 140, cx - 78, 230, cx - 72, 320)
    ctx.close_path()
    lin(ctx, cx - 100, 0, cx + 100, 0, [(0, "#8a0f1a"), (0.5, "#d62828"), (1, "#8a0f1a")])
    ctx.fill()
    ctx.move_to(cx - 72, 320)
    ctx.curve_to(cx - 78, 230, cx - 70, 140, cx, 132)
    ctx.curve_to(cx + 70, 140, cx + 78, 230, cx + 72, 320)
    rgba(ctx, GOLD)
    ctx.set_line_width(7)
    ctx.stroke()
    for i in range(13):  # border dots
        t = i / 12
        a = math.pi * (1 - t)
        ctx.arc(cx + math.cos(a) * 82, 230 - math.sin(a) * 110, 3, 0, TAU)
        rgba(ctx, "#fff0b0")
        ctx.fill()
    return finish(surf)


@lru_cache(None)
def couple(size=760):
    """Groom (viewer's left) and bride (viewer's right) standing together."""
    from compositor import place
    g = groom(int(size * 0.47))
    b = bride(int(size * 0.53))
    h = max(g.shape[0], b.shape[0])
    canvas = np.zeros((h, size, 4), np.float32)
    frame_like = canvas[..., :3].copy()
    alpha = canvas[..., 3:].copy()
    out = np.zeros((h, size, 4), np.float32)
    # composite manually to keep alpha
    for spr, x in ((b, size * 0.43), (g, 0)):
        sh, sw = spr.shape[:2]
        x0 = int(x)
        y0 = h - sh
        reg = out[y0:y0 + sh, x0:x0 + sw]
        reg *= 1 - spr[..., 3:4]
        reg += spr
    return out


@lru_cache(None)
def varmala(size=300):
    """A rose-and-jasmine wedding garland."""
    surf, ctx = cairo_canvas(size, size)
    c = size / 2
    for i in range(30):
        a = math.pi * 0.05 + i / 29 * math.pi * 0.9
        x = c + math.cos(a) * size * 0.4
        y = size * 0.1 + math.sin(a) * size * 0.8
        if i % 3 == 0:
            draw_jasmine(ctx, x, y, size * 0.04)
        else:
            draw_rose(ctx, x, y, size * 0.035)
    return finish(surf)


# ============================================================================ EXTRA SCENE PIECES
@lru_cache(None)
def rays(size=1600, n=28, color="#fff1c1", alpha=0.35):
    surf, ctx = cairo_canvas(size, size)
    c = size / 2
    for i in range(n):
        a = TAU * i / n
        w = (0.035 if i % 2 else 0.018)
        ctx.move_to(c, c)
        ctx.line_to(c + math.cos(a - w) * c, c + math.sin(a - w) * c)
        ctx.line_to(c + math.cos(a + w) * c, c + math.sin(a + w) * c)
        ctx.close_path()
        rad(ctx, c, c, 0, c, [(0, color, alpha), (0.6, color, alpha * 0.35), (1, color, 0)])
        ctx.fill()
    return finish(surf)


@lru_cache(None)
def lotus_pond(width=952, height=380, seed=4):
    """Water band with lotus pads and flowers (static base; lotuses are separate sprites)."""
    surf, ctx = cairo_canvas(width, height)
    top = 130
    lin(ctx, 0, top, 0, height, [(0, "#bfe3e6", 0.0), (0.2, "#a7d3d8", 0.9), (1, "#6fa9b5")])
    ctx.rectangle(0, top, width, height - top)
    ctx.fill()
    rng = random.Random(seed)
    for i in range(14):
        y = top + 30 + i * 17
        x = rng.uniform(0, width)
        ellipse(ctx, x, y, rng.uniform(60, 160), 3)
        rgba(ctx, "#ffffff", 0.35)
        ctx.fill()
    for i in range(9):
        draw_lotus_pad(ctx, rng.uniform(30, width - 30), rng.uniform(top + 50, height - 30),
                       rng.uniform(50, 85), rng.uniform(16, 24))
    return finish(surf)


@lru_cache(None)
def lotus_flower(size=200):
    surf, ctx = cairo_canvas(size, int(size * 1.15))
    draw_lotus(ctx, size / 2, size * 1.05, size * 0.5)
    ctx.move_to(size / 2, size * 1.05)
    ctx.line_to(size / 2, size * 1.15)
    rgba(ctx, "leaf_d")
    ctx.set_line_width(4)
    ctx.stroke()
    return finish(surf)


@lru_cache(None)
def rose_bouquet(size=520, seed=6, mirror=False):
    """Lush cluster of roses, buds and leaves (luxury video corners)."""
    surf, ctx = cairo_canvas(size, size)
    rng = random.Random(seed)
    for _ in range(26):
        a = rng.uniform(-math.pi, 0.3)
        r = rng.uniform(0.25, 0.48) * size
        x, y = size * 0.3 + math.cos(a) * r, size * 0.7 + math.sin(a) * r
        draw_leaf(ctx, x, y, rng.uniform(60, 110) * size / 520, rng.uniform(22, 34) * size / 520,
                  a + math.pi / 2, color=rng.choice(["#2f5d3a", "#3a7d44", "#24502b"]))
    for _ in range(9):
        x, y = rng.uniform(0.1, 0.7) * size, rng.uniform(0.35, 0.95) * size
        draw_jasmine(ctx, x, y, rng.uniform(8, 13) * size / 520)
    pal = [("#5c0a1f", "#a4133c", "#ff4d6d"), ("#7a0c24", "#c2185b", "#f48fb1"),
           ("#b76e59", "#e7b7a3", "#fff0ea"), ("#6b0f1a", "#b3121f", "#ef6f6c")]
    pts = [(0.28, 0.72, 0.16), (0.48, 0.78, 0.12), (0.16, 0.52, 0.12), (0.42, 0.56, 0.1), (0.62, 0.88, 0.09),
           (0.1, 0.86, 0.1), (0.3, 0.4, 0.08), (0.58, 0.68, 0.075)]
    for i, (x, y, r) in enumerate(pts):
        c = pal[i % len(pal)]
        draw_rose(ctx, x * size, y * size, r * size, *c)
    data = finish(surf)
    return data[:, ::-1].copy() if mirror else data


@lru_cache(None)
def marigold_cluster(size=420, seed=8, mirror=False):
    """Corner cluster of marigolds, lotus and mango leaves (traditional video)."""
    surf, ctx = cairo_canvas(size, size)
    rng = random.Random(seed)
    for _ in range(18):
        a = rng.uniform(-math.pi, 0.2)
        draw_leaf(ctx, size * 0.3 + math.cos(a) * size * 0.3, size * 0.75 + math.sin(a) * size * 0.3,
                  rng.uniform(70, 120) * size / 420, rng.uniform(20, 30) * size / 420, a + math.pi / 2,
                  color=rng.choice(["leaf", "leaf_d", "leaf_l"]))
    for _ in range(14):
        draw_marigold(ctx, rng.uniform(0.05, 0.6) * size, rng.uniform(0.45, 0.98) * size,
                      rng.uniform(22, 36) * size / 420, base=rng.choice(["marigold", "yellow", "red"]), rng=rng)
    draw_lotus(ctx, size * 0.3, size * 0.75, size * 0.28)
    data = finish(surf)
    return data[:, ::-1].copy() if mirror else data


@lru_cache(None)
def door_panel(w=540, h=1920, side="left"):
    """Ornate palace door panel (burgundy lacquer + gold inlay)."""
    surf, ctx = cairo_canvas(w, h)
    lin(ctx, 0, 0, w, 0, [(0, "#3a0612"), (0.5, "#6b0f1a"), (1, "#3a0612")])
    ctx.rectangle(0, 0, w, h)
    ctx.fill()
    for inset, lw in ((30, 6), (52, 2)):
        ctx.rectangle(inset, inset, w - 2 * inset, h - 2 * inset)
        rgba(ctx, "champagne", 0.9)
        ctx.set_line_width(lw)
        ctx.stroke()
    # arched inner panels
    for y0, y1 in ((120, 900), (1000, 1800)):
        ctx.move_to(90, y1)
        ctx.line_to(90, y0 + 200)
        ctx.curve_to(90, y0 + 60, w / 2, y0 + 20, w / 2, y0)
        ctx.curve_to(w / 2, y0 + 20, w - 90, y0 + 60, w - 90, y0 + 200)
        ctx.line_to(w - 90, y1)
        ctx.close_path()
        rgba(ctx, "champagne", 0.75)
        ctx.set_line_width(3)
        ctx.stroke()
        cy = (y0 + y1) / 2 + 60
        for r_i in range(3):
            for i in range(12):
                a = TAU * i / 12
                petal_path(ctx, w / 2 + math.cos(a) * 30 * r_i, cy + math.sin(a) * 30 * r_i, 60 - r_i * 10, 14, a + math.pi / 2)
                rgba(ctx, "champagne", 0.5)
                ctx.set_line_width(2)
                ctx.stroke()
    # studs
    for y in range(80, h, 120):
        for x in (60, w - 60):
            ctx.arc(x, y, 8, 0, TAU)
            rad(ctx, x - 2, y - 2, 0, 8, [(0, "#fff6d5"), (1, "#8a6a12")])
            ctx.fill()
    # handle ring on the inner edge
    hx = w - 40 if side == "left" else 40
    ctx.arc(hx, h / 2, 34, 0, TAU)
    rgba(ctx, "#e9d3a3")
    ctx.set_line_width(7)
    ctx.stroke()
    return finish(surf)


@lru_cache(None)
def lux_frame(color="#e9d3a3"):
    """Thin champagne-gold full-frame border with corner flourishes (luxury video)."""
    from compositor import W, H
    surf, ctx = cairo_canvas(W, H)
    m = 44
    for inset, lw, a in ((0, 3, 0.95), (12, 1.2, 0.6)):
        x0, y0, x1, y1 = m + inset, m + inset, W - m - inset, H - m - inset
        r = 40
        ctx.new_path()
        ctx.arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi)
        ctx.arc(x1 - r, y0 + r, r, 1.5 * math.pi, TAU)
        ctx.arc(x1 - r, y1 - r, r, 0, 0.5 * math.pi)
        ctx.arc(x0 + r, y1 - r, r, 0.5 * math.pi, math.pi)
        ctx.close_path()
        rgba(ctx, color, a)
        ctx.set_line_width(lw)
        ctx.stroke()
    for sx, sy, cx, cy in ((1, 1, m, m), (-1, 1, W - m, m), (1, -1, m, H - m), (-1, -1, W - m, H - m)):
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(sx, sy)
        for i in range(5):
            petal_path(ctx, 18, 18, 70 - i * 10, 10, math.pi * 0.75 + (i - 2) * 0.32)
            rgba(ctx, color, 0.85)
            ctx.set_line_width(1.6)
            ctx.stroke()
        ctx.arc(18, 18, 7, 0, TAU)
        rgba(ctx, color)
        ctx.fill()
        ctx.restore()
    # top & bottom centre crests
    for y, d in ((m, 1), (H - m, -1)):
        for a in range(7):
            petal_path(ctx, W / 2, y, 34, 9, math.pi / 2 + d * (math.pi / 2) + (a - 3) * 0.35)
            rgba(ctx, color, 0.9)
            ctx.fill()
    return finish(surf)


@lru_cache(None)
def medallion(size=560, color="#e9d3a3"):
    """Ornamental ring placed behind ceremony icons in the luxury video."""
    surf, ctx = cairo_canvas(size, size)
    c = size / 2
    rad(ctx, c, c, 0, c, [(0, "#7a1a2e", 0.65), (0.75, "#3a0612", 0.35), (1, "#3a0612", 0)])
    ctx.arc(c, c, c, 0, TAU)
    ctx.fill()
    for r, lw, a in ((c * 0.9, 2.5, 0.9), (c * 0.84, 1, 0.6), (c * 0.62, 1, 0.35)):
        ctx.arc(c, c, r, 0, TAU)
        rgba(ctx, color, a)
        ctx.set_line_width(lw)
        ctx.stroke()
    for i in range(48):
        a = TAU * i / 48
        petal_path(ctx, c + math.cos(a) * c * 0.9, c + math.sin(a) * c * 0.9, 22 if i % 2 else 34, 7, a + math.pi / 2)
        rgba(ctx, color, 0.8)
        ctx.fill()
    return finish(surf)
