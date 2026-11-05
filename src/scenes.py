"""Scene definitions for both invitation videos.

Video A – Traditional (ivory card, maroon & gold, flat colourful illustrations).
Video B – Luxury Bollywood (dark burgundy, champagne gold, bokeh, light rays, gold-foil icons).

Each scene: duration (s), transition into it, and render(t) -> float32 frame. All motion is a pure
function of t. Text items are kept in `self.texts` so QA can verify placement.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import zlib
from pathlib import Path

import numpy as np

import illustrations as I
from animations import (Bokeh, Fireworks, PetalField, Sparkles, TextItem, bell_angle, draw_flame, ease_in_out,
                        ease_out, ease_out_back, prog, sway_sprite)
from compositor import (H, W, blur, gold_map, hex_rgb, ken_burns, over, place, radial_mask, tint,
                        vertical_gradient, vignette)
from typography import CHAMPAGNE, GOLD_FOIL, text_sprite

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    with open(path or ROOT / "config" / "wedding.json", encoding="utf-8") as f:
        return json.load(f)


def fmt_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B')} {d.year}"


def seed_of(name: str) -> int:
    """Deterministic per-scene seed (str hash() is randomised per process)."""
    return zlib.crc32(name.encode()) % 100000


DARK_GOLD = [(0.0, "#c9a227"), (0.45, "#8a6212"), (0.7, "#b8860b"), (1.0, "#6d4c06")]
ROMAN = ["I", "II", "III", "IV", "V", "VI"]


# ============================================================================ base
class Scene:
    name = "scene"
    duration = 7.0
    transition = "fade"   # transition INTO this scene
    td = 0.8              # transition duration

    def __init__(self, cfg):
        self.cfg = cfg
        self._ready = False
        self.texts: list[TextItem] = []

    def ensure(self):
        if not self._ready:
            self.setup()
            self._ready = True

    def setup(self):
        pass

    def render(self, t: float) -> np.ndarray:
        raise NotImplementedError

    # -- text helpers
    def build(self, lines, y, start, step=0.3, x=W / 2, style="rise", maxw=860, end=None):
        items = []
        for ln in lines:
            if ln is None:
                continue
            if "space" in ln:
                y += ln["space"]
                continue
            if "sprite" in ln:
                spr = ln["sprite"]
            else:
                kw = {k: v for k, v in ln.items()
                      if k not in ("text", "font", "size", "gap", "style", "delay", "shine", "dur")}
                kw.setdefault("max_width", maxw)
                spr = text_sprite(ln["text"], ln["font"], ln["size"], **kw)
            st = start + ln.get("delay", 0.0)
            it = TextItem(spr, x, y, st, ln.get("style", style), dur=ln.get("dur", 0.9), end=end,
                          shine_at=(st + 0.8) if ln.get("shine") else None)
            it.text = ln.get("text")
            items.append(it)
            y += spr.shape[0] + ln.get("gap", 8)
            start += step
        self.texts.extend(items)
        return items, y

    @staticmethod
    def draw_texts(frame, items, t, alpha=1.0):
        for it in items:
            it.draw(frame, t, alpha)


def venue_lines(cfg, key, label_style, line_style, first_bold_style=None):
    v = cfg["addresses"][key]
    out = []
    if key == "tilak_venue":
        lines = v["lines_en"]
        out.append(dict(text=lines[0], **(first_bold_style or label_style)))
        out += [dict(text=l, **line_style) for l in lines[1:]]
    else:
        out.append(dict(text=v["label_en"], **label_style))
        out += [dict(text=l, **line_style) for l in v["lines_en"]]
    return out


def festoon(frame, t, x0, x1, y, sag, n=16, colors=("#ffd27a", "#ff8fab", "#fff1c1", "#9be7ff"), seed=0):
    for i in range(n):
        u = i / (n - 1)
        x = x0 + (x1 - x0) * u
        yy = y + sag * 4 * u * (1 - u)
        tw = 0.55 + 0.45 * math.sin(t * 5 + i * 1.7 + seed)
        place(frame, I.spark_sprite(48, colors[i % len(colors)]), x, yy, scale=0.9, alpha=tw, add=True)
        place(frame, I.spark_sprite(48, "#ffffff"), x, yy, scale=0.3, alpha=tw)


# ============================================================================ VIDEO A – TRADITIONAL
THEME_DARK = {"maroon": "#6b0f1a", "saffron": "#8a3b00", "haldi": "#7a4a00", "pink": "#8a1550",
              "red": "#8b0f1c", "peach": "#7a2e2e"}
A_BODY = "#4a0a12"


class ACard(Scene):
    """Ivory arch card on a patterned background with toran, bells and falling petals."""
    theme = "maroon"
    petal_kind = "mixed"
    petal_n = 22
    bell_rings = (0.6,)

    def setup(self):
        self.bg = I.outer_pattern(self.theme)[..., :3].copy()
        self.card = I.card_frame(self.theme)
        self.mand = I.mandala(860, "gold", 0.55)
        self.toran = I.toran(1080, 300)
        self.bellspr = I.bell(96)
        self.petals = PetalField(self.petal_kind, n=self.petal_n, seed=seed_of(self.name),
                                 speed=(80, 150), size=(0.45, 0.85))
        self.dark = THEME_DARK[self.theme]
        self.content_setup()

    def content_setup(self):
        pass

    def content(self, f, t):
        pass

    def render(self, t):
        f = ken_burns(self.bg, 1.04 + 0.02 * math.sin(t * 0.3), dy=-6 * t)
        over(f, self.card)
        place(f, self.mand, W / 2, 980, rot=t * 5, alpha=0.16)
        self.content(f, t)
        place(f, sway_sprite(self.toran, t, amp=5), W / 2, 0, anchor=(0.5, 0))
        for bx, ph in ((150, 0.0), (930, 0.7)):
            place(f, self.bellspr, bx, 40, rot=bell_angle(t + ph, self.bell_rings), anchor=(0.5, 0.0))
        self.petals.draw(f, t)
        vignette(f, 0.35)
        return f

    # common text styles for card scenes
    def styles(self):
        d = self.dark
        return dict(
            title_hi=dict(font="deva_bold", size=94, color=d, shadow=("#000000", 0.18, 3, 3), shine=True),
            title_en=dict(font="script", size=76, gradient=DARK_GOLD, gap=0),
            date=dict(font="cormorant_bold", size=56, color=A_BODY, gap=2),
            time=dict(font="cinzel_bold", size=50, color=d, gap=14),
            label=dict(font="cinzel", size=30, color="#8a6212", tracking=0.25, gap=6),
            vlabel=dict(font="cormorant_bold", size=50, color=d, gap=4),
            vline=dict(font="cormorant_semi", size=44, color=A_BODY, gap=0),
            hi_body=dict(font="deva_semi", size=48, color=A_BODY),
        )


class AGanesha(Scene):
    name = "A1_ganesha"
    duration = 7.0

    def setup(self):
        bg = vertical_gradient(W, H, [(0, "#2a0408"), (0.45, "#5a0a14"), (1, "#22030a")])
        m = radial_mask(W, H, W / 2, 720, 0, 900)[..., None]
        self.bg = bg * (1 - m * 0.6) + np.array(hex_rgb("#a4161a"), np.float32) * m * 0.6
        self.rays = I.rays(1700, 30, "#ffe08a", 0.3)
        self.mand = I.mandala(1150, "gold", 0.6)
        self.gan = I.ganesha(700)
        self.glow = blur(tint(self.gan, "#ffcc66"), 28)
        self.bellspr = I.bell(120)
        self.diya = I.diya(200)
        self.lotus = I.lotus_flower(170)
        self.spark = Sparkles(70, seed=11, color="#ffe9a8")
        self.petals = PetalField("marigold", n=16, seed=5, speed=(70, 120), size=(0.4, 0.8), start_t=-6)
        sh = self.cfg["texts"]["shloka"]
        st = dict(font="tiro_sa", size=58, gradient=GOLD_FOIL, shadow=("#000000", 0.6, 4, 3), style="wipe",
                  dur=1.4)
        self.items, _ = self.build([dict(text=sh[0], **st, gap=14), dict(text=sh[1], **st)], 1265, 2.4,
                                   step=1.2, maxw=960)

    def render(self, t):
        f = self.bg.copy()
        a = ease_out(prog(t, 0.0, 2.0))
        place(f, self.rays, W / 2, 720, rot=t * 7, alpha=a * 0.9, add=True)
        place(f, self.mand, W / 2, 720, rot=-t * 4, alpha=a * 0.35)
        p = ease_out(prog(t, 0.2, 2.4))
        pulse = 0.55 + 0.25 * math.sin(t * 2.2)
        place(f, self.glow, W / 2, 720, scale=0.88 + 0.12 * p, alpha=p * pulse, add=True)
        place(f, self.gan, W / 2, 720, scale=0.88 + 0.12 * p, alpha=p)
        self.draw_texts(f, self.items, t)
        for i, x in enumerate((70, 225, 380, 700, 855, 1010)):
            bob = 6 * math.sin(t * 1.3 + i)
            place(f, self.lotus, x, 1930 + bob, anchor=(0.5, 1.0), scale=0.95 + 0.1 * (i % 2))
        for x, s in ((180, 0), (900, 1)):
            place(f, self.diya, x, 1700, anchor=(0.5, 0.0))
            draw_flame(f, x - 200 * 0.31 + 200 * 0.09, 1700 + 200 * 0.12 + 2, t, scale=0.55, seed=s)
        for bx, ph in ((130, 0.0), (950, 0.5)):
            place(f, self.bellspr, bx, -10, rot=bell_angle(t + ph, (0.4, 3.6), amp=12), anchor=(0.5, 0.0))
        self.spark.draw(f, t, alpha=a)
        self.petals.draw(f, t, alpha=prog(t, 3.0, 2.0))
        vignette(f, 0.5)
        return f


class AAnnounce(ACard):
    name = "A2_announce"
    duration = 7.0
    transition = "iris"
    theme = "maroon"

    def content_setup(self):
        tx = self.cfg["texts"]
        self.kal = I.kalash(170)
        self.pond = I.lotus_pond(952, 380)
        self.lotus = I.lotus_flower(150)
        d = self.dark
        lines = [
            dict(text=tx["ganesh"], font="deva_bold", size=68, color="#a4161a", shine=True, gap=4),
            dict(sprite=I.divider(460), gap=26),
            dict(text=tx["blessing"], font="deva_semi", size=52, color=A_BODY, gap=0),
            dict(text="With love, blessings and good wishes", font="cormorant_it", size=40, color="#8a6212"),
            dict(space=40),
            dict(text=tx["invite"], font="deva_bold", size=66, color=d, gap=0, shine=True),
            dict(text="You and your family are cordially invited", font="cormorant_it", size=40, color="#8a6212",
                 gap=20),
            dict(sprite=I.divider(460)),
        ]
        self.items, self.text_end = self.build(lines, 520, 0.9, step=0.45)

    def content(self, f, t):
        p = ease_out_back(prog(t, 0.4, 1.0))
        place(f, self.kal, W / 2, 470, anchor=(0.5, 1.0), scale=max(p, 0.01), alpha=min(1, p * 2))
        self.draw_texts(f, self.items, t)
        place(f, self.pond, W / 2, 1856, anchor=(0.5, 1.0), alpha=ease_out(prog(t, 0.2, 1.2)))
        for i, (x, y) in enumerate(((210, 1700), (400, 1760), (680, 1745), (870, 1690))):
            rise = 60 * (1 - ease_out(prog(t, 0.6 + i * 0.25, 1.2)))
            place(f, self.lotus, x, y + rise + 5 * math.sin(t * 1.4 + i), anchor=(0.5, 1.0),
                  alpha=prog(t, 0.6 + i * 0.25, 0.6), rot=2 * math.sin(t + i))


class ACouple(ACard):
    name = "A3_couple"
    duration = 13.0
    transition = "slide"
    theme = "red"
    petal_kind = "rose"
    bell_rings = (0.3,)

    def content_setup(self):
        g, b = self.cfg["groom"], self.cfg["bride"]
        d = self.dark
        name = dict(font="deva_bold", size=72, color=d, shadow=("#000000", 0.15, 3, 3), gap=0, shine=True)
        en = dict(font="script", size=62, gradient=DARK_GOLD, gap=2)
        par = dict(font="deva_semi", size=40, color=A_BODY, gap=0)
        lines = [
            dict(text=f"{g['title_hi']} {g['name_hi']}", **name),
            dict(text=f"{g['title_en']} {g['name_en']}", **en),
            dict(text=f"{g['relation_hi']}: {g['father_hi']} एवं", **par),
            dict(text=g["mother_hi"], **par, delay=-0.1),
            dict(text=f"{g['grand_relation_hi']}: {g['grandfather_hi']}", **par),
            dict(space=6),
            dict(text=self.cfg["texts"]["with_hi"], font="deva_bold", size=48, color="#8a6212", style="pop", gap=0),
            dict(sprite=I.divider(360), gap=6, delay=-0.2),
            dict(text=f"{b['title_hi']} {b['name_hi']}", **name),
            dict(text=f"{b['title_en']} {b['name_en']}", **en),
            dict(text=f"{b['relation_hi']}: {b['father_hi']} एवं", **par),
            dict(text=b["mother_hi"], **par, delay=-0.1),
            dict(text=f"{b['grand_relation_hi']}: {b['grandfather_hi']}", **par),
        ]
        self.items, self.text_end = self.build(lines, 300, 0.6, step=0.55)
        self.groom = I.groom(250)
        self.bride = I.bride(286)
        self.mala = I.varmala(200)
        self.spark = Sparkles(30, seed=31, region=(250, 1250, 830, 1850), color="#fff1c1")

    def content(self, f, t):
        self.draw_texts(f, self.items, t)
        pg = ease_out(prog(t, 5.6, 1.8))
        pb = ease_out(prog(t, 6.0, 1.8))
        breathe = 1 + 0.006 * math.sin(t * 2)
        place(f, self.groom, -200 + (418 + 200) * pg, 1842, anchor=(0.5, 1.0), sy=breathe)
        place(f, self.bride, 1300 - (1300 - 666) * pb, 1842, anchor=(0.5, 1.0), sy=breathe)
        self.spark.draw(f, t, alpha=prog(t, 7.0, 1.0))


class ACeremony(ACard):
    """Generic ceremony card: icon in the arch, details in the middle, illustration band at the bottom."""
    key = "tilak"
    duration = 6.0
    text_y = 500

    def content_setup(self):
        c = next(x for x in self.cfg["ceremonies"] if x["key"] == self.key)
        self.c = c
        s = self.styles()
        lines = [
            dict(text=c["name_hi"], **s["title_hi"], gap=0),
            dict(text=c["name_en"], **s["title_en"]),
            dict(sprite=I.divider(460), gap=10),
            dict(text=fmt_date(c["date"]), **s["date"]),
            dict(text=c["time"], **s["time"]) if c["time"] else dict(space=10),
            dict(text="VENUE", **s["label"]),
        ] + venue_lines(self.cfg, c["venue"], s["vlabel"], s["vline"])
        self.items, self.text_end = self.build(lines, self.text_y, 0.7, step=0.22)
        self.extra_setup()

    def extra_setup(self):
        pass

    def content(self, f, t):
        self.icon(f, t)
        self.band(f, t)
        self.draw_texts(f, self.items, t)

    def icon(self, f, t):
        pass

    def band(self, f, t):
        pass


class ATilak(ACeremony):
    name, key, theme, duration, transition = "A4_tilak", "tilak", "maroon", 8.0, "wipe"

    def extra_setup(self):
        self.thali = I.tilak_thali(330)
        self.pal = I.palace(1080, 760, "ivory")
        self.cl = I.marigold_cluster(380)
        self.cr = I.marigold_cluster(380, mirror=True)

    def icon(self, f, t):
        p = ease_out(prog(t, 0.3, 1.0))
        y = 470 - 10 * math.sin(t * 1.5)
        place(f, self.thali, W / 2, y, anchor=(0.5, 1.0), alpha=p, scale=0.9 + 0.1 * p)
        k = 330 / 520
        draw_flame(f, W / 2 + (258 - 260) * k, y - (300 - 221) * k * (0.9 + 0.1 * p), t, scale=0.32)

    def band(self, f, t):
        p = ease_out(prog(t, 0.2, 1.4))
        place(f, self.pal, W / 2, 1856 + 40 * (1 - p), anchor=(0.5, 1.0), scale=0.62, alpha=p * 0.9)
        place(f, self.cl, 64, 1860, anchor=(0.0, 1.0), alpha=p, rot=1.5 * math.sin(t * 0.8))
        place(f, self.cr, 1016, 1860, anchor=(1.0, 1.0), alpha=p, rot=-1.5 * math.sin(t * 0.8))


class AYagyopaveet(ACeremony):
    name, key, theme, duration, transition = "A5_yagyopaveet", "yagyopaveet", "saffron", 6.0, "flash"
    petal_kind = "marigold"

    def extra_setup(self):
        self.jan = I.janeu(250)
        self.om = I.om_symbol(120, "#d35400")
        self.kund = I.havan_kund(380)
        self.kal = I.kalash(170)
        self.embers = Sparkles(40, seed=7, region=(380, 1450, 700, 1760), color="#ffb347", rise=(60, 140))

    def icon(self, f, t):
        p = ease_out(prog(t, 0.2, 1.0))
        place(f, self.jan, W / 2, 345, rot=t * 12, alpha=p, scale=0.8 + 0.2 * p)
        place(f, self.om, W / 2, 345, alpha=p)

    def band(self, f, t):
        p = ease_out(prog(t, 0.2, 1.2))
        by = 1850
        place(f, self.kund, W / 2, by + 30 * (1 - p), anchor=(0.5, 1.0), alpha=p)
        for i, (dx, s) in enumerate(((-55, 0.8), (55, 0.8), (0, 1.15))):
            draw_flame(f, W / 2 + dx, by - 165 * 380 / 420 + 30 * (1 - p), t, scale=s, seed=i, alpha=p)
        self.embers.draw(f, t, alpha=p)
        for x in (210, 870):
            place(f, self.kal, x, by, anchor=(0.5, 1.0), alpha=p)


class AHaldi(ACeremony):
    name, key, theme, duration, transition = "A6_haldi", "haldi", "haldi", 6.0, "slide"
    petal_kind = "marigold"
    petal_n = 30

    def extra_setup(self):
        self.bowls = I.haldi_bowls(600)
        self.cl = I.marigold_cluster(400, seed=12)
        self.cr = I.marigold_cluster(400, seed=13, mirror=True)
        s, ctx = I.cairo_canvas(230, 230)
        I.draw_marigold(ctx, 115, 115, 105, base="yellow")
        self.flower = I.finish(s)

    def icon(self, f, t):
        p = ease_out_back(prog(t, 0.2, 1.0))
        place(f, self.flower, W / 2, 350, rot=t * 15, scale=max(0.01, p * 0.9))

    def band(self, f, t):
        p = ease_out(prog(t, 0.2, 1.2))
        place(f, self.bowls, W / 2, 1850 + 40 * (1 - p), anchor=(0.5, 1.0), alpha=p)
        place(f, self.cl, 64, 1860, anchor=(0.0, 1.0), alpha=p, rot=2 * math.sin(t))
        place(f, self.cr, 1016, 1860, anchor=(1.0, 1.0), alpha=p, rot=-2 * math.sin(t))


class ABaraat(ACeremony):
    name, key, theme, duration, transition = "A7_baraat", "baraat", "pink", 7.0, "zoom"
    petal_kind = "pink"

    def extra_setup(self):
        self.dhol = I.dhol(220)
        self.horse = I.horse(470)
        self.pal = I.palace(1080, 760, "ivory")
        self.kal = I.kalash(150)

    def icon(self, f, t):
        beat = abs(math.sin(t * math.pi * 2)) ** 3
        place(f, self.dhol, W / 2, 360 - 18 * beat, rot=4 * math.sin(t * math.pi * 2), alpha=prog(t, 0.2, 0.6))

    def band(self, f, t):
        p = ease_out(prog(t, 0.2, 1.2))
        place(f, self.pal, W / 2, 1856, anchor=(0.5, 1.0), scale=0.6, alpha=0.55 * p)
        festoon(f, t, 100, 980, 1445, 70, n=18)
        x = 330 + 300 * ease_in_out(prog(t, 0.4, self.duration))
        step = abs(math.sin(t * 4.0))
        place(f, self.horse, x, 1846 - 8 * step, anchor=(0.5, 1.0), alpha=p, rot=1.2 * math.sin(t * 4.0))
        place(f, self.kal, 150, 1846, anchor=(0.5, 1.0), alpha=p)


class AVivaah(ACard):
    """Mandap, sacred fire and the couple – large at first, then settling to the bottom for details."""
    name, duration, transition, theme = "A8_vivaah", 11.0, "iris", "red"
    petal_kind = "rose"
    petal_n = 28
    bell_rings = (0.3, 5.0)

    def content_setup(self):
        c = next(x for x in self.cfg["ceremonies"] if x["key"] == "vivaah")
        s = self.styles()
        self.title, ty = self.build([dict(text=c["name_hi"], **s["title_hi"], gap=0),
                                     dict(text=c["name_en"], **s["title_en"])], 480, 0.4, step=0.4)
        lines = [dict(sprite=I.divider(460), gap=10),
                 dict(text=fmt_date(c["date"]), **s["date"]),
                 dict(text=c["time"], **s["time"]),
                 dict(text="VENUE", **s["label"])] + venue_lines(self.cfg, c["venue"], s["vlabel"], s["vline"])
        self.items, self.text_end = self.build(lines, ty + 6, 5.2, step=0.22)
        self.mandap = I.mandap(900)
        self.kund = I.havan_kund(230)
        self.groom = I.groom(190)
        self.bride = I.bride(218)
        self.embers = Sparkles(30, seed=8, region=(420, 1200, 660, 1700), color="#ffb347", rise=(50, 120))

    def group(self, f, t, cx, by, s, a):
        """Draw mandap + couple + fire with the group's bottom-centre at (cx, by), scale s."""
        place(f, self.mandap, cx, by, anchor=(0.5, 1.0), scale=s, alpha=a)
        floor = by - 80 * s
        place(f, self.groom, cx - 215 * s, floor, anchor=(0.5, 1.0), scale=s, alpha=a)
        place(f, self.bride, cx + 215 * s, floor, anchor=(0.5, 1.0), scale=s, alpha=a)
        place(f, self.kund, cx, floor + 20 * s, anchor=(0.5, 1.0), scale=s, alpha=a)
        fy = floor + 20 * s - 165 * (230 / 420) * s
        for i, (dx, fs) in enumerate(((-28, 0.55), (28, 0.55), (0, 0.8))):
            draw_flame(f, cx + dx * s, fy, t, scale=fs * s, seed=i, alpha=a)

    def content(self, f, t):
        self.draw_texts(f, self.title, t)
        m = ease_in_out(prog(t, 4.4, 1.3))
        s = 1.0 - 0.5 * m
        self.group(f, t, W / 2, 1830, s, ease_out(prog(t, 0.2, 1.2)))
        self.embers.draw(f, t, alpha=1 - m)
        self.draw_texts(f, self.items, t)


class AGauna(ACeremony):
    name, key, theme, duration, transition = "A9_gauna", "gauna", "peach", 5.0, "slide"
    petal_kind = "pink"

    def extra_setup(self):
        self.doli = I.doli(470)
        self.lotus = I.lotus_flower(130)
        self.cl = I.marigold_cluster(330, seed=21)
        self.cr = I.marigold_cluster(330, seed=22, mirror=True)

    def icon(self, f, t):
        for i, dx in enumerate((-60, 60)):
            place(f, self.lotus, W / 2 + dx, 470, anchor=(0.5, 1.0), rot=(-8 if i == 0 else 8) + 3 * math.sin(t),
                  alpha=prog(t, 0.2 + i * 0.2, 0.6))

    def band(self, f, t):
        p = ease_out(prog(t, 0.2, 1.2))
        place(f, self.doli, W / 2, 1810 + 6 * math.sin(t * 3), anchor=(0.5, 1.0), rot=1.5 * math.sin(t * 1.5),
              alpha=p)
        place(f, self.cl, 64, 1860, anchor=(0.0, 1.0), alpha=p)
        place(f, self.cr, 1016, 1860, anchor=(1.0, 1.0), alpha=p)


class AClosing(ACard):
    name, duration, transition, theme = "A10_closing", 15.0, "flash", "red"
    petal_kind = "mixed"
    petal_n = 30
    bell_rings = (0.3, 11.0)

    def content_setup(self):
        tx, ad = self.cfg["texts"], self.cfg["addresses"]
        d = self.dark
        self.kal = I.kalash(170)
        l1 = [
            dict(text=tx["families_hi"], font="deva_bold", size=72, color=d, shine=True, gap=4),
            dict(sprite=I.divider(460), gap=16),
            dict(text=tx["awaiting_hi"][0], font="deva_semi", size=46, color=A_BODY, gap=0),
            dict(text=tx["awaiting_hi"][1], font="deva_semi", size=46, color=A_BODY, gap=40),
            dict(text=tx["compliments_en"], font="cormorant_it", size=50, color="#8a6212", gap=0),
            dict(text=tx["families_en"], font="script", size=78, gradient=DARK_GOLD),
        ]
        self.p1, _ = self.build(l1, 520, 0.6, step=0.5, end=6.4)
        vl = dict(font="cormorant_bold", size=46, color=d, gap=2)
        ln = dict(font="cormorant_semi", size=40, color=A_BODY, gap=0)
        lh = dict(font="deva", size=38, color=A_BODY, gap=0)
        l2 = [dict(text="निवास स्थान", font="deva_bold", size=60, color=d, gap=0),
              dict(sprite=I.divider(400), gap=10),
              dict(text=ad["groom_home"]["label_en"], **vl)] + \
             [dict(text=x, **ln) for x in ad["groom_home"]["lines_en"]] + \
             [dict(space=22), dict(text=ad["bride_home"]["label_en"], **vl)] + \
             [dict(text=x, **lh) for x in ad["bride_home"]["lines_hi"]]
        self.p2, _ = self.build(l2, 500, 7.0, step=0.18, end=11.6)
        self.p3, _ = self.build([dict(text="शुभ विवाह", font="deva_bold", size=84, color=d, shine=True, gap=0),
                                 dict(text=f"{self.cfg['groom']['name_en'].split()[-1]} & "
                                           f"{self.cfg['bride']['name_en'].split()[-1]}",
                                      font="script", size=96, gradient=DARK_GOLD, gap=10),
                                 dict(sprite=I.divider(460))], 600, 12.0, step=0.5, style="zoom")
        self.groom = I.groom(250)
        self.bride = I.bride(286)
        self.flowers = []
        rng = np.random.default_rng(4)
        # floral frame: marigolds/roses blooming along the card border, starting at the bottom corners
        pts = [(64 + 22, y) for y in np.linspace(1840, 560, 14)] + [(1016 - 22, y) for y in np.linspace(1840, 560, 14)]
        for i, (x, y) in enumerate(pts):
            s, ctx = I.cairo_canvas(90, 90)
            if i % 3 == 0:
                I.draw_rose(ctx, 45, 45, 30)
            else:
                I.draw_marigold(ctx, 45, 45, 38, base=("marigold", "yellow", "red")[i % 3])
            self.flowers.append((I.finish(s), x, y, 11.0 + (i % 14) * 0.12 + rng.uniform(0, 0.1)))
        self.spark = Sparkles(40, seed=41, region=(100, 500, 980, 1850))

    def content(self, f, t):
        p = ease_out_back(prog(t, 0.3, 1.0))
        place(f, self.kal, W / 2, 470, anchor=(0.5, 1.0), scale=max(p, 0.01), alpha=min(1, 2 * p))
        self.draw_texts(f, self.p1, t)
        self.draw_texts(f, self.p2, t)
        self.draw_texts(f, self.p3, t)
        a = ease_out(prog(t, 0.6, 1.2))
        y = 1842
        place(f, self.groom, 418, y, anchor=(0.5, 1.0), alpha=a)
        place(f, self.bride, 666, y, anchor=(0.5, 1.0), alpha=a)
        for spr, x, yy, t0 in self.flowers:
            q = ease_out_back(prog(t, t0, 0.6))
            if q > 0:
                place(f, spr, x, yy, scale=max(0.01, q), rot=t * 20, alpha=min(1, q * 2))
        self.spark.draw(f, t, alpha=prog(t, 11.5, 1.0))
        if t > self.duration - 1.2:  # graceful fade-out at the very end
            f *= 1 - 0.85 * ease_in_out(prog(t, self.duration - 1.2, 1.2))


def video_a(cfg):
    cls = [AGanesha, AAnnounce, ACouple, ATilak, AYagyopaveet, AHaldi, ABaraat, AVivaah, AGauna, AClosing]
    return [c(cfg) for c in cls]


# ============================================================================ VIDEO B – LUXURY BOLLYWOOD
B_IVORY = "#fff6e6"
B_CHAMP = "#f1dcb0"


class BLux(Scene):
    """Dark burgundy cinematic base: top light rays, bokeh, gold dust, rose petals, gold frame."""
    petals_n = 18
    show_frame = True

    def setup(self):
        self.bg = vertical_gradient(W, H, [(0, "#1a0208"), (0.35, "#4a0818"), (0.6, "#5c0a1f"), (1, "#14020a")])
        self.rays = I.rays(2400, 22, "#ffe9c2", 0.22)
        self.bokeh = Bokeh(14, seed=seed_of(self.name) + 997)
        self.dust = Sparkles(80, seed=seed_of(self.name) + 991, color="#ffe9c2", size=(0.2, 0.7))
        self.petals = PetalField("rose", n=self.petals_n, seed=seed_of(self.name) + 983, speed=(55, 100),
                                 size=(0.5, 0.95))
        self.frame = I.lux_frame()
        self.content_setup()

    def content_setup(self):
        pass

    def content(self, f, t):
        pass

    def render(self, t):
        f = self.bg.copy()
        place(f, self.rays, W / 2, -250, rot=8 * math.sin(t * 0.25), alpha=0.75, add=True)
        self.bokeh.draw(f, t)
        self.content(f, t)
        self.dust.draw(f, t, alpha=0.8)
        self.petals.draw(f, t)
        if self.show_frame:
            over(f, self.frame)
        vignette(f, 0.6)
        return f

    @staticmethod
    def styles():
        return dict(
            title_hi=dict(font="deva_bold", size=92, gradient=CHAMPAGNE, glow=("#c9a46a", 0.35, 10), shine=True,
                          gap=4),
            title_en=dict(font="cinzel", size=40, color=B_CHAMP, tracking=0.14, gap=8),
            date=dict(font="playfair_it", size=50, color=B_IVORY, gap=4),
            time=dict(font="cinzel_bold", size=48, gradient=CHAMPAGNE, gap=18),
            label=dict(font="cinzel", size=28, color="#c9a46a", tracking=0.3, gap=8),
            vlabel=dict(font="playfair", size=46, gradient=CHAMPAGNE, gap=4),
            vline=dict(font="cormorant_semi", size=44, color=B_IVORY, gap=0),
        )


class BOpening(Scene):
    name, duration = "B1_opening", 7.0

    def setup(self):
        self.bg = vertical_gradient(W, H, [(0, "#1a0208"), (0.5, "#5c0a1f"), (1, "#14020a")])
        self.light = radial_mask(W, H, W / 2, 760, 0, 1100)[..., None] * np.array(hex_rgb("#ffcf8a"), np.float32)
        self.rays = I.rays(1900, 36, "#fff1c1", 0.32)
        self.gan = gold_map(I.ganesha(660), dark="#4a2a05", mid="#d4a640", light="#fff4cf")
        self.glow = blur(tint(self.gan, "#ffcf6a"), 30)
        self.med = I.medallion(1000)
        self.dl, self.dr = I.door_panel(540, 1920, "left"), I.door_panel(540, 1920, "right")
        self.dust = Sparkles(90, seed=3, color="#fff1c1")
        self.bokeh = Bokeh(12, seed=4)
        sh = self.cfg["texts"]["shloka"]
        st = dict(font="tiro_sa", size=56, gradient=CHAMPAGNE, glow=("#c9a46a", 0.3, 8), style="wipe", dur=1.4)
        self.items, _ = self.build([dict(text=sh[0], **st, gap=14), dict(text=sh[1], **st)], 1300, 2.5,
                                   step=0.9, maxw=960)
        self.frame = I.lux_frame()

    def render(self, t):
        f = self.bg.copy()
        o = ease_in_out(prog(t, 0.5, 2.4))  # door opening
        f += self.light * (0.25 + 0.5 * o)
        place(f, self.rays, W / 2, 740, rot=t * 6, alpha=0.4 + 0.6 * o, add=True)
        self.bokeh.draw(f, t, alpha=o)
        place(f, self.med, W / 2, 740, rot=-t * 3, alpha=o * 0.9)
        p = ease_out(prog(t, 1.0, 2.4))
        place(f, self.glow, W / 2, 740, scale=0.9 + 0.1 * p, alpha=p * (0.6 + 0.2 * math.sin(t * 2)), add=True)
        place(f, self.gan, W / 2, 740, scale=0.9 + 0.1 * p, alpha=p)
        self.draw_texts(f, self.items, t)
        self.dust.draw(f, t, alpha=o)
        over(f, self.frame, alpha=prog(t, 2.5, 1.0))
        if o < 1:
            ang = o * 88
            c = math.cos(math.radians(ang))
            sh = 1 - 0.6 * o
            for spr, x, anc in ((self.dl, 0, (0.0, 0.5)), (self.dr, W, (1.0, 0.5))):
                d = spr.copy()
                d[..., :3] *= sh  # darken as the door turns away from the light
                place(f, d, x, H / 2, anchor=anc, sx=max(c, 0.01))
            if o < 0.05:  # light leaking through the gap before opening
                f[:, W // 2 - 3:W // 2 + 3] += np.array([1.0, 0.85, 0.5], np.float32) * (0.6 + 0.4 * math.sin(t * 9))
        vignette(f, 0.55)
        return f


class BAnnounce(BLux):
    """Dolly forward through a corridor of golden palace arches."""
    name, duration, transition = "B2_announce", 7.0, "flash"
    show_frame = True

    def content_setup(self):
        tx = self.cfg["texts"]
        self.arch = I.grand_arch(1080, 1920, color="#e9d3a3", lw=5)
        champ = dict(gradient=CHAMPAGNE, glow=("#c9a46a", 0.35, 10))
        lines = [
            dict(text=tx["ganesh"], font="deva_bold", size=70, **champ, shine=True, gap=6),
            dict(sprite=I.divider(460, color="#e9d3a3"), gap=36),
            dict(text=tx["blessing"], font="deva_semi", size=52, color=B_IVORY, gap=2),
            dict(text="WITH LOVE, BLESSINGS & GOOD WISHES", font="cinzel", size=28, color="#c9a46a", tracking=0.12),
            dict(space=56),
            dict(text=tx["invite"], font="deva_bold", size=68, **champ, shine=True, gap=4),
            dict(text="You and your family are cordially invited", font="playfair_it", size=40, color=B_CHAMP),
        ]
        self.items, _ = self.build(lines, 760, 0.9, step=0.5, style="zoom")

    def content(self, f, t):
        for i in range(4):
            d = (i / 4 + t * 0.07) % 1.0
            s = 0.32 * (4.2 ** (d * 1.6))
            a = min(1, d * 5) * (1 - prog(d, 0.75, 0.25))
            place(f, self.arch, W / 2, 1000, scale=s, anchor=(0.5, 0.52), alpha=a * 0.85)
        # soft dark plate behind the text for legibility
        self.draw_texts(f, self.items, t)


class BCouple(BLux):
    name, duration, transition = "B3_couple", 13.0, "zoom"
    petals_n = 26

    def content_setup(self):
        g, b = self.cfg["groom"], self.cfg["bride"]
        name = dict(font="deva_bold", size=74, gradient=CHAMPAGNE, glow=("#c9a46a", 0.35, 10), shine=True, gap=0)
        en = dict(font="script", size=84, gradient=CHAMPAGNE, gap=6)
        par = dict(font="deva_semi", size=42, color=B_IVORY, gap=0)
        lab = dict(font="cinzel", size=28, color="#c9a46a", tracking=0.3, gap=8)

        def block(p, start, end):
            return self.build([
                dict(text="THE GROOM" if p is g else "THE BRIDE", **lab),
                dict(text=f"{p['title_hi']} {p['name_hi']}", **name),
                dict(text=f"{p['title_en']} {p['name_en']}", **en),
                dict(text=f"{p['relation_hi']}: {p['father_hi']} एवं", **par),
                dict(text=p["mother_hi"], **par),
            dict(text=f"{p['grand_relation_hi']}: {p['grandfather_hi']}", **par),
            ], 200, start, step=0.4, end=end)[0]

        self.gb = block(g, 0.6, 5.6)
        self.bb = block(b, 6.2, 10.0)
        self.both, _ = self.build([
            dict(text="॥ शुभ विवाह ॥", font="deva_bold", size=64, gradient=CHAMPAGNE, glow=("#c9a46a", 0.3, 8),
                 gap=0),
            dict(text=f"{g['name_en']}", font="script", size=82, gradient=CHAMPAGNE, gap=0),
            dict(text="&", font="script", size=80, color="#e7b7a3", gap=0),
            dict(text=f"{b['name_en']}", font="script", size=82, gradient=CHAMPAGNE),
        ], 220, 10.6, step=0.35, style="zoom")
        self.groom = I.groom(360)
        self.bride = I.bride(410)
        self.spot = (radial_mask(W, H, W / 2, 1400, 0, 700)[..., None] *
                     np.array(hex_rgb("#ffcf8a"), np.float32)) * 0.35
        self.bl, self.br = I.rose_bouquet(420), I.rose_bouquet(420, seed=9, mirror=True)
        self.arch = I.grand_arch(1080, 1920, color="#e9d3a3", lw=4)

    def content(self, f, t):
        f += self.spot * (0.7 + 0.3 * math.sin(t * 0.8))
        place(f, self.arch, W / 2, H, anchor=(0.5, 1.0), scale=0.98, alpha=0.5)
        self.draw_texts(f, self.gb, t)
        self.draw_texts(f, self.bb, t)
        self.draw_texts(f, self.both, t)
        pg = ease_out(prog(t, 0.2, 1.6))
        mv = ease_in_out(prog(t, 5.6, 1.4))
        gx = W / 2 - 230 * mv
        place(f, self.groom, gx, 1880, anchor=(0.5, 1.0), alpha=pg, scale=1 - 0.08 * mv)
        pb = ease_out(prog(t, 6.0, 1.6))
        place(f, self.bride, W + 300 - (W + 300 - (W / 2 + 200)) * pb, 1880, anchor=(0.5, 1.0),
              scale=0.92, alpha=pb)
        place(f, self.bl, 0, H + 20, anchor=(0.0, 1.0), rot=1.5 * math.sin(t * 0.7), alpha=prog(t, 0, 1))
        place(f, self.br, W, H + 20, anchor=(1.0, 1.0), rot=-1.5 * math.sin(t * 0.7), alpha=prog(t, 0, 1))


class BCeremony(BLux):
    key = "tilak"
    index = 0
    icon_y = 470
    text_y = 790
    gold_icon = True

    def content_setup(self):
        c = next(x for x in self.cfg["ceremonies"] if x["key"] == self.key)
        self.c = c
        s = self.styles()
        self.chap, _ = self.build([dict(text=f"— {ROMAN[self.index]} —", font="cinzel", size=34, color="#c9a46a",
                                        tracking=0.2)], 118, 0.2, style="fade")
        lines = [
            dict(text=c["name_hi"], **s["title_hi"]),
            dict(text=c["name_en"].upper(), **s["title_en"]),
            dict(sprite=I.divider(460, color="#e9d3a3"), gap=14),
            dict(text=fmt_date(c["date"]), **s["date"]),
            dict(text=c["time"], **s["time"]) if c["time"] else dict(space=14),
            dict(text="VENUE", **s["label"]),
        ] + venue_lines(self.cfg, c["venue"], s["vlabel"], s["vline"])
        self.items, self.text_end = self.build(lines, self.text_y, 0.7, step=0.22, style="zoom")
        self.med = I.medallion(560)
        self.bl, self.br = I.rose_bouquet(330, seed=self.index + 1), I.rose_bouquet(330, seed=self.index + 7,
                                                                                     mirror=True)
        self.extra_setup()

    def extra_setup(self):
        pass

    def goldify(self, spr):
        return gold_map(spr, dark="#4a2a05", mid="#d4a640", light="#fff4cf") if self.gold_icon else spr

    def icon(self, f, t):
        pass

    def content(self, f, t):
        p = ease_out(prog(t, 0.1, 1.2))
        place(f, self.med, W / 2, self.icon_y, rot=t * 4, alpha=p, scale=0.9 + 0.1 * p)
        self.icon(f, t)
        self.draw_texts(f, self.chap, t)
        self.draw_texts(f, self.items, t)
        place(f, self.bl, 0, H + 30, anchor=(0.0, 1.0), rot=1.2 * math.sin(t * 0.8), alpha=p)
        place(f, self.br, W, H + 30, anchor=(1.0, 1.0), rot=-1.2 * math.sin(t * 0.8), alpha=p)


def _glow_of(spr, color="#ffcf6a", sigma=18):
    return blur(tint(spr, color), sigma)


class BTilak(BCeremony):
    name, key, index, duration, transition = "B4_tilak", "tilak", 0, 8.0, "iris"

    def extra_setup(self):
        self.th = self.goldify(I.tilak_thali(400))
        self.gl = _glow_of(self.th)

    def icon(self, f, t):
        p = ease_out(prog(t, 0.3, 1.0))
        y = self.icon_y + 60
        place(f, self.gl, W / 2, y, anchor=(0.5, 0.5), alpha=0.5 * p, add=True)
        place(f, self.th, W / 2, y, anchor=(0.5, 0.5), alpha=p, scale=0.9 + 0.1 * p)
        k = 400 / 520
        draw_flame(f, W / 2 + (258 - 260) * k, y + (221 - 150) * k, t, scale=0.38, alpha=p)


class BYagyopaveet(BCeremony):
    name, key, index, duration, transition = "B5_yagyopaveet", "yagyopaveet", 1, 6.0, "slide"

    def extra_setup(self):
        self.kund = self.goldify(I.havan_kund(330))
        self.om = text_sprite("ॐ", "deva_bold", 130, gradient=CHAMPAGNE, glow=("#ffcf6a", 0.5, 12))
        self.jan = self.goldify(I.janeu(300))
        self.embers = Sparkles(30, seed=17, region=(400, 250, 680, 560), color="#ffb347", rise=(60, 140))

    def icon(self, f, t):
        p = ease_out(prog(t, 0.2, 1.0))
        place(f, self.jan, W / 2, self.icon_y - 70, rot=t * 10, alpha=p * 0.9)
        place(f, self.om, W / 2, self.icon_y - 70, alpha=p)
        by = self.icon_y + 230
        place(f, self.kund, W / 2, by, anchor=(0.5, 1.0), alpha=p)
        for i, (dx, s) in enumerate(((-45, 0.65), (45, 0.65), (0, 0.95))):
            draw_flame(f, W / 2 + dx, by - 165 * 330 / 420, t, scale=s, seed=i, alpha=p)
        self.embers.draw(f, t, alpha=p)


class BHaldi(BCeremony):
    name, key, index, duration, transition = "B6_haldi", "haldi", 2, 6.0, "flash"
    gold_icon = False

    def extra_setup(self):
        self.bowls = I.haldi_bowls(520)
        self.gl = _glow_of(self.bowls, "#ffd34d", 24)
        self.mpet = PetalField("marigold", n=14, seed=61, speed=(70, 120), size=(0.5, 0.9))

    def icon(self, f, t):
        p = ease_out(prog(t, 0.2, 1.0))
        place(f, self.gl, W / 2, self.icon_y + 20, alpha=0.6 * p, add=True)
        place(f, self.bowls, W / 2, self.icon_y + 20 + 20 * (1 - p), alpha=p)
        self.mpet.draw(f, t)


class BBaraat(BCeremony):
    name, key, index, duration, transition = "B7_baraat", "baraat", 3, 7.0, "wipe"

    def extra_setup(self):
        self.horse = self.goldify(I.horse(470))
        self.gl = _glow_of(self.horse)
        self.fw = Fireworks(seed=5, region=(140, 160, 940, 520), every=0.9)

    def icon(self, f, t):
        p = ease_out(prog(t, 0.2, 1.0))
        self.fw.draw(f, t)
        festoon(f, t, 70, 1010, 190, 90, n=20)
        step = abs(math.sin(t * 4.0))
        x = W / 2 - 40 + 80 * ease_in_out(prog(t, 0.2, self.duration))
        place(f, self.gl, x, self.icon_y + 230, anchor=(0.5, 1.0), alpha=0.45 * p, add=True)
        place(f, self.horse, x, self.icon_y + 230 - 8 * step, anchor=(0.5, 1.0), alpha=p,
              rot=1.2 * math.sin(t * 4.0))


class BVivaah(BLux):
    name, duration, transition = "B8_vivaah", 11.0, "iris"
    petals_n = 28

    def content_setup(self):
        c = next(x for x in self.cfg["ceremonies"] if x["key"] == "vivaah")
        s = self.styles()
        self.chap, _ = self.build([dict(text="— V —", font="cinzel", size=34, color="#c9a46a", tracking=0.2)],
                                  118, 0.2, style="fade")
        self.title, ty = self.build([dict(text=c["name_hi"], **s["title_hi"]),
                                     dict(text=c["name_en"].upper(), **s["title_en"])], 200, 0.4, step=0.4)
        lines = [dict(sprite=I.divider(460, color="#e9d3a3"), gap=14),
                 dict(text=fmt_date(c["date"]), **s["date"]),
                 dict(text=c["time"], **s["time"]),
                 dict(text="VENUE", **s["label"])] + venue_lines(self.cfg, c["venue"], s["vlabel"], s["vline"])
        self.items, self.text_end = self.build(lines, ty + 10, 5.3, step=0.22, style="zoom")
        self.mandap = gold_map(I.mandap(920), dark="#4a2a05", mid="#d4a640", light="#fff4cf")
        self.mgl = _glow_of(self.mandap, "#ffcf6a", 26)
        self.kund = gold_map(I.havan_kund(230), dark="#4a2a05", mid="#d4a640", light="#fff4cf")
        self.groom = I.groom(196)
        self.bride = I.bride(224)
        self.embers = Sparkles(30, seed=88, region=(420, 1100, 660, 1700), color="#ffb347", rise=(50, 120))

    def content(self, f, t):
        self.draw_texts(f, self.chap, t)
        self.draw_texts(f, self.title, t)
        m = ease_in_out(prog(t, 4.5, 1.3))
        s = 1.0 - 0.48 * m
        a = ease_out(prog(t, 0.2, 1.2))
        cx, by = W / 2, 1850
        place(f, self.mgl, cx, by, anchor=(0.5, 1.0), scale=s, alpha=0.35 * a, add=True)
        place(f, self.mandap, cx, by, anchor=(0.5, 1.0), scale=s, alpha=a)
        floor = by - 82 * s
        place(f, self.groom, cx - 220 * s, floor, anchor=(0.5, 1.0), scale=s, alpha=a)
        place(f, self.bride, cx + 220 * s, floor, anchor=(0.5, 1.0), scale=s, alpha=a)
        place(f, self.kund, cx, floor + 20 * s, anchor=(0.5, 1.0), scale=s, alpha=a)
        fy = floor + 20 * s - 165 * (230 / 420) * s
        for i, (dx, fs) in enumerate(((-28, 0.55), (28, 0.55), (0, 0.8))):
            draw_flame(f, cx + dx * s, fy, t, scale=fs * s, seed=i, alpha=a)
        self.embers.draw(f, t, alpha=1 - m)
        self.draw_texts(f, self.items, t)


class BGauna(BCeremony):
    name, key, index, duration, transition = "B9_gauna", "gauna", 5, 5.0, "slide"

    def extra_setup(self):
        self.doli = self.goldify(I.doli(460))
        self.gl = _glow_of(self.doli)

    def icon(self, f, t):
        p = ease_out(prog(t, 0.2, 1.0))
        y = self.icon_y + 190 + 6 * math.sin(t * 3)
        place(f, self.gl, W / 2, y, anchor=(0.5, 1.0), alpha=0.45 * p, add=True)
        place(f, self.doli, W / 2, y, anchor=(0.5, 1.0), alpha=p, rot=1.5 * math.sin(t * 1.5))


class BClosing(BLux):
    name, duration, transition = "B10_closing", 15.0, "flash"
    petals_n = 30

    def content_setup(self):
        tx, ad = self.cfg["texts"], self.cfg["addresses"]
        champ = dict(gradient=CHAMPAGNE, glow=("#c9a46a", 0.35, 10))
        l1 = [
            dict(text="WITH BEST COMPLIMENTS", font="cinzel", size=34, color="#c9a46a", tracking=0.22, gap=18),
            dict(text=tx["families_hi"], font="deva_bold", size=74, **champ, shine=True, gap=8),
            dict(sprite=I.divider(460, color="#e9d3a3"), gap=22),
            dict(text=tx["awaiting_hi"][0], font="deva_semi", size=46, color=B_IVORY, gap=0),
            dict(text=tx["awaiting_hi"][1], font="deva_semi", size=46, color=B_IVORY, gap=46),
            dict(text=tx["compliments_en"], font="playfair_it", size=46, color=B_CHAMP, gap=0),
            dict(text=tx["families_en"], font="script", size=86, gradient=CHAMPAGNE),
        ]
        self.p1, _ = self.build(l1, 330, 0.6, step=0.45, end=6.4, style="zoom")
        vl = dict(font="playfair", size=44, gradient=CHAMPAGNE, gap=4)
        ln = dict(font="cormorant_semi", size=42, color=B_IVORY, gap=0)
        l2 = [dict(text="RESIDENCES", font="cinzel", size=34, color="#c9a46a", tracking=0.3, gap=10),
              dict(sprite=I.divider(400, color="#e9d3a3"), gap=16),
              dict(text=ad["groom_home"]["label_en"], **vl)] + \
             [dict(text=x, **ln) for x in ad["groom_home"]["lines_en"]] + \
             [dict(space=36), dict(text=ad["bride_home"]["label_en"], **vl)] + \
             [dict(text=x, **ln) for x in ad["bride_home"]["lines_en"]]
        self.p2, _ = self.build(l2, 330, 7.0, step=0.18, end=11.6)
        g, b = self.cfg["groom"], self.cfg["bride"]
        self.p3, _ = self.build([
            dict(text="॥ शुभ विवाह ॥", font="deva_bold", size=66, **champ, gap=0),
            dict(text=f"{g['name_en'].split()[-1]} & {b['name_en'].split()[-1]}", font="script", size=110,
                 gradient=CHAMPAGNE, gap=4),
            dict(text="22 – 30 NOVEMBER 2026", font="cinzel", size=34, color="#c9a46a", tracking=0.2),
        ], 300, 12.0, step=0.4, style="zoom")
        self.groom = I.groom(300)
        self.bride = I.bride(340)
        rng = np.random.default_rng(12)
        self.roses = []
        pal = [("#5c0a1f", "#a4133c", "#ff4d6d"), ("#b76e59", "#e7b7a3", "#fff0ea"), ("#7a0c24", "#c2185b", "#f48fb1")]
        pts = [(x, 60) for x in np.linspace(80, 1000, 9)] + [(x, H - 60) for x in np.linspace(80, 1000, 9)] + \
              [(60, y) for y in np.linspace(200, H - 200, 10)] + [(W - 60, y) for y in np.linspace(200, H - 200, 10)]
        for i, (x, y) in enumerate(pts):
            s, ctx = I.cairo_canvas(120, 120)
            if i % 4 == 3:
                I.draw_leaf(ctx, 60, 60, 55, 22, rng.uniform(0, 6.28), color="#2f5d3a")
            I.draw_rose(ctx, 60, 60, rng.uniform(30, 44), *pal[i % 3])
            self.roses.append((I.finish(s), x, y, 11.0 + rng.uniform(0, 1.6)))
        self.bl, self.br = I.rose_bouquet(420), I.rose_bouquet(420, seed=9, mirror=True)
        self.spot = (radial_mask(W, H, W / 2, 1450, 0, 700)[..., None] *
                     np.array(hex_rgb("#ffcf8a"), np.float32)) * 0.3

    def content(self, f, t):
        f += self.spot
        self.draw_texts(f, self.p1, t)
        self.draw_texts(f, self.p2, t)
        self.draw_texts(f, self.p3, t)
        a = ease_out(prog(t, 11.2, 1.4))
        place(f, self.groom, W / 2 - 150, 1880 + 80 * (1 - a), anchor=(0.5, 1.0), alpha=a)
        place(f, self.bride, W / 2 + 150, 1880 + 80 * (1 - a), anchor=(0.5, 1.0), alpha=a)
        place(f, self.bl, 0, H + 30, anchor=(0.0, 1.0), rot=1.2 * math.sin(t * 0.8), alpha=1 - a)
        place(f, self.br, W, H + 30, anchor=(1.0, 1.0), rot=-1.2 * math.sin(t * 0.8), alpha=1 - a)
        for spr, x, y, t0 in self.roses:
            q = ease_out_back(prog(t, t0, 0.7))
            if q > 0:
                place(f, spr, x, y, scale=max(0.01, q), rot=t * 10, alpha=min(1, 2 * q))

    def render(self, t):
        f = super().render(t)
        if t > self.duration - 1.4:
            f *= 1 - ease_in_out(prog(t, self.duration - 1.4, 1.4))
        return f


def video_b(cfg):
    cls = [BOpening, BAnnounce, BCouple, BTilak, BYagyopaveet, BHaldi, BBaraat, BVivaah, BGauna, BClosing]
    return [c(cfg) for c in cls]


from scenes_c import video_c  # noqa: E402,F401  (Video C lives in its own module)
