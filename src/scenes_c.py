"""Video C – Pink floral (white ground, pink peonies, hanging hearts, dotted Ganesha medallion,
jharokha arch window). Same ten-scene storyboard and timings as videos A and B, so the music
composer's scene indices still line up.
"""
from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np

import floral as F
import illustrations as I
from animations import PetalField, Sparkles, ease_in_out, ease_out, ease_out_back, prog, sway_sprite
from compositor import H, W, cairo_canvas, ken_burns, pil_to_sprite, place
from scenes import ROOT, Scene, fmt_date, seed_of
from typography import text_sprite

MAG = "#c2185b"
BODY = "#8e2453"
SOFT = "#ad1457"


def paste_rgba(dst, spr, x, y):
    """Premultiplied 'over' of spr onto an RGBA canvas at integer (x, y) (top-left)."""
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4]
    d += s


def date_block(c, width=860):
    """'Save the date' block: month on top, WEEKDAY | day | time row between rules, year below."""
    d = dt.date.fromisoformat(c["date"])
    month = text_sprite(d.strftime("%B").upper(), "mont", 34, color=MAG, tracking=0.18)
    wk = text_sprite(d.strftime("%A").upper(), "mont", 32, color=MAG, tracking=0.08)
    day = text_sprite(f"{d.day:02d}", "playfair", 92, color=MAG)
    right = text_sprite(f"AT {c['time']}" if c["time"] else str(d.year), "mont", 32, color=MAG, tracking=0.08)
    year = text_sprite(str(d.year), "mont", 34, color=MAG, tracking=0.18) if c["time"] else None
    rh = day.shape[0]
    hgt = month.shape[0] + rh + (year.shape[0] if year is not None else 0) + 12
    out = np.zeros((hgt, width, 4), np.float32)
    paste_rgba(out, month, (width - month.shape[1]) / 2, 0)
    y = month.shape[0] + 4
    cx = width / 2
    paste_rgba(out, day, cx - day.shape[1] / 2, y)
    side = (width - day.shape[1]) / 2 - 30
    col = np.array([*[int(MAG[i:i + 2], 16) / 255 for i in (1, 3, 5)], 1.0], np.float32)
    for k, spr in enumerate((wk, right)):
        sx = 0 if k == 0 else width - side
        paste_rgba(out, spr, sx + (side - spr.shape[1]) / 2, y + (rh - spr.shape[0]) / 2)
        for ry in (y + rh * 0.2, y + rh * 0.8):
            out[int(ry):int(ry) + 3, int(sx + 10):int(sx + side - 10)] = col * 0.85
    if year is not None:
        paste_rgba(out, year, (width - year.shape[1]) / 2, y + rh + 4)
    return out


@lru_cache(None)
def template_ganesha():
    """Line-art Ganesha taken from the user's pink template video (Untitled design.mp4), as requested;
    extracted once to assets/illustrations/ganesha_lineart_template.png."""
    from PIL import Image
    return pil_to_sprite(Image.open(ROOT / "assets" / "illustrations" / "ganesha_lineart_template.png"))


def date_range(cfg):
    """'22 – 30 NOVEMBER 2026' from the first and last ceremony dates."""
    ds = sorted(dt.date.fromisoformat(c["date"]) for c in cfg["ceremonies"])
    a, b = ds[0], ds[-1]
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day} – {b.day} {b.strftime('%B').upper()} {b.year}"
    return f"{a.day} {a.strftime('%B').upper()} – {b.day} {b.strftime('%B').upper()} {b.year}"


class HeartFloat:
    """Small hearts drifting upward with a gentle sway."""

    def __init__(self, n=14, seed=1, alpha=0.55):
        rng = np.random.default_rng(seed)
        self.n, self.alpha = n, alpha
        self.x = rng.uniform(40, W - 40, n)
        self.y0 = rng.uniform(0, H, n)
        self.v = rng.uniform(35, 70, n)
        self.ph = rng.uniform(0, math.tau, n)
        self.s = rng.uniform(0.5, 1.0, n)
        self.spr = []
        for col in ("#f06292", "#e91e63", "#f8bbd0"):
            s, ctx = cairo_canvas(44, 44)
            F.heart_path(ctx, 22, 4, 36)
            I.rgba(ctx, col)
            ctx.fill()
            self.spr.append(I.finish(s))
        self.c = rng.integers(0, 3, n)

    def draw(self, f, t, alpha=1.0):
        for i in range(self.n):
            y = (self.y0[i] - self.v[i] * t) % (H + 100) - 50
            x = self.x[i] + 26 * math.sin(t * 0.9 + self.ph[i])
            place(f, self.spr[self.c[i]], x, y, scale=self.s[i], rot=12 * math.sin(t * 1.3 + self.ph[i]),
                  alpha=alpha * self.alpha)


class CFloral(Scene):
    """White floral ground with drifting pink petals and hearts."""
    petals_n = 12
    hearts_n = 10

    def setup(self):
        self.bg = F.floral_pattern()
        self.petals = PetalField("pink", n=self.petals_n, seed=seed_of(self.name), speed=(60, 110),
                                 size=(0.45, 0.8))
        self.hearts = HeartFloat(self.hearts_n, seed=seed_of(self.name) + 7)
        self.content_setup()

    def content_setup(self):
        pass

    def content(self, f, t):
        pass

    def render(self, t):
        f = ken_burns(self.bg, 1.03 + 0.015 * math.sin(t * 0.3), dy=-5 * t)
        self.hearts.draw(f, t)
        self.petals.draw(f, t)  # behind the content so petals never cross the text
        self.content(f, t)
        return f

    # shared pieces
    def toran(self, f, t, t0=0.0):
        p = ease_out(prog(t, t0, 1.3))
        place(f, sway_sprite(F.heart_toran(), t, amp=6), W / 2, -360 * (1 - p), anchor=(0.5, 0))

    def corners(self, f, t, top=True, bottom=True, size=470, t0=0.0):
        p = ease_out(prog(t, t0, 1.2))
        k = 1 - p
        sw = 1.5 * math.sin(t * 0.8)
        if top:
            place(f, F.peony_corner(size), -120 * k, -120 * k, anchor=(0, 0), rot=sw, alpha=p)
            place(f, F.peony_corner(size, seed=11, flip_x=True), W + 120 * k, -120 * k, anchor=(1, 0), rot=-sw,
                  alpha=p)
        if bottom:
            place(f, F.peony_corner(size, seed=13, flip_y=True), -120 * k, H + 120 * k, anchor=(0, 1), rot=-sw,
                  alpha=p)
            place(f, F.peony_corner(size, seed=15, flip_x=True, flip_y=True), W + 120 * k, H + 120 * k,
                  anchor=(1, 1), rot=sw, alpha=p)

    def medallion(self, f, t, x, y, size, t0=0.0, inner=None, inner_scale=1.0):
        o, m, c = F.dot_medallion(size)
        p = ease_out_back(prog(t, t0, 1.1))
        if p <= 0:
            return
        s = max(0.01, p)
        a = min(1.0, p * 2)
        place(f, o, x, y, scale=s, rot=t * 8, alpha=a)
        place(f, m, x, y, scale=s, rot=-t * 12, alpha=a)
        place(f, c, x, y, scale=s, alpha=a)
        if inner is not None:
            q = ease_out(prog(t, t0 + 0.4, 0.9))
            place(f, inner, x, y + 2 * math.sin(t * 1.6), scale=inner_scale * (0.85 + 0.15 * q), alpha=q)

    @staticmethod
    def fit(spr, box):
        return box / max(spr.shape[:2])

    def styles(self):
        return dict(
            title_hi=dict(font="deva_bold", size=84, color=MAG, gap=0, shine=True),
            title_en=dict(font="playfair", size=62, color=MAG, gap=26),
            venue=dict(font="playfair", size=70, color=MAG, gap=6),
            vlabel=dict(font="mont_semi", size=38, color=MAG, gap=6),
            vline=dict(font="mont", size=36, color=BODY, gap=4),
        )


class CGanesha(CFloral):
    name, duration = "C1_ganesha", 7.0
    hearts_n = 0

    def content_setup(self):
        tx = self.cfg["texts"]
        self.gan = template_ganesha()
        self.border = F.peony_border()
        sh = tx["shloka"]
        st = dict(font="tiro_sa", size=50, color=BODY, style="wipe", dur=1.3)
        self.items, _ = self.build([dict(text=tx["ganesh"], font="deva_bold", size=62, color=MAG, gap=20,
                                         style="fade"),
                                    dict(text=sh[0], **st, gap=10), dict(text=sh[1], **st)], 1075, 2.0, step=0.9,
                                   maxw=940)

    def content(self, f, t):
        self.toran(f, t)
        self.medallion(f, t, W / 2, 720, 600, t0=0.5, inner=self.gan, inner_scale=self.fit(self.gan, 600 * 0.48))
        p = ease_out(prog(t, 1.0, 1.4))
        place(f, self.border, W / 2, H + 200 * (1 - p), anchor=(0.5, 1.0), alpha=p)
        self.draw_texts(f, self.items, t)
        if t < 1.0:  # open gently from white
            f[:] = f + (1 - f) * (1 - ease_in_out(t / 1.0))


class CAnnounce(CFloral):
    name, duration, transition = "C2_announce", 7.0, "fade"

    def content_setup(self):
        tx = self.cfg["texts"]
        self.gan = template_ganesha()
        it = dict(font="mont_it", size=32, color=SOFT)
        lines = [
            dict(text=tx["blessing"], font="deva_semi", size=50, color=BODY, gap=4),
            dict(text="With love, blessings and good wishes", **it),
            dict(space=50),
            dict(text=tx["invite"], font="deva_bold", size=64, color=MAG, gap=6, shine=True),
            dict(text="You and your family are cordially invited", **it, gap=24),
            dict(sprite=I.divider(420, color=MAG)),
        ]
        self.items, _ = self.build(lines, 880, 1.0, step=0.45)

    def content(self, f, t):
        self.corners(f, t)
        self.medallion(f, t, W / 2, 640, 360, t0=0.3, inner=self.gan, inner_scale=self.fit(self.gan, 360 * 0.48))
        self.draw_texts(f, self.items, t)


class CCouple(CFloral):
    name, duration, transition = "C3_couple", 13.0, "iris"

    def content_setup(self):
        g, b = self.cfg["groom"], self.cfg["bride"]
        self.frame, self.mask, _ = F.jharokha()
        self.inside = F.window_scene()
        self.groom = I.groom(190)
        self.bride = I.bride(212)
        self.twinkle = Sparkles(16, seed=3, region=(110, 170, 490, 450), color="#fffbe6")
        nm = dict(font="deva_bold", size=70, color=MAG, gap=0, shine=True)
        en = dict(font="playfair", size=52, color=MAG, gap=14)
        rel = dict(font="mont", size=28, color=SOFT, tracking=0.1, gap=4)
        par = dict(font="deva_semi", size=35, color=BODY, gap=2)

        def col(p, x, start):
            return self.build([
                dict(text=f"{p['title_hi']} {p['name_hi']}", **nm),
                dict(text=f"{p['title_en']} {p['name_en']}", **en),
                dict(text=p["relation_en"], **rel),
                dict(text=f"{p['relation_hi']}: {p['father_hi']} एवं", **par),
                dict(text=p["mother_hi"], **{**par, "gap": 10}),
                dict(text=f"{p['grand_relation_hi']}: {p['grandfather_hi']}", **par),
            ], 1150, start, step=0.3, x=x, maxw=450)[0]

        self.top, _ = self.build([dict(text="॥ शुभ विवाह ॥", font="deva_bold", size=50, color=MAG, gap=0),
                                  dict(text="WEDDING CEREMONY", font="mont", size=30, color=SOFT, tracking=0.2)],
                                 1000, 1.0, step=0.3)
        self.gcol = col(g, 290, 1.8)
        self.bcol = col(b, 790, 2.6)
        s, ctx = cairo_canvas(60, 60)
        F.heart_path(ctx, 30, 6, 48)
        I.rgba(ctx, MAG)
        ctx.fill()
        self.heart = I.finish(s)

    def content(self, f, t):
        p = ease_out_back(prog(t, 0.2, 1.1))
        inner = self.inside.copy()
        br = 1 + 0.006 * math.sin(t * 2)
        place(inner, self.groom, 248, 676, anchor=(0.5, 1.0), sy=br)
        place(inner, self.bride, 352, 676, anchor=(0.5, 1.0), sy=br)
        self.twinkle.draw(inner, t)
        win = np.dstack([inner * self.mask[..., None], self.mask])
        sc = max(0.01, 0.8 + 0.2 * p)
        a = min(1.0, max(p, 0) * 2)
        sc *= 1.14
        place(f, win, W / 2, 975, anchor=(0.5, 1.0), scale=sc, alpha=a)
        place(f, self.frame, W / 2, 975, anchor=(0.5, 1.0), scale=sc, alpha=a)
        self.corners(f, t, top=False, size=430, t0=0.4)
        self.draw_texts(f, self.top, t)
        self.draw_texts(f, self.gcol, t)
        self.draw_texts(f, self.bcol, t)
        q = ease_out_back(prog(t, 2.4, 0.8))
        place(f, self.heart, W / 2, 1196, scale=max(0.01, q) * (1 + 0.08 * math.sin(t * 4)), alpha=min(1, q))


class CCeremony(CFloral):
    """'Save the date' card: dotted medallion with the ceremony icon, name, date block, venue."""
    key = "tilak"
    duration = 6.0

    def icon_sprite(self):
        return I.tilak_thali(200)

    def content_setup(self):
        c = next(x for x in self.cfg["ceremonies"] if x["key"] == self.key)
        self.c = c
        s = self.styles()
        self.icon = self.icon_sprite()
        lines = [
            dict(text=c["name_hi"], **s["title_hi"]),
            dict(text=c["name_en"], **s["title_en"]),
            dict(sprite=date_block(c), text=f"{fmt_date(c['date'])} {c['time'] or ''}".strip(), gap=40,
                 style="pop"),
            dict(text="Venue", **s["venue"]),
        ]
        v = self.cfg["addresses"][c["venue"]]
        if c["venue"] == "tilak_venue":
            lines.append(dict(text=v["lines_en"][0], **s["vlabel"]))
            lines += [dict(text=l, **s["vline"]) for l in v["lines_en"][1:]]
        else:
            lines.append(dict(text=v["label_en"], **s["vlabel"]))
            lines += [dict(text=l, **s["vline"]) for l in v["lines_en"]]
        self.items, _ = self.build(lines, 730, 0.6, step=0.2)

    def content(self, f, t):
        self.corners(f, t, size=430)
        self.medallion(f, t, W / 2, 450, 440, t0=0.1, inner=self.icon, inner_scale=self.fit(self.icon, 220))
        self.draw_texts(f, self.items, t)
        self.extra(f, t)

    def extra(self, f, t):
        pass


class CTilak(CCeremony):
    name, key, duration, transition = "C4_tilak", "tilak", 8.0, "wipe"


class CYagyopaveet(CCeremony):
    name, key, duration, transition = "C5_yagyopaveet", "yagyopaveet", 6.0, "fade"

    def icon_sprite(self):
        s = I.janeu(200).copy()
        om = I.om_symbol(90, "#d35400")
        oy, ox = (s.shape[0] - om.shape[0]) // 2, (s.shape[1] - om.shape[1]) // 2
        paste_rgba(s, om, ox, oy)
        return s


class CHaldi(CCeremony):
    name, key, duration, transition = "C6_haldi", "haldi", 6.0, "slide"

    def icon_sprite(self):
        s, ctx = I.cairo_canvas(200, 200)
        I.draw_marigold(ctx, 100, 100, 92, base="yellow")
        return I.finish(s)


class CBaraat(CCeremony):
    name, key, duration, transition = "C7_baraat", "baraat", 7.0, "wipe"

    def icon_sprite(self):
        return I.horse(200)

    def extra(self, f, t):
        beat = abs(math.sin(t * math.pi * 2)) ** 3
        place(f, I.dhol(130), W / 2 + 265, 560 - 12 * beat, rot=4 * math.sin(t * math.pi * 2),
              alpha=prog(t, 0.6, 0.6))


class CVivaah(CCeremony):
    name, key, duration, transition = "C8_vivaah", "vivaah", 11.0, "iris"
    hearts_n = 18

    def icon_sprite(self):
        return I.mandap(220)

    def content_setup(self):
        super().content_setup()
        self.burst = HeartFloat(26, seed=88, alpha=0.8)

    def extra(self, f, t):
        self.burst.draw(f, t + 3, alpha=prog(t, 4.0, 1.5) * (1 - prog(t, 9.5, 1.0)))


class CGauna(CCeremony):
    name, key, duration, transition = "C9_gauna", "gauna", 5.0, "fade"

    def icon_sprite(self):
        return I.doli(220)


class CClosing(CFloral):
    name, duration, transition = "C10_closing", 15.0, "zoom"
    hearts_n = 14

    def content_setup(self):
        tx, ad = self.cfg["texts"], self.cfg["addresses"]
        self.gan = template_ganesha()
        self.border = F.peony_border()
        it = dict(font="mont_it", size=34, color=SOFT)
        l1 = [
            dict(text=tx["families_hi"], font="deva_bold", size=72, color=MAG, shine=True, gap=6),
            dict(sprite=I.divider(420, color=MAG), gap=18),
            dict(text=tx["awaiting_hi"][0], font="deva_semi", size=44, color=BODY, gap=0),
            dict(text=tx["awaiting_hi"][1], font="deva_semi", size=44, color=BODY, gap=40),
            dict(text=tx["compliments_en"], **it, gap=6),
            dict(text=tx["families_en"], font="playfair", size=64, color=MAG),
        ]
        self.p1, _ = self.build(l1, 560, 0.8, step=0.5, end=6.4)
        vl = dict(font="mont_semi", size=36, color=MAG, gap=4)
        ln = dict(font="mont", size=32, color=BODY, gap=2)
        lh = dict(font="deva", size=36, color=BODY, gap=0)
        l2 = [dict(text="निवास स्थान", font="deva_bold", size=62, color=MAG, gap=0),
              dict(sprite=I.divider(380, color=MAG), gap=12),
              dict(text=ad["groom_home"]["label_en"], **vl)] + \
             [dict(text=x, **ln) for x in ad["groom_home"]["lines_en"]] + \
             [dict(space=26), dict(text=ad["bride_home"]["label_en"], **vl)] + \
             [dict(text=x, **lh) for x in ad["bride_home"]["lines_hi"]]
        self.p2, _ = self.build(l2, 470, 7.0, step=0.16, end=11.6)
        g, b = self.cfg["groom"], self.cfg["bride"]
        self.p3, _ = self.build([dict(text="शुभ विवाह", font="deva_bold", size=80, color=MAG, shine=True, gap=4),
                                 dict(text=f"{g['name_en']} & {b['name_en']}", font="playfair", size=76,
                                      color=MAG, gap=10),
                                 dict(text=date_range(self.cfg), font="mont", size=30, color=SOFT,
                                      tracking=0.2)], 980, 12.6, step=0.4, style="zoom")

    def content(self, f, t):
        self.toran(f, t)
        self.draw_texts(f, self.p1, t)
        self.draw_texts(f, self.p2, t)
        self.medallion(f, t, W / 2, 680, 520, t0=12.0, inner=self.gan, inner_scale=self.fit(self.gan, 520 * 0.48))
        self.draw_texts(f, self.p3, t)
        p = ease_out(prog(t, 11.8, 1.4))
        place(f, self.border, W / 2, H + 220 * (1 - p), anchor=(0.5, 1.0), alpha=p)
        if t < 12.4:
            self.corners(f, t, top=False, size=380, t0=0.3)
        if t > self.duration - 1.2:
            f[:] = f + (1 - f) * 0.85 * ease_in_out(prog(t, self.duration - 1.2, 1.2))


def video_c(cfg):
    cls = [CGanesha, CAnnounce, CCouple, CTilak, CYagyopaveet, CHaldi, CBaraat, CVivaah, CGauna, CClosing]
    return [c(cfg) for c in cls]
