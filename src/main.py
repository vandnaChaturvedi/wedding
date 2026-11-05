"""Build the three wedding invitation videos.

    python src/main.py                 # QA checks + render both videos + previews
    python src/main.py --only a        # just the traditional video
    python src/main.py --qa-only       # verify text/layout without rendering
    python src/main.py --only c --music song.mp3 --out name.mp4
                                       # use a song instead of the synthesised music; cuts land on its downbeats
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio  # noqa: E402
import scenes  # noqa: E402
from compositor import FPS, H, W  # noqa: E402
from typography import check_fonts  # noqa: E402
from video_export import (Timeline, build_scenes, compact_copy, contact_sheet, decode_check, extract_frame, probe,  # noqa: E402
                          probe_duration, render_video)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
PREV = OUT / "previews"

VIDEOS = {
    "a": ("video_a", "traditional_wedding_invitation.mp4", audio.compose_traditional),
    "b": ("video_b", "luxury_bollywood_wedding_invitation.mp4", audio.compose_luxury),
    "c": ("video_c", "pink_floral_wedding_invitation.mp4", audio.compose_floral),
}


def required_strings(cfg):
    g, b, ad = cfg["groom"], cfg["bride"], cfg["addresses"]
    req = [f"{g['title_hi']} {g['name_hi']}", f"{g['title_en']} {g['name_en']}", f"{b['title_hi']} {b['name_hi']}", f"{b['title_en']} {b['name_en']}",
           g["father_hi"], g["mother_hi"], b["father_hi"], b["mother_hi"], g["grandfather_hi"], b["grandfather_hi"], cfg["texts"]["families_hi"],
           cfg["texts"]["families_en"], cfg["texts"]["invite"], cfg["texts"]["blessing"], cfg["texts"]["ganesh"],
           *cfg["texts"]["shloka"], *cfg["texts"]["awaiting_hi"], cfg["texts"]["compliments_en"]]
    for c in cfg["ceremonies"]:
        req += [c["name_hi"], scenes.fmt_date(c["date"])]
        if c["time"]:
            req.append(c["time"])
    for k in ("groom_home", "bride_home", "tilak_venue"):
        req += ad[k]["lines_en"]
    return req


def qa_layout(tl, cfg, label):
    """Every text item inside the safe area; no two simultaneously-visible items overlap;
    every piece of wedding information is present verbatim."""
    problems = []
    seen = []
    for i, sc in enumerate(tl.scenes):
        sc.ensure()
        card = isinstance(sc, scenes.ACard)
        xmin, xmax = (90, W - 90) if card else (60, W - 60)
        items = [it for it in sc.texts]
        for it in items:
            x0, y0, x1, y1 = it.bbox()
            # sprites carry transparent padding for glow/shadow: measure the visible glyph box
            a = it.spr[..., 3] > 0.25
            ys, xs = a.nonzero()
            gx0, gx1 = x0 + xs.min(), x0 + xs.max()
            gy0, gy1 = y0 + ys.min(), y0 + ys.max()
            it.gbox = (gx0, gy0, gx1, gy1)
            if gx0 < xmin or gx1 > xmax or gy0 < 60 or gy1 > H - 60:
                problems.append(f"{label}/{sc.name}: text out of safe area {it.text!r} {tuple(map(int, it.gbox))}")
            if it.text:
                seen.append(it.text)
        for a_i in range(len(items)):
            for b_i in range(a_i + 1, len(items)):
                A, B = items[a_i], items[b_i]
                ea = A.end if A.end is not None else 1e9
                eb = B.end if B.end is not None else 1e9
                if A.start >= eb or B.start >= ea:  # never on screen together
                    continue
                ax0, ay0, ax1, ay1 = A.gbox
                bx0, by0, bx1, by1 = B.gbox
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    problems.append(f"{label}/{sc.name}: overlap {A.text!r} / {B.text!r}")
    joined = "\n".join(seen)
    for r in required_strings(cfg):
        if r not in joined:
            problems.append(f"{label}: missing required text {r!r}")
    return problems


def report_data(cfg):
    print("\n=== Wedding data used (verify) ===")
    g, b = cfg["groom"], cfg["bride"]
    print(f"Groom : {g['title_hi']} {g['name_hi']} / {g['title_en']} {g['name_en']}  |  {g['father_hi']} ({g['father_en']}), "
          f"{g['mother_hi']} ({g['mother_en']}), {g['grandfather_hi']} ({g['grandfather_en']})  |  {g['family_en']}")
    print(f"Bride : {b['title_hi']} {b['name_hi']} / {b['title_en']} {b['name_en']}  |  {b['father_hi']} ({b['father_en']}), "
          f"{b['mother_hi']} ({b['mother_en']}), {b['grandfather_hi']} ({b['grandfather_en']})  |  {b['family_en']}")
    for c in cfg["ceremonies"]:
        print(f"  {c['name_en']:<28} {scenes.fmt_date(c['date']):<28} {c['time'] or '(no time)':<9} -> "
              f"{cfg['addresses'][c['venue']]['label_en']}")
    for k, v in cfg["addresses"].items():
        print(f"  {k}: {' '.join(v['lines_en'])}")


def previews(tl, mp4, tag):
    PREV.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for i, sc in enumerate(tl.scenes):
        end = tl.starts[i] + sc.duration
        t = end - (tl.scenes[i + 1].td if i + 1 < len(tl.scenes) else 0) - 0.4
        if i == len(tl.scenes) - 1:
            t = tl.starts[i] + sc.duration - 2.0
        p = PREV / f"{tag}_scene{i + 1:02d}_{sc.name}.png"
        extract_frame(mp4, t, p)
        paths.append(p)
        labels.append(f"{i + 1}. {sc.name.split('_', 1)[1]}  @{t:.1f}s")
    contact_sheet(paths, labels, OUT / f"{tag}_contact_sheet.jpg", cols=5, thumb_w=360)
    extract_frame(mp4, tl.starts[2] + 9.0, OUT / f"{tag}_thumbnail.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["a", "b", "c"])
    ap.add_argument("--qa-only", action="store_true")
    ap.add_argument("--seconds", type=float, help="render only the first N seconds (testing)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--music", help="song file to use as the soundtrack (scenes are re-timed to its bars)")
    ap.add_argument("--out", help="output file name (with --only)")
    args = ap.parse_args()
    if (args.music or args.out) and not args.only:
        sys.exit("--music/--out need --only a|b|c")

    probs = check_fonts()
    if probs:
        sys.exit("Font/shaping problems:\n" + "\n".join(probs))
    cfg = scenes.load_config()
    report_data(cfg)
    OUT.mkdir(exist_ok=True)

    keys = [args.only] if args.only else ["a", "b", "c"]
    durations, song = None, None
    if args.music:
        import beatsync
        (OUT / "audio").mkdir(parents=True, exist_ok=True)
        song = beatsync.decode(args.music, OUT / "audio" / "_song.wav")
        grid = beatsync.analyse(song)
        durations = beatsync.plan_durations(build_scenes(VIDEOS[args.only][0]), grid)
        print(f"\n=== Music: {Path(args.music).name} ===\n{grid['bpm']:.1f} BPM, bar {grid['bar']:.3f}s, "
              f"music {grid['start']:.1f}–{grid['end']:.1f}s\nscene durations: "
              + ", ".join(f"{d:.2f}" for d in durations) + f"  (total {sum(durations):.2f}s)")
    all_probs = []
    for k in keys:
        builder, _, _ = VIDEOS[k]
        tl = Timeline(build_scenes(builder, durations))
        all_probs += qa_layout(tl, cfg, k.upper())
    print("\n=== Layout / text QA ===")
    print("\n".join(all_probs) if all_probs else "OK – all text present, inside safe area, no overlaps")
    if args.qa_only:
        return
    if all_probs:
        sys.exit("Fix QA problems before rendering.")

    for k in keys:
        builder, fname, composer = VIDEOS[k]
        fname = args.out or fname
        tl = Timeline(build_scenes(builder, durations))
        (OUT / "audio").mkdir(exist_ok=True)
        wav = OUT / "audio" / f"{Path(fname).stem}.wav"  # kept: the soundtrack on its own
        t0 = time.time()
        print(f"\n[{k.upper()}] composing music ({tl.total:.1f}s)…", flush=True)
        if song is not None:
            beatsync.prepare_audio(song, tl.total, str(wav))
            (OUT / "audio" / "_song.wav").unlink(missing_ok=True)
        else:
            composer(tl, str(wav))
        print(f"[{k.upper()}] rendering {tl.n_frames} frames…", flush=True)
        render_video(builder, OUT / fname, wav, workers=args.workers, max_seconds=args.seconds,
                     durations=durations)
        print(f"[{k.upper()}] done in {time.time() - t0:.0f}s", flush=True)
        if args.seconds:
            continue
        compact_copy(OUT / fname, OUT / f"{Path(fname).stem}_whatsapp_16MB.mp4")
        err = decode_check(OUT / fname)
        print(f"[{k.upper()}] decode check: {'clean' if not err else err[:400]}")
        info = [l.strip() for l in probe(OUT / fname).splitlines() if "Stream" in l or "Duration" in l]
        print("\n".join(info))
        print(f"duration {probe_duration(OUT / fname):.2f}s, size {(OUT / fname).stat().st_size / 1e6:.1f} MB")
        previews(tl, OUT / fname, Path(fname).stem)


if __name__ == "__main__":
    main()
