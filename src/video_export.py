"""Timeline assembly, parallel frame rendering, H.264/AAC export, thumbnails and contact sheets."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from animations import transition
from compositor import FPS, H, W, to_uint8

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


class Timeline:
    """Scenes back to back; scene i+1 begins its transition `td` seconds before scene i ends.
    Scene durations sum to the total length."""

    def __init__(self, scenes):
        self.scenes = scenes
        self.starts = []
        t = 0.0
        for s in scenes:
            self.starts.append(t)
            t += s.duration
        self.total = t

    @property
    def n_frames(self):
        return int(round(self.total * FPS))

    def local(self, i, T):
        td_in = self.scenes[i].td if i > 0 else 0.0
        return T - (self.starts[i] - td_in)

    def frame(self, T):
        i = max(k for k, s in enumerate(self.starts) if s <= T + 1e-9)
        sc = self.scenes[i]
        sc.ensure()
        f = sc.render(self.local(i, T))
        if i + 1 < len(self.scenes):
            nxt = self.scenes[i + 1]
            t0 = self.starts[i + 1] - nxt.td
            if T >= t0:
                nxt.ensure()
                g = nxt.render(self.local(i + 1, T))
                f = transition(nxt.transition, f, g, (T - t0) / nxt.td)
        return f

    def events(self):
        """Musical cue points: transition midpoints + scene starts (for beat-synchronised audio)."""
        ev = []
        for i, s in enumerate(self.scenes):
            if i > 0:
                ev.append(("transition", self.starts[i] - s.td / 2, s.transition))
            for it in s.texts:
                ev.append(("text", self.starts[i] - (s.td if i else 0) + it.start, s.name))
        return sorted(ev, key=lambda e: e[1])

    def scene_at(self, T):
        return max(k for k, s in enumerate(self.starts) if s <= T + 1e-9)


# --------------------------------------------------------------------------- worker side
_TL = None


def build_scenes(builder_name, durations=None):
    """The scene list for a video; `durations` (seconds per scene) overrides the storyboard timings."""
    import scenes
    sc = getattr(scenes, builder_name)(scenes.load_config())
    if durations:
        for s, d in zip(sc, durations):
            s.duration = d
    return sc


_COVER_T = None


def _init_worker(builder_name, durations=None, cover_t=None):
    global _TL, _COVER_T
    cv2.setNumThreads(1)
    _TL = Timeline(build_scenes(builder_name, durations))
    _COVER_T = cover_t


def _render_chunk(args):
    idx, f0, f1, out_path = args
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
           "-pix_fmt", "yuv420p", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(f0, f1):
        # Frame 0 doubles as the poster: WhatsApp/galleries build the preview from the first frame,
        # and the real opening fades in from white, so show the chosen cover moment for 1/30 s instead.
        fr = _TL.frame(_COVER_T if (k == 0 and _COVER_T is not None) else k / FPS)
        p.stdin.write(to_uint8(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode:
        raise RuntimeError(f"ffmpeg chunk {idx} failed")
    return idx


def render_video(builder_name, out_mp4, audio_wav, workers=None, chunk_seconds=2.0, max_seconds=None,
                 durations=None, cover_t=None):
    tl = Timeline(build_scenes(builder_name, durations))
    n = tl.n_frames if max_seconds is None else int(max_seconds * FPS)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    tmp = Path(tempfile.mkdtemp(prefix="wedding_"))
    per = int(chunk_seconds * FPS)
    jobs = [(i, f0, min(f0 + per, n), str(tmp / f"chunk_{i:04d}.mp4")) for i, f0 in enumerate(range(0, n, per))]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(builder_name, durations, cover_t)) as ex:
        for done, _ in enumerate(ex.map(_render_chunk, jobs), 1):
            if done % 5 == 0 or done == len(jobs):
                print(f"  [{builder_name}] chunks {done}/{len(jobs)}", flush=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{j[3]}'\n" for j in jobs))
    joined = tmp / "joined.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(joined)], check=True)
    # Final WhatsApp-friendly encode: H.264 High@4.1, yuv420p, AAC, faststart.
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(joined), "-i", str(audio_wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-maxrate", "9M", "-bufsize", "18M", "-profile:v", "high", "-level", "4.1",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-ac", "2", "-shortest", "-movflags", "+faststart", str(out_mp4)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return tl


def compact_copy(src, dst, target_mb=15.5):
    """Extra small copy that fits WhatsApp's classic 16 MB media limit."""
    dur = probe_duration(src)
    kbps = int(target_mb * 8192 / dur) - 128
    tmp = Path(tempfile.mkdtemp())
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p",
              "-profile:v", "high", "-level", "4.1"]
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "1",
                    "-passlogfile", str(tmp / "p"), "-an", "-f", "mp4", os.devnull], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "2",
                    "-passlogfile", str(tmp / "p"), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                    str(dst)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def probe(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return r.stderr


def probe_duration(path):
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe(path))
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def extract_frame(path, t, out_png):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                    str(out_png)], check=True)


def decode_check(path):
    """Fully decode the file; returns ffmpeg error output ('' = clean)."""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    return r.stderr.strip()


def contact_sheet(images, labels, out_path, cols=5, thumb_w=360):
    from PIL import Image, ImageDraw, ImageFont
    from typography import FONT_DIR
    ims = [Image.open(p).convert("RGB") for p in images]
    tw = thumb_w
    th = int(tw * ims[0].height / ims[0].width)
    rows = math.ceil(len(ims) / cols)
    lab_h = 44
    sheet = Image.new("RGB", (cols * (tw + 16) + 16, rows * (th + lab_h + 16) + 16), (24, 6, 10))
    f = ImageFont.truetype(str(FONT_DIR / "Cinzel.ttf"), 20)
    d = ImageDraw.Draw(sheet)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(k, cols)
        x, y = 16 + c * (tw + 16), 16 + r * (th + lab_h + 16)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + tw / 2, y + th + 10), lab, fill=(233, 211, 163), font=f, anchor="ma")
    sheet.save(out_path, quality=90)
