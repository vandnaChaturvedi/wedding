# Wedding Invitation Videos – Abhinandan & Sakshi

Three 85-second, 1080×1920 (9:16), 30 fps, H.264/AAC invitation videos are generated entirely in code:

| File | Look |
|---|---|
| `output/traditional_wedding_invitation.mp4` | Ivory arch card on maroon jaal pattern, marigold toran, swinging temple bells, diyas, flat colourful illustrations, shehnai + tabla + sitar music |
| `output/luxury_bollywood_wedding_invitation.mp4` | Dark burgundy cinematic: palace doors opening, gold-foil Ganesha, a corridor of golden arches, bokeh, light rays, rose petals, gold-embossed ceremony medallions, orchestral strings, choir, santoor, bansuri and timpani |
| `output/pink_floral_wedding_invitation.mp4` | White floral ground in the style of the supplied pink template: hanging-heart toran, dotted Ganesha medallion, jharokha arch window with the illustrated couple, "save the date" ceremony cards, pink peonies; light music: bansuri in Bhupali, santoor, soft strings, gentle tabla and chimes (no dhol) |
| `*_whatsapp_16MB.mp4` | Two-pass compressed copies under 16 MB for older WhatsApp limits |
| `output/*_contact_sheet.jpg`, `output/*_thumbnail.jpg`, `output/previews/` | One frame per scene, plus thumbnails |

Every illustration (Ganesha, couple, mandap, horse, doli, kalash, thali, havan kund, bells, diyas, flowers, palace) is
original vector art drawn with cairo in `src/illustrations.py`. All music is synthesised in `src/audio.py`, so the
project uses no stock assets. Fonts (including Montserrat) are SIL Open Font License (`assets/fonts/OFL.txt`).

## Regenerate

```bash
cd wedding_invitation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py              # QA + all three videos + previews (~5–10 min on 24 cores)
.venv/bin/python src/main.py --only a     # just the traditional video (b = luxury, c = pink floral)
.venv/bin/python src/main.py --qa-only    # check text/layout without rendering
```

## Change details

Edit **`config/wedding.json`**. It holds names, parents, ceremonies (date, time or `null`, venue key), addresses
and invitation texts. Weekdays are computed from the dates. Before rendering, `main.py` checks that every name,
date, time and address line appears verbatim on screen, inside the safe area, with no overlapping text.

## Code layout

```
src/main.py          orchestration + QA (data report, text presence, safe-area & overlap checks)
src/scenes.py        the 10 scenes of videos A and B (timings follow the storyboard: 7,7,13,8,6,6,7,11,5,15 s)
src/animations.py    easing, petals, gold dust, bokeh, fireworks, flames, bell swing, text reveals, transitions
src/scenes_c.py      the 10 scenes of video C (pink floral)
src/illustrations.py procedural vector artwork (cairo)
src/floral.py        video C artwork: peonies, heart toran, dotted medallion, jharokha window
src/typography.py    Devanagari-correct text (Pillow + libraqm), gold-foil gradients, glow, shadow
src/compositor.py    premultiplied-alpha compositing, Ken Burns, vignette
src/audio.py         music composition & synthesis, synchronised to scene transitions/text reveals
src/video_export.py  parallel chunk rendering → ffmpeg, final WhatsApp-friendly encode, contact sheets
assets/reference_frames/  frames extracted from the supplied reference video (style study only)
```
