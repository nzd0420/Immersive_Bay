"""Caption sprites for the demo video, as tight RGBA PNGs.

    python3 make_text.py      (on a Mac: uses the system's SF fonts)

One typeface family, white with one accent colour, a soft shadow and nothing else: a small
tracked-out label in SF Mono above a headline in SF Semibold, and a line of plain text
below the picture.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SANS, MONO = "/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/SFNSMono.ttf"
OUT = Path("text")
WHITE, SOFT, ACCENT, RED = (255, 255, 255), (214, 222, 232), (124, 249, 201), (255, 69, 58)


def font(path: str, size: int, weight: str):
    f = ImageFont.truetype(path, size)
    f.set_variation_by_name(weight)
    return f


def sprite(parts, shadow: float = 0.6, bar: bool = False, dot: bool = False) -> Image.Image:
    """A line of text. `parts` are (text, font, colour, tracking in em), set one after another."""
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    height = max(f.size for _, f, _, _ in parts)
    lead = (int(height * 0.85) if dot else 0) + (int(height * 1.5) if bar else 0)
    width = lead + sum(probe.textlength(ch, font=f) + t * f.size for text, f, _, t in parts for ch in text)
    pad = 40
    image = Image.new("RGBA", (int(width) + 2 * pad, int(height * 1.5) + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x, base = pad, pad + int(height * 1.12)
    middle = base - int(height * 0.36)
    if dot:
        r = int(height * 0.2)
        draw.ellipse([x, middle - r, x + 2 * r, middle + r], fill=RED + (255,))
    if bar:
        draw.rectangle([x, middle - 1, x + int(height * 1.05), middle + 1], fill=parts[0][2] + (255,))
    x += lead
    for text, f, colour, tracking in parts:
        for ch in text:
            draw.text((x, base), ch, font=f, fill=colour + (255,), anchor="ls")
            x += probe.textlength(ch, font=f) + tracking * f.size
    if shadow:
        alpha = image.getchannel("A")
        soft = alpha.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(min(255, v * 1.6) * shadow))
        under = Image.new("RGBA", image.size, (0, 0, 0, 0))
        under.putalpha(soft)
        image = Image.alpha_composite(under, image)
    a = np.asarray(image)[..., 3]
    rows, cols = np.where(a.max(1) > 6)[0], np.where(a.max(0) > 6)[0]
    return image.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def label(text, colour=ACCENT, size=34, **extra):
    return sprite([(text, font(MONO, size, "Medium"), colour, 0.16)], **extra)


def head(text, size=96, weight="Semibold"):
    return sprite([(text, font(SANS, size, weight), WHITE, -0.012)])


def line(text, size=46):
    return sprite([(text, font(SANS, size, "Regular"), SOFT, 0.0)])


SPRITES = {
    "title": head("Live Splat", 168, "Bold"),
    "title_end": head("Live Splat", 150, "Bold"),
    "a_label": label("IMMERSE THE BAY 2026", bar=True),
    "a_line": line("Scan a room. Watch it rebuild around you."),
    "b_label": label("FULLY WIRELESS", bar=True),
    "b_head": head("No app. No cable."),
    "b_line": line("One headset. One web page."),
    "c1_label": label("01  SCAN", bar=True),
    "c1_head": head("Look around."),
    "c1_line": line("Each green frame: one photo + head pose."),
    "c2_label": label("02  UPLOAD", bar=True),
    "c2_head": head("Over Wi-Fi."),
    "c2_line": line("No COLMAP. Poses come from the headset."),
    "d_label": label("03  TRAIN", bar=True),
    "d_head": head("Training, live."),
    "d_line": line("Real-time. Not a replay."),
    "live": label("LIVE", WHITE, 32, dot=True),
    "e_label": label("ANY PLACE", bar=True),
    "e_head": head("Outdoors too."),
    "e_line": line("From random points to a real place."),
    "e_tag": label("TIME-LAPSE OF ONE TRAINING RUN", SOFT, 24),
    "f_label": label("04  COMPARE", bar=True),
    "f_real": head("Reality"),
    "f_splat": head("Gaussian splat"),
    "f_line": line("Same place. Real scale. Press B to switch."),
    "g_1": label("REAL-TIME 3D GAUSSIAN SPLATTING", WHITE, 36),
    "g_2": label("FULLY WIRELESS", WHITE, 36),
    "g_3": label("NO APP  ·  NO COLMAP", WHITE, 36),
    "g_4": label("ONE WEB PAGE ON QUEST 3S", WHITE, 36),
    "g_foot": label("IMMERSE THE BAY 2026", ACCENT, 28, bar=True),
    "x2": label("2× SPEED", SOFT, 24),
    "x25": label("2.5× SPEED", SOFT, 24),
}

OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()
for name, image in SPRITES.items():
    image.save(OUT / f"{name}.png")
# The step counter of the outdoor clip: one sprite per frame of the video (180 of them), all
# the same size, so the number does not shift as its digits change.
mono, small = font(MONO, 52, "Medium"), font(MONO, 34, "Medium")
for k in range(180):
    step = round(15000 * min(k * (6.5 / 6.0) / 132, 1.0) ** 2.4)
    cell = Image.new("RGBA", (640, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cell)
    draw.text((6, 70), "STEP", font=small, fill=ACCENT + (255,), anchor="ls")
    number = f"{step:,}"
    draw.text((330, 70), number, font=mono, fill=WHITE + (255,), anchor="rs")
    draw.text((346, 70), "/ 15,000", font=small, fill=SOFT + (200,), anchor="ls")
    soft = cell.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(min(255, v * 1.5) * 0.6))
    under = Image.new("RGBA", cell.size, (0, 0, 0, 0)); under.putalpha(soft)
    Image.alpha_composite(under, cell).save(OUT / f"step_{k:03d}.png")

sheet = Image.new("RGB", (1080, 2400), (58, 62, 70))
y = 30
for name in SPRITES:
    im = SPRITES[name]
    sheet.paste(im, (64, y), im); y += im.height + 16
im = Image.open(OUT / "step_090.png"); sheet.paste(im, (64, y), im); y += 110
sheet.crop((0, 0, 1080, y)).save("text_proof.jpg", quality=90)
print(len(list(OUT.glob("*.png"))), "sprites;", {k: v.size for k, v in SPRITES.items() if k in ("title", "b_head", "f_splat", "c1_line", "c2_line", "g_1")})
