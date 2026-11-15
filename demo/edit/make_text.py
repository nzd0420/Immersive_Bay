"""Caption sprites for the demo video, as tight RGBA PNGs.

    python3 make_text.py      (on a Mac: uses the system's SF fonts)

One typeface family, white with a narrow black outline: a small tracked-out label in SF
Mono above a headline in SF Semibold. Also the B button shown while the view switches
(up and pressed), and the step counter of the outdoor render.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SANS, MONO, ROUND = "/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/SFNSRounded.ttf"
OUT = Path("text")
WHITE, SOFT, RED, BLACK = (255, 255, 255), (214, 222, 232), (255, 69, 58), (0, 0, 0)


def font(path: str, size: int, weight: str):
    f = ImageFont.truetype(path, size)
    f.set_variation_by_name(weight)
    return f


def sprite(parts, outline: int = 3, shadow: float = 0.35, bar: bool = False, dot: bool = False) -> Image.Image:
    """A line of text. `parts` are (text, font, colour, tracking in em), set one after another."""
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    height = max(f.size for _, f, _, _ in parts)
    lead = (int(height * 0.85) if dot else 0) + (int(height * 1.5) if bar else 0)
    width = lead + sum(probe.textlength(ch, font=f) + t * f.size for text, f, _, t in parts for ch in text)
    pad = 40
    image = Image.new("RGBA", (int(width) + 2 * pad, int(height * 1.5) + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    base = pad + int(height * 1.12)
    middle = base - int(height * 0.36)
    for layer in ("outline", "fill"):   # all the outlines first, so no letter's outline covers its neighbour
        x = pad
        if dot:
            r = int(height * 0.2)
            box = [x, middle - r, x + 2 * r, middle + r]
            draw.ellipse([box[0] - outline, box[1] - outline, box[2] + outline, box[3] + outline] if layer == "outline" else box,
                         fill=(BLACK if layer == "outline" else RED) + (255,))
        if bar:
            box = [x, middle - 1, x + int(height * 1.05), middle + 1]
            draw.rectangle([box[0] - outline, box[1] - outline, box[2] + outline, box[3] + outline] if layer == "outline" else box,
                           fill=(BLACK if layer == "outline" else parts[0][2]) + (255,))
        x += lead
        for text, f, colour, tracking in parts:
            for ch in text:
                if layer == "outline":
                    draw.text((x, base), ch, font=f, fill=BLACK + (255,), anchor="ls", stroke_width=outline, stroke_fill=BLACK + (255,))
                else:
                    draw.text((x, base), ch, font=f, fill=colour + (255,), anchor="ls")
                x += probe.textlength(ch, font=f) + tracking * f.size
    if shadow:
        alpha = image.getchannel("A")
        soft = alpha.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(min(255, v * 1.4) * shadow))
        under = Image.new("RGBA", image.size, (0, 0, 0, 0))
        under.putalpha(soft)
        image = Image.alpha_composite(under, image)
    a = np.asarray(image)[..., 3]
    rows, cols = np.where(a.max(1) > 6)[0], np.where(a.max(0) > 6)[0]
    return image.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def label(text, size=34, **extra):
    return sprite([(text, font(MONO, size, "Medium"), WHITE, 0.16)], outline=2, **extra)


def head(text, size=96, weight="Semibold"):
    return sprite([(text, font(SANS, size, weight), WHITE, -0.012)], outline=max(3, size // 30))


def line(text, size=44, weight="Regular"):
    return sprite([(text, font(SANS, size, weight), WHITE, 0.0)], outline=2)


def button(pressed: bool) -> Image.Image:
    """The controller's B button, in the manner of a system glyph: a disc with the letter cut
    dark into it; grey while it is held down."""
    size, scale = 220, 4
    big = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    c, r = size * scale // 2, 70 * scale
    draw.ellipse([c - r, c - r, c + r, c + r], fill=(142, 142, 147, 240) if pressed else (255, 255, 255, 245))
    letter = font(ROUND, 92 * scale, "Bold")
    draw.text((c, c + 2 * scale), "B", font=letter, fill=(44, 44, 46, 255) if pressed else (28, 28, 30, 255), anchor="mm")
    disc = big.resize((size, size), Image.LANCZOS)
    soft = disc.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.4))
    under = Image.new("RGBA", disc.size, (0, 0, 0, 0)); under.putalpha(soft)
    return Image.alpha_composite(under, disc)


SPRITES = {
    "title": head("Darkroom", 168, "Bold"),
    "a_label": label("IMMERSE THE BAY 2026", bar=True),
    "b_label": label("FULLY WIRELESS", bar=True),
    "b_head": head("No app. No cable."),
    "c1_label": label("01  SCAN", bar=True),
    "c1_head": head("Look around."),
    "d_label": label("02  TRAIN", bar=True),
    "d_head": head("Training, live."),
    "live": label("LIVE", 32, dot=True),
    "f_label": label("03  COMPARE", bar=True),
    "f_real": head("Reality"),
    "f_splat": head("Gaussian splat"),
    "e_label": label("ANY PLACE", bar=True),
    "e_head": head("Outdoors too."),
    "g_name": head("Darkroom", 168, "Bold"),
    "g_people": line("Yann Wu  &  Zidong Ning", 62, "Medium"),
    "g_claim1": line("All headset footage is a live screen recording,", 40),
    "g_claim2": line("rendered in real time. Not a replay.", 40),
    "b_up": button(False),
    "b_down": button(True),
}

OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()
for name, image in SPRITES.items():
    image.save(OUT / f"{name}.png")
# The step counter of the outdoor render: one sprite for each of its 195 frames, all the
# same size, so the number does not shift as its digits change.
mono, small = font(MONO, 52, "Medium"), font(MONO, 34, "Medium")
EDGE = dict(stroke_width=2, stroke_fill=BLACK + (255,))
for k in range(195):
    step = round(15000 * min(k / 132, 1.0) ** 2.4)
    cell = Image.new("RGBA", (660, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cell)
    draw.text((8, 72), "STEP", font=small, fill=WHITE + (255,), anchor="ls", **EDGE)
    draw.text((336, 72), f"{step:,}", font=mono, fill=WHITE + (255,), anchor="rs", **EDGE)
    draw.text((352, 72), "/ 15,000", font=small, fill=SOFT + (255,), anchor="ls", **EDGE)
    cell.save(OUT / f"step_{k:03d}.png")

sheet = Image.new("RGB", (1080, 2600), (96, 104, 112))
y = 30
for name, im in SPRITES.items():
    sheet.paste(im, (64, y), im); y += im.height + 16
im = Image.open(OUT / "step_090.png"); sheet.paste(im, (64, y), im); y += 110
sheet.crop((0, 0, 1080, y)).save("text_proof.jpg", quality=90)
print(len(list(OUT.glob("*.png"))), "sprites;", {k: v.size for k, v in SPRITES.items() if k in ("title", "g_people", "g_claim1", "g_claim2", "b_up")})
