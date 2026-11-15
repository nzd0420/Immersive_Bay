#!/usr/bin/env python
"""Cut the 30-second vertical demo video.

    python make_demo.py --src demo/src --text demo/text --out demo/live-splat-demo.mp4

Reads the source clips with ffmpeg, lays every frame out at 1080x1920, adds the caption
sprites made by make_text.py, and writes H.264. Landscape footage (the headset recording)
is shown as a 3:4 cut-out over a blurred copy of itself; the captions sit above and below.

rec_*.mp4 are stretches of the headset recording: scan from 2:36.5, later from 4:54.8 and
compare from 6:11.8 (each starting one second early), live from 3:18. The music comes from make_music.py.
"""
import argparse
import math
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

W, H, FPS = 1080, 1920, 30
TEAM = "945ddc20d1da3aec4872cc00674fe91a.mp4"
PUSH = "scan-195510-orbit-pushin-portrait-15s.mp4"
LEARN = "learning-smooth.mp4"

# name, file, start in the source (s), length taken (s), length in the video (s), layout.
# Every cut falls on a bar line of the music (120 beats a minute: a bar is two seconds),
# except the fade through black at 3 s.
SEGMENTS = [
    ("open", TEAM, 0.0, 0.0, 6.0, "portrait"),       # cut by opening(): the phone clip into the fly-in
    ("scan", "rec_scan.mp4", 1.0, 8.0, 4.0, "landscape"),
    ("live", "rec_live.mp4", 0.0, 8.0, 4.0, "canvas"),   # laid out by training(): two stages, one above the other
    ("compare", "rec_compare.mp4", 1.0, 8.4, 6.0, "landscape"),
    ("outdoor", LEARN, 0.0, 6.5, 6.0, "portrait"),   # cut by outdoor(), around the third-person shot
    ("end", PUSH, 10.0, 5.0, 3.5, "portrait"),
]
LEFT, LABEL, HEAD = 64, 92, 176
# sprite, left edge, centre y, the stretch it belongs to, from and to in seconds after the
# stretch starts (no end: until the stretch ends). The headline of the compare stretch is
# not here: it follows the switches found in the footage.
CAPTIONS = [
    ("a_label", LEFT, LABEL, "open", 0.2, 2.7), ("title", 58, 232, "open", 0.3, 2.7),
    ("b_label", LEFT, LABEL, "open", 3.3, None), ("b_head", 60, HEAD, "open", 3.4, None),
    ("c1_label", LEFT, LABEL, "scan", 0.05, None), ("c1_head", 60, HEAD, "scan", 0.15, None),
    ("d_label", LEFT, 915, "live", 0.05, None), ("d_head", 60, 1000, "live", 0.15, None), ("live", 900, 915, "live", 0.9, None),
    ("f_label", LEFT, LABEL, "compare", 0.05, None),
    ("e_label", LEFT, LABEL, "outdoor", 0.05, None), ("e_head", 60, HEAD, "outdoor", 0.15, None),
    ("g_name", 58, 640, "end", 0.2, None), ("g_people", LEFT, 800, "end", 0.55, None),
    ("g_claim1", LEFT, 930, "end", 0.9, None), ("g_claim2", LEFT, 990, "end", 0.9, None),
]


def reader(path: Path, start: float, length: float, speed: float, size=None):
    """Frames of a stretch of a clip, sped up, at 30 a second."""
    scale = f",scale={size[0]}:{size[1]}:flags=lanczos" if size else ""
    frames = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="rgb24", input_params=["-ss", f"{start}", "-t", f"{length}"],
        output_params=["-vf", f"setpts=PTS/{speed},fps={FPS}{scale}", "-an"])
    meta = next(frames)
    width, height = meta["size"]
    for data in frames:
        yield np.frombuffer(data, np.uint8).reshape(height, width, 3)


def steady(frames: list, sigma: float = 45.0, enlarge: float = 1.12) -> list:
    """Take the head's shakes and slow turning out of a stretch of the recording.

    The picture's drift from frame to frame is measured from tracked corners (not on the
    info panel, which is fixed to the head), the path is smoothed over a few seconds, and
    every frame is moved onto the smooth path, enlarged slightly to hide the edges. The
    picture is cut from the middle of a wider frame, so sideways there is room to spare."""
    small = [cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
    mask = np.full((360, 640), 255, np.uint8)
    mask[230:, 200:480] = 0
    moves = np.zeros((len(frames), 3))
    for i in range(1, len(frames)):
        if small[i].mean() < 25 or small[i - 1].mean() < 25:
            continue  # passing through black: nothing to track
        corners = cv2.goodFeaturesToTrack(small[i - 1], 400, 0.01, 10, mask=mask)
        if corners is None or len(corners) < 12:
            continue
        moved, found, _ = cv2.calcOpticalFlowPyrLK(small[i - 1], small[i], corners, None)
        keep = found.ravel() == 1
        if keep.sum() < 12:
            continue
        matrix, _ = cv2.estimateAffinePartial2D(corners[keep], moved[keep])
        if matrix is not None:
            moves[i] = matrix[0, 2], matrix[1, 2], math.atan2(matrix[1, 0], matrix[0, 0])
    path = np.cumsum(moves, axis=0)
    reach = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(path, ((reach, reach), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, i], kernel, "valid") for i in range(3)], axis=1)
    fix = smooth - path
    print(f"steadied {len(frames)} frames: largest correction {np.abs(fix[:, 0]).max() * 3:.0f} px sideways, "
          f"{np.abs(fix[:, 1]).max() * 3:.0f} px up or down, {math.degrees(np.abs(fix[:, 2]).max()):.2f} degrees", flush=True)
    h, w = frames[0].shape[:2]
    out = []
    for frame, (dx, dy, da) in zip(frames, fix):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(da), enlarge)
        matrix[0, 2] += dx * 3
        matrix[1, 2] += dy * 3
        out.append(cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    return out


def between(a: np.ndarray, b: np.ndarray, part: float, flow_of: dict, key: int) -> np.ndarray:
    """The picture a fraction `part` of the way from frame a to frame b, following how
    things move between them (so slowed footage does not turn into double images)."""
    h, w = a.shape[:2]
    if key not in flow_of:
        small = (w // 2, h // 2)
        ga = cv2.cvtColor(cv2.resize(a, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ga, gb, None)
        flow_of[key] = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2
    flow = flow_of[key]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    from_a = cv2.remap(a, xs - part * flow[..., 0], ys - part * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    from_b = cv2.remap(b, xs + (1 - part) * flow[..., 0], ys + (1 - part) * flow[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(from_a, 1 - part, from_b, part, 0)


def compare(path: Path, start: float, length: float, count: int):
    """A turn through the room with the view switching as it goes (6:12.8 to 6:21.2 of the
    recording: coffee machine, fridge, door, cash machine; training finished).

    Played 1.4 times as fast, the frames in between made by optical flow, and only lightly
    steadied, so the turning stays. Returns the frames and, for each, what is on screen:
    the switch to the reconstruction passes through black, the one back only dims, so the
    two are told apart by how dark the picture gets."""
    frames = steady(list(reader(path, 0.0, start + length + 0.6, 1.0)), sigma=10.0, enlarge=1.06)
    light = np.array([f[:700:8, ::8].mean() for f in frames])
    switches = []  # (frame of the stretch, what comes after it)
    for i in range(15, len(light) - 15):
        around = np.median(np.r_[light[i - 15:i - 6], light[i + 7:i + 16]])
        if light[i] == light[i - 6:i + 7].min() and light[i] < around * 0.72 and around > 25:
            if not switches or i - switches[-1][0] > 9:
                switches.append((i, "splat" if light[i] / around < 0.2 else "real"))
    at = start * FPS + np.arange(count) * (length / (count / FPS))
    first = switches[0][1] if switches else "splat"
    state, shown = "real" if first == "splat" else "splat", []
    out, flow_of = [], {}
    for position in at:
        for frame_of, after in switches:
            if frame_of <= position:
                state = after
        shown.append(state)
        low = min(int(position), len(frames) - 2)
        part = position - low
        out.append(frames[low] if part < 0.02 else between(frames[low], frames[low + 1], part, flow_of, low))
    moments = [(round((f - start * FPS) / (length / (count / FPS)) / FPS, 2), s) for f, s in switches
               if start * FPS <= f < at[-1]]
    print("switches, seconds into the stretch:", moments, flush=True)
    return out, shown


def training(src: Path, count: int) -> list:
    """Two moments of the same training run, one above the other, with room for the caption
    between them. Below, its first seconds: the real room goes dark and points appear and
    gather into the room (3:18 to 3:26 of the recording). Above, about a minute and a half
    later, near step 9,000: looking around inside the reconstruction (4:55.8 to 5:03.8).
    Both at twice the speed."""
    early = list(reader(src / "rec_live.mp4", 0.0, 8.0, 2.0))
    later = list(reader(src / "rec_later.mp4", 1.0, 8.0, 2.0))
    tall = round(W * 9 / 16)
    edge = 110
    out = []
    for k in range(count):
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[edge:edge + tall] = cv2.resize(later[min(k, len(later) - 1)], (W, tall), interpolation=cv2.INTER_AREA)
        canvas[H - edge - tall:H - edge] = cv2.resize(early[min(k, len(early) - 1)], (W, tall), interpolation=cv2.INTER_AREA)
        out.append(canvas)
    return out


def opening(src: Path, count: int) -> list:
    """The phone clip of the headset going on, dissolving into the fly-in to the scanned
    corner: the two overlap for half a second around the three-second mark."""
    half = 8
    first = list(reader(src / TEAM, 0.8, 4.4, 1.2))
    second = list(reader(src / PUSH, 2.5, 6.5, 2.0, (W, H)))
    turn = count // 2
    out = []
    for k in range(count):
        a = portrait(first[min(k, len(first) - 1)])
        a = zoom(a, 1 + 0.04 * k / turn)
        if k < turn - half:
            out.append(a)
            continue
        b = second[min(max(0, k - (turn - half)), len(second) - 1)]
        if k >= turn + half:
            out.append(b)
        else:
            weight = (k - (turn - half) + 1) / (2 * half + 1)
            out.append(cv2.addWeighted(a, 1 - weight, b, weight, 0))
    return out


def outdoor(src: Path, count: int):
    """The outdoor scene forming, the person who is scanning it, the scene finished.

    The middle is 21.7 to 24.1 s of our rough cut exactly as it is there, with its own
    dissolves from the render to the third-person shot and back (third_person.mp4 holds
    21.0-24.6 s of it; the picture is the middle 484x858 of a wider frame, at 24 frames a
    second, so every fourth frame shows twice). Before and after it comes the render
    itself, at full resolution, joined where the pictures match: the frames of the render
    that the rough cut starts from and arrives at are found by comparing them.
    Returns the frames and, for each, the frame of the render it shows (or None)."""
    render = list(reader(src / LEARN, 0.0, 6.5, 1.0, (W, H)))
    raw = imageio_ffmpeg.read_frames(
        str(src / "third_person.mp4"), pix_fmt="rgb24", input_params=["-ss", "0.70", "-t", "2.40"],
        output_params=["-vf", f"crop=484:858:782:0,scale={W}:{H}:flags=lanczos,fps={FPS}", "-an"])
    next(raw)
    theirs = [np.frombuffer(data, np.uint8).reshape(H, W, 3) for data in raw]
    tiny = lambda f: cv2.resize(f, (54, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = np.stack([tiny(f) for f in render])
    differs = lambda f: np.abs(small - tiny(f)).mean((1, 2, 3))
    leave, arrive = differs(theirs[0]), differs(theirs[-1])
    a = int(leave.argmin())
    b = a + 1 + int(arrive[a + 1:].argmin())
    print(f"rough cut: {len(theirs)} frames; it leaves the render at frame {a} (difference {leave[a]:.1f}) "
          f"and comes back at frame {b} (difference {arrive[b]:.1f})", flush=True)
    before = (count - len(theirs)) * 48 // 108          # a little under half of what is left
    after = count - len(theirs) - before
    out, shows = [], []
    for k in range(before):
        i = round(k * a / max(1, before - 1))
        out.append(render[i]); shows.append(i)
    for j, frame in enumerate(theirs):
        out.append(frame); shows.append(None)
    for k in range(after):
        i = round(b + k * (len(render) - 1 - b) / max(1, after - 1))
        out.append(render[i]); shows.append(i)
    # Where the two meet the picture is the same, only softer in the rough cut: blend
    # across three frames so the change in sharpness does not show as a step.
    for j, weight in enumerate((0.25, 0.5, 0.75)):
        k = before - 3 + j
        out[k] = cv2.addWeighted(out[k], 1 - weight, theirs[0], weight, 0)
        k = before + len(theirs) + j
        out[k] = cv2.addWeighted(theirs[-1], 1 - weight, out[k], weight, 0)
    return out, shows


def portrait(frame: np.ndarray) -> np.ndarray:
    """Fill the frame, cutting off what does not fit."""
    h, w = frame.shape[:2]
    scale = max(W / w, H / h)
    if abs(scale - 1) > 1e-3:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    h, w = frame.shape[:2]
    return frame[(h - H) // 2:(h - H) // 2 + H, (w - W) // 2:(w - W) // 2 + W]


def landscape(frame: np.ndarray) -> np.ndarray:
    """The middle 3:4 of a 16:9 frame, over a blurred, darkened copy of the frame."""
    h, w = frame.shape[:2]
    cut = round(h * 3 / 4)
    middle = frame[:, (w - cut) // 2:(w - cut) // 2 + cut]
    middle = cv2.resize(middle, (W, round(W * 4 / 3)), interpolation=cv2.INTER_CUBIC)
    back = frame[:, (w - round(h * 9 / 16)) // 2:(w + round(h * 9 / 16)) // 2]
    back = cv2.resize(back, (W // 12, H // 12), interpolation=cv2.INTER_AREA)
    back = cv2.GaussianBlur(back, (0, 0), 4)
    back = (cv2.resize(back, (W, H), interpolation=cv2.INTER_CUBIC) * 0.22).astype(np.uint8)
    top = (H - middle.shape[0]) // 2
    back[top:top + middle.shape[0]] = middle
    return back


def zoom(frame: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.001:
        return frame
    matrix = cv2.getRotationMatrix2D((W / 2, H / 2), 0, factor)
    return cv2.warpAffine(frame, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def paste(canvas: np.ndarray, sprite: np.ndarray, left: float, cy: float, alpha: float = 1.0):
    """Blend an RGBA sprite onto the frame, its left edge at `left`, centred on `cy`."""
    if alpha <= 0:
        return
    h, w = sprite.shape[:2]
    x0, y0 = round(left), round(cy - h / 2)
    a, b, c, d = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if a >= c or b >= d:
        return
    part = sprite[b - y0:d - y0, a - x0:c - x0].astype(np.float32)
    weight = part[..., 3:] / 255.0 * alpha
    region = canvas[b:d, a:c].astype(np.float32)
    canvas[b:d, a:c] = (region * (1 - weight) + part[..., :3] * weight).astype(np.uint8)


def stamp(canvas: np.ndarray, sprite: np.ndarray, cx: float, cy: float, alpha: float = 1.0, scale: float = 1.0):
    """Blend an RGBA sprite onto the frame, centred on (cx, cy), at a size."""
    if abs(scale - 1) > 1e-3:
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    paste(canvas, sprite, cx - sprite.shape[1] / 2, cy, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src, text = Path(args.src), Path(args.text)
    sprites = {p.stem: cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
               for p in text.glob("*.png")}
    # The step counter's sprites, one for each frame of the outdoor render.
    counters = {int(name.split("_")[1]): sprites[name] for name in sprites if name.startswith("step_")}
    # Darker towards the top and bottom edges, so the captions read on any footage.
    ys = np.arange(H, dtype=np.float32)
    shade = 1 - 0.62 * np.clip(1 - ys / 420, 0, 1) ** 1.6 - 0.62 * np.clip((ys - 1540) / 380, 0, 1) ** 1.6
    shade = shade[:, None, None]

    out = imageio_ffmpeg.write_frames(
        args.out, (W, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=None,
        output_params=["-crf", "15", "-preset", "slow", "-movflags", "+faststart"], macro_block_size=2)
    out.send(None)
    n = 0  # frame of the video
    begins, total = {}, 0.0
    for name, _, _, _, shown, _ in SEGMENTS:
        begins[name] = (total, total + shown)
        total += shown
    captions = [(sprite, left, cy, begins[seg][0] + t0, begins[seg][0] + t1 if t1 is not None else begins[seg][1] - 0.1)
                for sprite, left, cy, seg, t0, t1 in CAPTIONS]
    for name, file, start, length, shown, layout in SEGMENTS:
        count = round(shown * FPS)
        on_screen = render_frame = None
        if name == "open":
            source = iter(opening(src, count))
        elif name == "live":
            source = iter(training(src, count))
        elif name == "outdoor":
            pictures, render_frame = outdoor(src, count)
            source = iter(pictures)
        elif name == "compare":
            pictures, on_screen = compare(src / file, start, length, count)
            source = iter(pictures)
        else:
            source = reader(src / file, start, length, length / shown, (W, H) if file in (PUSH, LEARN) else None)
        last = None
        for k in range(count):
            raw = next(source, None)
            if raw is None:
                raw = last
            last = raw
            frame = (raw if layout == "canvas" else portrait(raw) if layout == "portrait" else landscape(raw)).copy()
            t = n / FPS
            # Every cut settles from a slight enlargement; the opening and the ending fade.
            if name not in ("open", "end") and k < 10:
                frame = zoom(frame, 1 + 0.035 * (1 - k / 10) ** 2)
            if name == "open" and k < 6:
                frame = (frame * ((k + 1) / 7)).astype(np.uint8)
            if name == "end":
                frame = cv2.GaussianBlur(frame, (0, 0), 6)
                frame = (frame * 0.34).astype(np.uint8)
                if k < 8:
                    frame = (frame * ((k + 1) / 9)).astype(np.uint8)
                if k >= count - 15:
                    frame = (frame * ((count - 1 - k) / 15)).astype(np.uint8)
            if layout == "portrait" and name != "end":
                frame = (frame * shade).astype(np.uint8)
            if name == "outdoor" and render_frame[k] is not None:
                # The training step this frame of the render shows, with a thin bar that fills
                # as the run goes on. Both are hidden while the rough cut's footage plays.
                near = min(k + 1, 8, *(abs(k - j) for j in range(count) if render_frame[j] is None))
                appear = min(1.0, near / 8)
                paste(frame, counters[render_frame[k]], LEFT - 12, 276, appear)
                step = 15000 * min(render_frame[k] / 132, 1.0) ** 2.4
                y = 318
                bar = frame[y:y + 3, LEFT:LEFT + 560].astype(np.float32)
                bar = bar * (1 - 0.3 * appear) + 255 * 0.3 * appear
                filled = round(560 * step / 15000)
                bar[:, :filled] = bar[:, :filled] * (1 - appear) + 255 * appear
                frame[y:y + 3, LEFT:LEFT + 560] = bar.astype(np.uint8)
            if on_screen:
                # The headline says what is on screen and changes with every switch.
                since = next((j for j in range(1, k + 1) if on_screen[k - j] != on_screen[k]), k) / FPS
                paste(frame, sprites["f_real" if on_screen[k] == "real" else "f_splat"], 60, HEAD,
                      min(1.0, (since + 1 / FPS) / 0.1) * min(1.0, (count - k) / 4))
                # The controller's B button under the picture: it goes grey and dips each time
                # it is pressed, a moment before the view passes over.
                held = 0.0
                for j in range(1, count):
                    if on_screen[j] != on_screen[j - 1]:
                        d = k - j
                        held = max(held, 1.0 if -7 <= d <= -2 else (d + 9) / 2 if -9 < d < -7 else 1 - (d + 2) / 5 if -2 < d < 3 else 0.0)
                appear = min(1.0, (k + 1) / 8) * min(1.0, (count - k) / 5)
                stamp(frame, sprites["b_up"], W / 2, 1790, appear * (1 - held), 1 - 0.07 * held)
                stamp(frame, sprites["b_down"], W / 2, 1790, appear * held, 1 - 0.07 * held)
            for sprite, left, cy, t0, t1 in captions:
                if t0 <= t < t1:
                    age, remaining = t - t0, t1 - t
                    alpha = min(1.0, age / 0.22) * min(1.0, remaining / 0.18)
                    rise = (1 - min(1.0, age / 0.32)) ** 3 * 20  # eases up into place
                    paste(frame, sprites[sprite], left, cy + rise, alpha)
            out.send(np.ascontiguousarray(frame).tobytes())
            n += 1
        if hasattr(source, "close"):
            source.close()
        print(f"{name}: {count} frames, video now {n / FPS:.2f} s", flush=True)
    out.close()
    print(f"done: {n} frames, {n / FPS:.2f} s, {args.out}", flush=True)


if __name__ == "__main__":
    main()
