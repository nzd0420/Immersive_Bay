#!/usr/bin/env python
"""Cut the 30-second vertical demo video.

    python make_demo.py --src demo/src --text demo/text --out demo/live-splat-demo.mp4

Reads the source clips with ffmpeg, lays every frame out at 1080x1920, adds the caption
sprites made by make_text.py, and writes H.264. Landscape footage (the headset recording)
is shown as a 3:4 cut-out over a blurred copy of itself; the captions sit above and below.

rec_*.mp4 are stretches of the headset recording: scan from 2:36.5 (the scan itself starts
one second in), live from 3:18, compare from 6:08. The music comes from make_music.py.
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
    ("team", TEAM, 0.8, 3.6, 3.0, "portrait"),
    ("push", PUSH, 3.0, 6.0, 3.0, "portrait"),
    ("scan", "rec_scan.mp4", 1.0, 8.0, 4.0, "landscape"),
    ("live", "rec_live.mp4", 0.0, 8.0, 4.0, "landscape"),
    ("compare", "rec_compare.mp4", 1.0, 8.4, 6.0, "landscape"),
    ("outdoor", LEARN, 0.0, 6.5, 6.0, "portrait"),
    ("end", PUSH, 10.0, 5.0, 3.5, "portrait"),
]
LEFT, LABEL, HEAD, LINE, TAG = 64, 92, 176, 1766, 1846
# sprite, left edge, centre y, the stretch it belongs to, from and to in seconds after the
# stretch starts (no end: until the stretch ends). The headline of the compare stretch is
# not here: it follows the switches found in the footage.
CAPTIONS = [
    ("a_label", LEFT, LABEL, "team", 0.2, 2.8), ("title", 58, 232, "team", 0.3, 2.8), ("a_line", LEFT, LINE, "team", 0.8, 2.8),
    ("b_label", LEFT, LABEL, "push", 0.2, None), ("b_head", 60, HEAD, "push", 0.3, None), ("b_line", LEFT, LINE, "push", 0.7, None),
    ("c1_label", LEFT, LABEL, "scan", 0.05, None), ("c1_head", 60, HEAD, "scan", 0.15, None),
    ("c1_line", LEFT, LINE, "scan", 0.5, 2.1), ("c2_line", LEFT, LINE, "scan", 2.25, None), ("x2", LEFT, TAG, "scan", 0.1, None),
    ("d_label", LEFT, LABEL, "live", 0.05, None), ("d_head", 60, HEAD, "live", 0.15, None),
    ("d_line", LEFT, LINE, "live", 0.9, None), ("x2", LEFT, TAG, "live", 0.1, None), ("live", 900, LABEL, "live", 0.9, None),
    ("f_label", LEFT, LABEL, "compare", 0.05, None), ("f_line", LEFT, LINE, "compare", 0.4, None),
    ("x14", LEFT, TAG, "compare", 0.1, None),
    ("e_label", LEFT, LABEL, "outdoor", 0.05, None), ("e_head", 60, HEAD, "outdoor", 0.15, None),
    ("e_line", LEFT, LINE, "outdoor", 0.6, None), ("e_tag", LEFT, TAG, "outdoor", 0.6, None),
    ("title_end", 58, 620, "end", 0.2, None), ("g_1", LEFT, 820, "end", 0.5, None), ("g_2", LEFT, 905, "end", 0.75, None),
    ("g_3", LEFT, 990, "end", 1.0, None), ("g_4", LEFT, 1075, "end", 1.25, None), ("g_foot", LEFT, LINE, "end", 1.5, None),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src, text = Path(args.src), Path(args.text)
    sprites = {p.stem: cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
               for p in text.glob("*.png")}
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
        on_screen = None
        if name == "compare":
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
            frame = (portrait(raw) if layout == "portrait" else landscape(raw)).copy()
            t = n / FPS
            # Every cut settles from a slight enlargement; the opening and the ending fade.
            if name not in ("push", "end") and k < 10:
                frame = zoom(frame, 1 + 0.035 * (1 - k / 10) ** 2)
            if name == "team":
                frame = zoom(frame, 1 + 0.04 * k / count)
                if k < 6:
                    frame = (frame * ((k + 1) / 7)).astype(np.uint8)
                if k >= count - 9:
                    frame = (frame * ((count - 1 - k) / 9) ** 1.5).astype(np.uint8)
            if name == "push" and k < 7:
                frame = (frame * ((k + 1) / 8)).astype(np.uint8)
            if name == "end":
                frame = cv2.GaussianBlur(frame, (0, 0), 6)
                frame = (frame * 0.34).astype(np.uint8)
                if k < 8:
                    frame = (frame * ((k + 1) / 9)).astype(np.uint8)
                if k >= count - 15:
                    frame = (frame * ((count - 1 - k) / 15)).astype(np.uint8)
            if layout == "portrait" and name != "end":
                frame = (frame * shade).astype(np.uint8)
            if name == "outdoor":
                # The training step this frame shows (the sprites carry the numbers), with a
                # thin bar that fills as the run goes on.
                step = round(15000 * min(k * (length / shown) / 132, 1.0) ** 2.4)
                appear = min(1.0, (k + 1) / 8)
                paste(frame, sprites[f"step_{k:03d}"], LEFT - 12, 276, appear)
                y = 318
                bar = frame[y:y + 3, LEFT:LEFT + 560].astype(np.float32)
                bar = bar * (1 - 0.3 * appear) + 255 * 0.3 * appear
                filled = round(560 * step / 15000)
                bar[:, :filled] = bar[:, :filled] * (1 - appear) + np.array([124, 249, 201]) * appear
                frame[y:y + 3, LEFT:LEFT + 560] = bar.astype(np.uint8)
            if on_screen:
                # The headline says what is on screen and changes with every switch.
                since = next((j for j in range(1, k + 1) if on_screen[k - j] != on_screen[k]), k) / FPS
                paste(frame, sprites["f_real" if on_screen[k] == "real" else "f_splat"], 60, HEAD,
                      min(1.0, (since + 1 / FPS) / 0.1) * min(1.0, (count - k) / 4))
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
