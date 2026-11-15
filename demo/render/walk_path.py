#!/usr/bin/env python3
"""The camera path of a first-person video of a scan made in the viewer.

    python3 demo/render/walk_path.py scan-164255 > path.json

Follows the left passthrough camera along the head path recorded while scanning, at 30
frames a second. Stretches in which nothing was photographed for more than 1.5 s (the
tracker was paused, or the walker stood still) are shortened to 1.5 s, crossing them in
a straight line. For every video frame the output gives the camera pose in the COLMAP
world of the scan's workspace and the photo that was the most recent one at that moment.

One camera move out of the walk, steadied, ending with a metre forward:

    python3 demo/render/walk_path.py scan-164255 --frames 755:1005 --steady 12 --forward 0.9

--steady is how many frames (at 30 a second) the head's bobbing and jitter is averaged
over; --forward adds two seconds at the end that glide straight ahead, level, looking
the way the last frame looks; --speed 1.6 then plays the whole move 1.6 times as fast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"  # the scan code lives there
sys.path.insert(0, str(SCRIPTS))
from webxr_scan import CAPTURES, FLIP, Lens, PoseHistory, quat_to_matrix, slerp  # noqa: E402


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x, y, z, w)."""
    from webxr_scan import matrix_to_qvec
    w, x, y, z = matrix_to_qvec(R)
    return np.array([x, y, z, w])


def build(scan_id: str, fps: int = 30, latency_ms: float = 15.0, gap_ms: float = 1500.0) -> dict:
    scan_dir = CAPTURES / scan_id
    calibration = json.loads((SCRIPTS / "quest_cameras.json").read_text())
    history = PoseHistory(json.loads((scan_dir / "poses.json").read_text()))
    records = [json.loads(line) for line in (scan_dir / "frames.jsonl").read_text().splitlines() if line]
    left = sorted((r for r in records if r["side"] == "left"), key=lambda r: r["t"])
    lens = Lens(calibration["left"], left[0]["width"], left[0]["height"])

    # Photos that have a pose, and the pose itself.
    photos = []
    for record in left:
        head = history.at(record["t"] - latency_ms, 200.0)
        if head is not None:
            R, C = lens.pose(*head)
            photos.append((record["t"], f"left_{record['n']:04d}.jpg", R, C))

    frames = []
    for index in range(len(photos) - 1):
        t0, _, R0, C0 = photos[index]
        t1, _, R1, C1 = photos[index + 1]
        real = t1 - t0
        shown = min(real, gap_ms)
        count = max(1, round(shown / 1000 * fps))
        q0, q1 = matrix_to_quat(R0), matrix_to_quat(R1)
        # Normal walking follows the head path between the two photos. If any of it is
        # missing, the whole stretch is crossed in a straight line, so nothing jumps.
        heads = []
        if real <= gap_ms:
            heads = [history.at(t0 + step / count * real - latency_ms, 80.0) for step in range(count)]
        followed = bool(heads) and all(head is not None for head in heads)
        for step in range(count):
            a = step / count
            if followed:
                R, C = lens.pose(*heads[step])
            else:
                R, C = quat_to_matrix(slerp(q0, q1, a)), (1 - a) * C0 + a * C1
            frames.append({"R": np.round(R, 6).flatten().tolist(), "C": np.round(C, 5).tolist(),
                           "photo": index})
    # The tracker sometimes snaps to a corrected position (0.28 m in one frame in one of
    # our scans). Average each pose with its neighbours so that reads as a quick slide.
    centres = np.array([frame["C"] for frame in frames])
    quats = np.array([matrix_to_quat(np.array(frame["R"]).reshape(3, 3)) for frame in frames])
    for i in range(1, len(quats)):  # keep neighbouring quaternions on the same side
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    weights = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    reach = len(weights) // 2
    for i, frame in enumerate(frames):
        lo, hi = max(0, i - reach), min(len(frames), i + reach + 1)
        w = weights[lo - i + reach: hi - i + reach, None]
        frame["C"] = np.round((centres[lo:hi] * w).sum(0) / w.sum(), 5).tolist()
        q = (quats[lo:hi] * w).sum(0)
        frame["R"] = np.round(quat_to_matrix(q / np.linalg.norm(q)), 6).flatten().tolist()

    fov = float(np.degrees(2 * np.arctan(lens.height / 2 / lens.fy)))
    return {"scan": scan_id, "fps": fps, "fov": fov, "photos": [p[1] for p in photos],
            "frames": frames}


def cut(walk: dict, first: int, last: int, steady: float = 0.0, forward: float = 0.0,
        forward_seconds: float = 2.0, speed: float = 1.0) -> dict:
    """Frames first..last of a walk, optionally steadied and continued straight ahead."""
    frames = walk["frames"]
    lead = min(first, round(3 * steady))  # real frames before the cut, so its start is smoothed too
    part = frames[first - lead:last + 1]
    centres = np.array([frame["C"] for frame in part])
    quats = np.array([matrix_to_quat(np.array(frame["R"]).reshape(3, 3)) for frame in part])
    photo = [frame["photo"] for frame in part]
    if forward:
        # Level (the workspace's y axis is gravity), in the direction the last frame looks.
        ahead = quat_to_matrix(quats[-1])[:, 2] * np.array([1.0, 0.0, 1.0])
        ahead /= np.linalg.norm(ahead)
        count = round(forward_seconds * walk["fps"])
        t = np.arange(1, count + 1) / count
        glide = t * t * (3 - 2 * t)  # starts and ends at rest
        centres = np.concatenate([centres, centres[-1] + np.outer(glide * forward, ahead)])
        quats = np.concatenate([quats, np.repeat(quats[-1:], count, 0)])
        photo += [photo[-1]] * count
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    if steady:
        reach = round(3 * steady)
        kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / steady) ** 2)
        kernel /= kernel.sum()

        def smooth(values):
            padded = np.pad(values, ((reach, reach), (0, 0)), mode="edge")
            return np.stack([np.convolve(padded[:, i], kernel, "valid") for i in range(values.shape[1])], 1)
        centres, quats = smooth(centres), smooth(quats)
    centres, quats, photo = centres[lead:], quats[lead:], photo[lead:]
    if speed != 1.0:
        # Poses at evenly spaced moments of the faster move, between the frames around them.
        at = np.linspace(0, len(centres) - 1, round((len(centres) - 1) / speed) + 1)
        low = np.minimum(at.astype(int), len(centres) - 2)
        part = (at - low)[:, None]
        centres = (1 - part) * centres[low] + part * centres[low + 1]
        quats = (1 - part) * quats[low] + part * quats[low + 1]
        photo = [photo[i] for i in np.round(at).astype(int)]
    out = [{"R": np.round(quat_to_matrix(q / np.linalg.norm(q)), 6).flatten().tolist(),
            "C": np.round(c, 5).tolist(), "photo": n}
           for c, q, n in zip(centres, quats, photo)]
    return {**walk, "frames": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scan")
    parser.add_argument("--frames", help="FIRST:LAST, only these frames of the walk (both included)")
    parser.add_argument("--steady", type=float, default=0.0, help="frames to average the motion over")
    parser.add_argument("--forward", type=float, default=0.0, help="metres to glide ahead at the end")
    parser.add_argument("--forward-seconds", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0, help="play the move this many times as fast")
    args = parser.parse_args()
    walk = build(args.scan)
    if args.frames or args.steady or args.forward or args.speed != 1.0:
        first, last = (int(v) for v in args.frames.split(":")) if args.frames else (0, len(walk["frames"]) - 1)
        walk = cut(walk, first, last, args.steady, args.forward, args.forward_seconds, args.speed)
    json.dump(walk, sys.stdout)
