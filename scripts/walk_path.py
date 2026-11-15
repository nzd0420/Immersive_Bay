#!/usr/bin/env python3
"""The camera path of a first-person video of a scan made in the viewer.

    python3 scripts/walk_path.py scan-164255 > path.json

Follows the left passthrough camera along the head path recorded while scanning, at 30
frames a second. Stretches in which nothing was photographed for more than 1.5 s (the
tracker was paused, or the walker stood still) are shortened to 1.5 s, crossing them in
a straight line. For every video frame the output gives the camera pose in the COLMAP
world of the scan's workspace and the photo that was the most recent one at that moment.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webxr_scan import CAPTURES, FLIP, Lens, PoseHistory, quat_to_matrix, slerp  # noqa: E402


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x, y, z, w)."""
    from webxr_scan import matrix_to_qvec
    w, x, y, z = matrix_to_qvec(R)
    return np.array([x, y, z, w])


def build(scan_id: str, fps: int = 30, latency_ms: float = 15.0, gap_ms: float = 1500.0) -> dict:
    scan_dir = CAPTURES / scan_id
    calibration = json.loads(Path(__file__).with_name("quest_cameras.json").read_text())
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


if __name__ == "__main__":
    json.dump(build(sys.argv[1]), sys.stdout)
