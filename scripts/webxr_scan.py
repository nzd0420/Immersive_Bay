"""Turn a scan made inside the WebXR viewer into a scene the training server can load.

The viewer sends passthrough camera frames while its AR session runs, then the headset's
pose history. Here each frame gets the camera pose it was taken from, the result is
written as a COLMAP workspace, copied to the GPU machine, and training is started.

Because the poses come from the same AR session that later displays the result, the
trained Gaussians are already in the viewer's coordinates: they appear where the real
things are, with nothing to line up.

Coordinates. WebXR and three.js: x right, y up, z towards the viewer. COLMAP world and
OpenCV cameras: x right, y down, z forward. The two differ by FLIP = diag(1, -1, -1),
for the world and for the camera axes alike.

The lens-to-head convention follows scripts/quest_to_colmap.py, which was checked against
real OpenQuestCapture recordings from the same headset.

Needs numpy only.
"""
from __future__ import annotations

import bisect
import http.client
import json
import shutil
import struct
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "captures"
FLIP = np.diag([1.0, -1.0, -1.0])  # three.js <-> COLMAP, world and camera axes
TO_UNITY = np.diag([1.0, 1.0, -1.0])  # three.js <-> Unity (left-handed, z forward)
FLIP_Y = np.diag([1.0, -1.0, 1.0])  # Unity <-> COLMAP


def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP quaternion (w, x, y, z)."""
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def slerp(q0, q1, a: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


class Lens:
    """One passthrough camera: intrinsics for a given image size, and its pose in the head."""

    def __init__(self, calibration: dict, width: int, height: int):
        scale = width / calibration["width"]
        self.width, self.height = width, height
        self.fx, self.fy = calibration["fx"] * scale, calibration["fy"] * scale
        self.cx, self.cy = calibration["cx"] * scale, calibration["cy"] * scale
        t, r = calibration["translation"], calibration["rotation"]
        # Android lens pose -> a Unity camera in the head frame (see quest_to_colmap.py).
        self.t_head = np.array([t[0], t[1], -t[2]])
        self.R_head = quat_to_matrix((-r[0], -r[1], r[2], r[3])).T @ np.diag([1.0, -1.0, -1.0])

    def pose(self, R_head_three: np.ndarray, p_head_three: np.ndarray):
        """Camera-to-world rotation and centre in the COLMAP world, from a WebXR head pose."""
        R_u = TO_UNITY @ R_head_three @ TO_UNITY
        p_u = TO_UNITY @ p_head_three
        R_cam, p_cam = R_u @ self.R_head, p_u + R_u @ self.t_head
        return FLIP_Y @ R_cam @ FLIP_Y, FLIP_Y @ p_cam


class PoseHistory:
    """Head poses [t_ms, px, py, pz, qx, qy, qz, qw], interpolated to any time."""

    def __init__(self, rows):
        data = np.array(sorted(rows, key=lambda row: row[0]), dtype=np.float64)
        self.t = data[:, 0].tolist()
        self.p, self.q = data[:, 1:4], data[:, 4:8]

    def at(self, t_ms: float, window_ms: float = 80.0):
        i = bisect.bisect_left(self.t, t_ms)
        if i == 0 or i == len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t_ms - t0 > window_ms or t1 - t_ms > window_ms:
            return None
        a = (t_ms - t0) / (t1 - t0)
        return quat_to_matrix(slerp(self.q[i - 1], self.q[i], a)), (1 - a) * self.p[i - 1] + a * self.p[i]


def write_colmap(out: Path, lenses: dict, frames: list, points: np.ndarray, colours: np.ndarray):
    """frames: (name, camera id, R camera-to-world, centre). Binary COLMAP model."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(lenses)))
        for cid, lens in lenses.items():
            f.write(struct.pack("<iiQQ", cid, 1, lens.width, lens.height))  # 1 = PINHOLE
            f.write(struct.pack("<4d", lens.fx, lens.fy, lens.cx, lens.cy))
    with open(out / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for iid, (name, cid, R, C) in enumerate(frames, start=1):
            R_wc = R.T
            f.write(struct.pack("<i4d3di", iid, *matrix_to_qvec(R_wc), *(-R_wc @ C), cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, (p, c) in enumerate(zip(points, colours), start=1):
            f.write(struct.pack("<Q3d3Bd", pid, *p, *c, 0.0))
            f.write(struct.pack("<Q", 0))


def read_depth(scan_dir: Path) -> list:
    """Depth captures as (time, points [N, 3] in three.js room coordinates)."""
    path = scan_dir / "depth.bin"
    captures = []
    if path.exists():
        data = path.read_bytes()
        offset = 0
        while offset + 12 <= len(data):
            t, count = struct.unpack_from("<dI", data, offset)
            offset += 12
            points = np.frombuffer(data, np.float32, count * 3, offset).reshape(-1, 3)
            offset += count * 12
            captures.append((t, points.astype(np.float64)))
    return captures


def depth_points(captures: list, frames: list, lenses: dict, times: dict, images: Path,
                 voxel: float = 0.015, limit: int = 150000):
    """Starting points from the depth sensor, coloured by the photo taken at the same time.

    Each capture is projected into the left-camera photo nearest in time. Points that
    photo does not see stay grey. One point is kept per `voxel`-sized cell.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None
    left = [(times[name], name, cid, R, C) for name, cid, R, C in frames if name.startswith("left")]
    left = left or [(times[name], name, cid, R, C) for name, cid, R, C in frames]
    stamps = np.array([item[0] for item in left])
    xyz, rgb = [], []
    for t, points in captures:
        world = points @ FLIP.T  # three.js room -> COLMAP world
        colour = np.full(world.shape, 128, np.uint8)
        _, name, cid, R, C = left[int(np.argmin(np.abs(stamps - t)))]
        if Image is not None:
            lens = lenses[cid]
            cam = (world - C) @ R  # world -> camera
            z = np.where(cam[:, 2] > 0.1, cam[:, 2], np.nan)
            u = lens.fx * cam[:, 0] / z + lens.cx
            v = lens.fy * cam[:, 1] / z + lens.cy
            seen = (u >= 0) & (u < lens.width - 1) & (v >= 0) & (v < lens.height - 1)
            if seen.any():
                photo = np.asarray(Image.open(images / name).convert("RGB"))
                colour[seen] = photo[v[seen].astype(int), u[seen].astype(int)]
        xyz.append(world)
        rgb.append(colour)
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    # Prefer coloured points when several fall in one cell.
    order = np.argsort((rgb == 128).all(axis=1), kind="stable")
    xyz, rgb = xyz[order], rgb[order]
    _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    if len(first) > limit:
        first = np.random.default_rng(0).choice(first, limit, replace=False)
    return xyz[first], rgb[first]


def build_workspace(scan_dir: Path, out: Path, calibration: dict, latency_ms: float) -> dict:
    """Pair every received frame with its camera pose and write the COLMAP workspace."""
    history = PoseHistory(json.loads((scan_dir / "poses.json").read_text()))
    records = [json.loads(line) for line in (scan_dir / "frames.jsonl").read_text().splitlines() if line]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    lenses, ids, frames, times, dropped = {}, {}, [], {}, 0
    for record in sorted(records, key=lambda r: (r["t"], r["side"])):
        side = record["side"]
        if side not in ids:
            ids[side] = len(ids) + 1
            lenses[ids[side]] = Lens(calibration[side], record["width"], record["height"])
        head = history.at(record["t"] - latency_ms)
        if head is None:
            dropped += 1
            continue
        R, C = lenses[ids[side]].pose(*head)
        name = f"{side}_{record['n']:04d}.jpg"
        shutil.copyfile(scan_dir / "frames" / name, out / "images" / name)
        frames.append((name, ids[side], R, C))
        times[name] = record["t"]
    if len(frames) < 8:
        raise ValueError(f"only {len(frames)} usable frames ({dropped} had no head pose)")

    captures = read_depth(scan_dir)
    if sum(len(points) for _, points in captures) >= 3000:
        points, colours = depth_points(captures, frames, lenses, times, out / "images")
        seeded = "depth sensor"
    else:
        # No depth: seed points along the view rays, where the cameras were looking.
        rng = np.random.default_rng(0)
        pick = rng.integers(0, len(frames), 40000)
        centres = np.stack([frames[i][3] for i in pick])
        forward = np.stack([frames[i][2][:, 2] for i in pick])
        side_step = np.stack([frames[i][2][:, 0] for i in pick]) * rng.uniform(-0.6, 0.6, (len(pick), 1))
        up_step = np.stack([frames[i][2][:, 1] for i in pick]) * rng.uniform(-0.6, 0.6, (len(pick), 1))
        depth = rng.uniform(0.3, 3.0, (len(pick), 1))
        points = centres + (forward + side_step + up_step) * depth
        colours = rng.integers(90, 170, points.shape).astype(np.uint8)
        seeded = "random"
    write_colmap(out / "sparse" / "0", lenses, frames, points, colours)

    cams = np.stack([frame[3] for frame in frames])
    summary = {
        "source": "WebXR scan", "frames": len(frames), "dropped": dropped,
        "latency_ms": latency_ms, "points": int(len(points)), "points_from": seeded,
        "depth_captures": len(captures),
        "coloured_points": int((colours != 128).any(axis=1).sum()),
        "camera_extent_m": np.round(cams.max(0) - cams.min(0), 2).tolist(),
        # The scene is already in the AR session's coordinates: nothing to line up.
        "anchored": True,
        "world": "COLMAP; y down (gravity), metres; three.js world times diag(1, -1, -1)",
    }
    (out / "quest_capture.json").write_text(json.dumps(summary, indent=2))
    return summary


class Scans:
    """Receives scans from the viewer and turns each into a training run."""

    def __init__(self, upstream_host: str, upstream_port: int, ssh_host: str, remote_dir: str,
                 latency_ms: float = 30.0, steps: int = 3000):
        self.upstream = (upstream_host, upstream_port)
        self.ssh_host, self.remote_dir = ssh_host, remote_dir
        self.latency_ms, self.steps = latency_ms, steps
        self.calibration = json.loads((Path(__file__).with_name("quest_cameras.json")).read_text())
        self.status = {"state": "idle"}
        self.lock = threading.Lock()

    def _dir(self, scan_id: str) -> Path:
        safe = "".join(c for c in scan_id if c.isalnum() or c in "_-")
        if not safe:
            raise ValueError("bad scan id")
        return CAPTURES / safe

    def add_frame(self, scan_id: str, side: str, n: int, t: float, width: int, height: int,
                  jpeg: bytes):
        if side not in ("left", "right"):
            raise ValueError("bad side")
        scan_dir = self._dir(scan_id)
        (scan_dir / "frames").mkdir(parents=True, exist_ok=True)
        (scan_dir / "frames" / f"{side}_{n:04d}.jpg").write_bytes(jpeg)
        with self.lock:
            with open(scan_dir / "frames.jsonl", "a") as f:
                f.write(json.dumps({"side": side, "n": n, "t": t, "width": width, "height": height}) + "\n")
            received = self.status.get("frames", 0) + 1 if self.status.get("id") == scan_id else 1
            self.status = {"id": scan_id, "state": "receiving", "frames": received}

    def add_depth(self, scan_id: str, t: float, data: bytes):
        """One depth capture: float32 x, y, z triples in three.js room coordinates."""
        scan_dir = self._dir(scan_id)
        scan_dir.mkdir(parents=True, exist_ok=True)
        count = len(data) // 12
        with self.lock:
            with open(scan_dir / "depth.bin", "ab") as f:
                f.write(struct.pack("<dI", t, count) + data[: count * 12])

    def finish(self, scan_id: str, poses: list, options: dict):
        scan_dir = self._dir(scan_id)
        (scan_dir / "poses.json").write_text(json.dumps(poses))
        (scan_dir / "options.json").write_text(json.dumps(options))
        threading.Thread(target=self._process, args=(scan_dir.name, options), daemon=True).start()

    def _set(self, scan_id, state, **extra):
        self.status = {"id": scan_id, "state": state, **extra}
        print(f"[scan {scan_id}] {state} {extra}", flush=True)

    def _process(self, scan_id: str, options: dict):
        try:
            scan_dir, out = CAPTURES / scan_id, CAPTURES / f"{scan_id}_colmap"
            self._set(scan_id, "building")
            summary = build_workspace(scan_dir, out, self.calibration,
                                      float(options.get("latency_ms", self.latency_ms)))
            self._set(scan_id, "uploading", frames=summary["frames"])
            remote = f"{self.remote_dir}/scenes/{scan_id}"
            subprocess.run(["ssh", "-o", "BatchMode=yes", self.ssh_host, f"mkdir -p {remote}"],
                           check=True, timeout=60)
            subprocess.run(["rsync", "-az", "--delete", f"{out}/", f"{self.ssh_host}:{remote}/"],
                           check=True, timeout=900)
            self._set(scan_id, "starting", frames=summary["frames"])
            body = json.dumps({"scene": scan_id, "max_steps": int(options.get("steps", self.steps))})
            connection = http.client.HTTPConnection(*self.upstream, timeout=60)
            connection.request("POST", "/api/start", body=body)
            reply = json.loads(connection.getresponse().read())
            connection.close()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "the training server refused the scene"))
            self._set(scan_id, "training", frames=summary["frames"])
        except Exception as error:
            traceback.print_exc()
            self._set(scan_id, "error", message=str(error)[:300])
