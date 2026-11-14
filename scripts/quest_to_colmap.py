#!/usr/bin/env python3
"""Turn an OpenQuestCapture recording into a COLMAP workspace that the training server reads.

    python scripts/quest_to_colmap.py captures/20261114_153012
    python scripts/quest_to_colmap.py captures/20261114_153012 -o captures/desk --check

No COLMAP run is needed: camera poses come from the headset's own tracking, and the
initial point cloud comes from the headset's depth maps. The output is what
server/live_server.py expects for --scene NAME=DIR:

    OUT/images/left_<unix ms>.jpg, right_<unix ms>.jpg    undistorted RGB frames
    OUT/sparse/0/cameras.bin   one PINHOLE camera per passthrough camera
    OUT/sparse/0/images.bin    world-to-camera pose of every frame
    OUT/sparse/0/points3D.bin  coloured points from the depth maps
    OUT/check/*.jpg            (--check) the points projected onto a few frames

Coordinates. The recording is in Unity's left-handed tracking space (x right, y up,
z forward; cameras look along +z with y up). COLMAP wants a right-handed world and
OpenCV cameras (x right, y down, z forward). Flipping y does both at once:
world_colmap = S world_unity and R_colmap = S R_unity S, with S = diag(1, -1, 1).
The y axis of the output world therefore points down, as in most COLMAP scenes.

The recording format and the head-to-camera convention follow OpenQuestCapture and
its quest-3d-reconstruction scripts (both MIT): https://github.com/samuelm2/OpenQuestCapture

Needs numpy and opencv-python.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("This script needs OpenCV: pip install opencv-python")

SIDES = ("left", "right")
FLIP_Y = np.diag([1.0, -1.0, 1.0])  # Unity <-> COLMAP, for both world and camera axes
POSE_WINDOW_MS = 60  # a frame needs a head pose sample at most this far on each side


# ------------------------------------------------------------------ rotations


def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> COLMAP qvec (w, x, y, z), w >= 0."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def slerp(q0, q1, a: float) -> np.ndarray:
    """Spherical interpolation between (x, y, z, w) quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def unity_to_colmap_c2w(R_u: np.ndarray, p_u: np.ndarray):
    """Unity camera-to-world (camera y up) -> COLMAP world, OpenCV camera (y down)."""
    return FLIP_Y @ R_u @ FLIP_Y, FLIP_Y @ p_u


# ------------------------------------------------------------------ recording


@dataclass
class Camera:
    side: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_head: np.ndarray  # camera-to-head rotation, Unity camera axes
    t_head: np.ndarray  # camera position in the head frame
    planes: list  # [(buffer_size, row_stride, pixel_stride)] * 3


def load_camera(rec: Path, side: str) -> Camera:
    with open(rec / f"{side}_camera_image_format.json", encoding="utf-8") as f:
        fmt = json.load(f)
    with open(rec / f"{side}_camera_characteristics.json", encoding="utf-8") as f:
        ch = json.load(f)
    if fmt.get("format", "YUV_420_888") not in ("YUV_420_888", 35):
        raise ValueError(f"{side}: unsupported image format {fmt.get('format')!r}")
    width, height = int(fmt["width"]), int(fmt["height"])
    planes = [(int(p["bufferSize"]), int(p["rowStride"]), int(p["pixelStride"])) for p in fmt["planes"]]
    if len(planes) != 3:
        raise ValueError(f"{side}: expected 3 YUV planes, got {len(planes)}")

    # Camera2 intrinsics are in active-array pixels. A stream with another size is the
    # active array, centre-cropped to the stream's aspect ratio, then scaled.
    a = ch["sensor"]["activeArraySize"]
    aw, ah = a["right"] - a["left"], a["bottom"] - a["top"]
    intr = ch["intrinsics"]
    fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    if (aw, ah) != (width, height):
        if aw * height > ah * width:  # active array is wider: crop its sides
            cw, chh = ah * width / height, ah
        else:
            cw, chh = aw, aw * height / width
        x0, y0 = (aw - cw) / 2, (ah - chh) / 2
        s = width / cw
        fx, fy, cx, cy = fx * s, fy * s, (cx - x0) * s, (cy - y0) * s
    if ch.get("distortion") and any(abs(float(v)) > 1e-6 for v in ch["distortion"]):
        print(f"[{side}] note: the camera reports lens distortion {ch['distortion']}; "
              "frames are used as already rectified, as OpenQuestCapture's own exporter does")

    # Lens pose relative to the head, converted from Android's convention to a Unity
    # camera (x right, y up, z forward). Same steps as quest-3d-reconstruction.
    pose = ch.get("pose", {})
    t = list(pose.get("translation", []))
    t_head = np.array([t[0], t[1], -t[2]], dtype=np.float64) if len(t) >= 3 else np.zeros(3)
    r = pose.get("rotation", [])
    if len(r) >= 4:
        q = (-r[0], -r[1], r[2], r[3])
        R_head = quat_to_matrix(q).T @ np.diag([1.0, -1.0, -1.0])  # inverse, then 180 deg about x
    else:
        R_head = np.eye(3)
    return Camera(side, width, height, fx, fy, cx, cy, R_head, t_head, planes)


class HeadPoses:
    """hmd_poses.csv, interpolated to arbitrary unix-millisecond timestamps."""

    def __init__(self, path: Path):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append([float(row[k]) for k in (
                        "unix_time", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w")])
                except (KeyError, TypeError, ValueError):
                    continue  # partial last line, header repeats
        if not rows:
            raise ValueError(f"no head poses in {path}")
        data = np.array(rows)
        data = data[np.argsort(data[:, 0], kind="stable")]
        self.t = data[:, 0].tolist()
        self.p = data[:, 1:4]
        self.q = data[:, 4:8]

    def at(self, t_ms: float):
        """(R, p) of the head in Unity space at t_ms, or None if no pose is close enough."""
        i = bisect.bisect_left(self.t, t_ms)
        if i < len(self.t) and self.t[i] == t_ms:
            return quat_to_matrix(self.q[i]), self.p[i].copy()
        if i == 0 or i == len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t_ms - t0 > POSE_WINDOW_MS or t1 - t_ms > POSE_WINDOW_MS:
            return None
        a = (t_ms - t0) / (t1 - t0)
        p = (1 - a) * self.p[i - 1] + a * self.p[i]
        return quat_to_matrix(slerp(self.q[i - 1], self.q[i], a)), p


# ------------------------------------------------------------------ images


def _plane(buf: np.ndarray, offset: int, size: int, rows: int, cols: int,
           row_stride: int, pixel_stride: int) -> np.ndarray:
    data = buf[offset:offset + size]
    need = rows * row_stride
    if data.size < need:  # the last row of a plane is usually not padded to the stride
        data = np.concatenate([data, np.zeros(need - data.size, np.uint8)])
    return data[:need].reshape(rows, row_stride)[:, : cols * pixel_stride : pixel_stride]


def decode_yuv(path: Path, cam: Camera) -> np.ndarray:
    """A raw Camera2 YUV_420_888 dump (planes back to back, with strides) -> BGR image."""
    buf = np.fromfile(path, dtype=np.uint8)
    (ys, yr, yp), (us, ur, up), (vs, vr, vp) = cam.planes
    if buf.size < ys + us + vs:
        raise ValueError(f"{path.name}: truncated ({buf.size} bytes)")
    w, h = cam.width, cam.height
    cw, chh = (w + 1) // 2, (h + 1) // 2
    y = _plane(buf, 0, ys, h, w, yr, yp)
    u = _plane(buf, ys, us, chh, cw, ur, up)
    v = _plane(buf, ys + us, vs, chh, cw, vr, vp)
    i420 = np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(h * 3 // 2, w)
    return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)


def sharpness(bgr: np.ndarray) -> float:
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


@dataclass
class Frame:
    side: str
    t_ms: int
    name: str
    R: np.ndarray  # camera-to-world, COLMAP
    C: np.ndarray  # camera centre, COLMAP
    thumb: np.ndarray  # small RGB copy for colouring points
    sharp: float


# ------------------------------------------------------------------ depth


def depth_points(rec: Path, side: str, every: int, stride: int, depth_max: float):
    """Yield (t_ms, points [N, 3] in COLMAP world) for the depth frames of one side."""
    desc_path = rec / f"{side}_depth_descriptors.csv"
    left_path = rec / "left_depth_descriptors.csv"
    if not desc_path.exists():
        return
    with open(desc_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    # Both eyes' files are named after the left eye's timestamp.
    with open(left_path, newline="", encoding="utf-8") as f:
        left_rows = [r for r in csv.DictReader(f)]
    for i in range(0, len(rows), every):
        r = rows[i]
        try:
            t_ms = int(float(r["timestamp_ms"]))
            file_t = int(float(left_rows[i]["timestamp_ms"])) if i < len(left_rows) else t_ms
            w, h = int(float(r["width"])), int(float(r["height"]))
            near, far = float(r["near_z"]), float(r["far_z"])
            tl, tr = float(r["fov_left_angle_tangent"]), float(r["fov_right_angle_tangent"])
            tt, tb = float(r["fov_top_angle_tangent"]), float(r["fov_down_angle_tangent"])
            p = np.array([float(r[f"create_pose_location_{k}"]) for k in "xyz"])
            q = [float(r[f"create_pose_rotation_{k}"]) for k in "xyzw"]
        except (KeyError, ValueError):
            continue
        path = rec / f"{side}_depth" / f"{file_t}.raw"
        if not path.exists():
            path = rec / f"{side}_depth" / f"{t_ms}.raw"
        if not path.exists():
            continue
        d = np.fromfile(path, dtype="<f4")
        if d.size != w * h:
            continue
        d = d.reshape(h, w)[stride // 2 :: stride, stride // 2 :: stride]
        # Depth buffer (0..1, OpenGL-style NDC) -> distance along the view axis.
        ndc = d.astype(np.float64) * 2 - 1
        if math.isinf(far) or far < near:
            z = 2 * near / (1 - ndc)
        else:
            z = 2 * far * near / ((far + near) - ndc * (far - near))
        vs, us = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
        ok = np.isfinite(z) & (d > 0) & (d < 1) & (z > near * 1.01) & (z < depth_max)
        if not ok.any():
            continue
        z, us, vs = z[ok], us[ok] + 0.5, vs[ok] + 0.5
        fx, fy = w / (tl + tr), h / (tt + tb)
        cx, cy = w * tl / (tl + tr), h * tt / (tt + tb)  # rows run top to bottom
        local = np.stack([(us - cx) / fx * z, (cy - vs) / fy * z, z], axis=1)  # Unity camera
        world_u = local @ quat_to_matrix(q).T + p
        yield t_ms, world_u @ FLIP_Y.T


def project(frame: Frame, K: np.ndarray, pts: np.ndarray):
    """Pixel coordinates [N, 2] and depth [N] of COLMAP-world points in a frame."""
    cam = (pts - frame.C) @ frame.R  # world -> camera: R^T (x - C)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


# ------------------------------------------------------------------ COLMAP files


def write_model(out: Path, cams: dict, frames: list, xyz: np.ndarray, rgb: np.ndarray):
    """Binary COLMAP model, the layout read by COLMAP, pycolmap and gsplat."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, c in cams.items():
            f.write(struct.pack("<iiQQ", cid, 1, c.width, c.height))  # model 1 = PINHOLE
            f.write(struct.pack("<4d", c.fx, c.fy, c.cx, c.cy))
    with open(out / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for iid, (fr, cid) in enumerate(frames, start=1):
            R_wc = fr.R.T
            qvec = matrix_to_qvec(R_wc)
            tvec = -R_wc @ fr.C
            f.write(struct.pack("<i4d3di", iid, *qvec, *tvec, cid))
            f.write(fr.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D keypoints
    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for pid, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(struct.pack("<Q3d3Bd", pid, *p, *c, 0.0))
            f.write(struct.pack("<Q", 0))  # empty track


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = xyz, rgb
        f.write(rec.tobytes())


# ------------------------------------------------------------------ main


def convert(args) -> int:
    rec = Path(args.recording)
    out = Path(args.output) if args.output else rec.with_name(rec.name + "_colmap")
    if not (rec / "hmd_poses.csv").exists():
        print(f"{rec} is not an OpenQuestCapture recording (no hmd_poses.csv)", file=sys.stderr)
        return 1
    sides = [s for s in args.cameras.split(",") if s]
    started = time.time()
    poses = HeadPoses(rec / "hmd_poses.csv")
    print(f"{len(poses.t)} head poses over {(poses.t[-1] - poses.t[0]) / 1000:.1f} s")

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.iterdir():  # the parser pairs every file in images/ with a pose
        if old.is_file():
            old.unlink()

    cams, Ks, frames = {}, {}, []
    thumb_scale = 320
    for side in sides:
        if not (rec / f"{side}_camera_raw").is_dir():
            print(f"[{side}] no {side}_camera_raw/, skipped")
            continue
        cam = load_camera(rec, side)
        print(f"[{side}] {cam.width}x{cam.height}  fx={cam.fx:.1f} fy={cam.fy:.1f} "
              f"cx={cam.cx:.1f} cy={cam.cy:.1f}  lens at {np.round(cam.t_head, 3).tolist()} m from head")
        files = sorted((rec / f"{side}_camera_raw").glob("*.yuv"), key=lambda p: int(p.stem))
        files = files[:: args.every]
        kept, no_pose, bad = [], 0, 0
        for n, path in enumerate(files):
            t_ms = int(path.stem)
            head = poses.at(t_ms)
            if head is None:
                no_pose += 1
                continue
            try:
                bgr = decode_yuv(path, cam)
            except ValueError as e:
                print(f"[{side}] {e}")
                bad += 1
                continue
            R_h, p_h = head
            R_c, p_c = unity_to_colmap_c2w(R_h @ cam.R_head, p_h + R_h @ cam.t_head)
            name = f"{side}_{t_ms}.{'png' if args.png else 'jpg'}"
            ts = thumb_scale / max(bgr.shape[:2])
            thumb = cv2.cvtColor(cv2.resize(bgr, None, fx=ts, fy=ts, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB)
            kept.append((Frame(side, t_ms, name, R_c, p_c, thumb, sharpness(bgr)), bgr))
            if (n + 1) % 50 == 0:
                print(f"[{side}] decoded {n + 1}/{len(files)}", flush=True)
        if args.min_sharpness > 0:
            before = len(kept)
            kept = [k for k in kept if k[0].sharp >= args.min_sharpness]
            print(f"[{side}] dropped {before - len(kept)} blurry frames")
        cid = len(cams) + 1
        cams[cid] = cam
        Ks[side] = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        for fr, bgr in kept:
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if args.png else [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            cv2.imwrite(str(img_dir / fr.name), bgr, params)
            frames.append((fr, cid))
        print(f"[{side}] {len(kept)} frames kept, {no_pose} without a head pose, {bad} unreadable")

    if len(frames) < 2:
        print("Fewer than two usable frames; nothing to train on.", file=sys.stderr)
        return 1
    frames.sort(key=lambda fc: fc[0].name)

    # Initial points: depth maps, coloured from the closest frame of the same camera.
    cam_of = {c.side: c for c in cams.values()}
    by_side = {s: sorted([fr for fr, _ in frames if fr.side == s], key=lambda f: f.t_ms) for s in sides}
    xyz_all, rgb_all, agree = [], [], []
    for side in sides:
        seq = by_side.get(side) or by_side.get("left") or by_side.get("right")
        if not seq:
            continue
        other = by_side.get("right" if side == "left" else "left") or []
        times = [f.t_ms for f in seq]
        for t_ms, pts in depth_points(rec, side, args.depth_every, args.depth_stride, args.depth_max):
            j = min(range(len(times)), key=lambda k: abs(times[k] - t_ms))
            if abs(times[j] - t_ms) > 500:
                continue
            fr = seq[j]
            K = Ks[fr.side]
            uv, z = project(fr, K, pts)
            s = fr.thumb.shape[1] / cam_of[fr.side].width
            ui, vi = np.floor(uv[:, 0] * s).astype(np.int64), np.floor(uv[:, 1] * s).astype(np.int64)
            ok = (z > 0.05) & (ui >= 0) & (vi >= 0) & (ui < fr.thumb.shape[1]) & (vi < fr.thumb.shape[0])
            if not ok.any():
                continue
            xyz_all.append(pts[ok])
            rgb_all.append(fr.thumb[vi[ok], ui[ok]])
            if other and len(agree) < 200:  # do both cameras see the same colour there?
                times_o = [f.t_ms for f in other]
                k = min(range(len(other)), key=lambda kk: abs(times_o[kk] - t_ms))
                fo = other[k]
                uv2, z2 = project(fo, Ks[fo.side], pts[ok])
                s2 = fo.thumb.shape[1] / cam_of[fo.side].width
                u2, v2 = np.floor(uv2[:, 0] * s2).astype(np.int64), np.floor(uv2[:, 1] * s2).astype(np.int64)
                ok2 = (z2 > 0.05) & (u2 >= 0) & (v2 >= 0) & (u2 < fo.thumb.shape[1]) & (v2 < fo.thumb.shape[0])
                if ok2.sum() > 50:
                    a = fr.thumb[vi[ok], ui[ok]][ok2].astype(np.float64).mean(1)
                    b = fo.thumb[v2[ok2], u2[ok2]].astype(np.float64).mean(1)
                    agree.append((np.median(np.abs(a - b)), np.median(np.abs(a - np.random.permutation(b)))))

    if xyz_all:
        xyz = np.concatenate(xyz_all)
        rgb = np.concatenate(rgb_all)
        key = np.floor(xyz / args.voxel).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        counts = np.bincount(inv)
        xyz = np.stack([np.bincount(inv, xyz[:, i]) / counts for i in range(3)], axis=1)
        rgb = np.stack([np.bincount(inv, rgb[:, i].astype(np.float64)) / counts for i in range(3)], axis=1)
        if len(xyz) > args.max_points:
            pick = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
            xyz, rgb = xyz[pick], rgb[pick]
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        print(f"{len(xyz)} initial points from the depth maps ({args.voxel * 100:.1f} cm voxels)")
    else:
        # No depth: scatter points in front of the cameras so training still starts.
        print("No usable depth maps; seeding random points in front of the cameras")
        rng = np.random.default_rng(0)
        pick = rng.integers(0, len(frames), 20000)
        dirs = np.stack([frames[i][0].R[:, 2] for i in pick])
        xyz = np.stack([frames[i][0].C for i in pick]) + dirs * rng.uniform(0.3, args.depth_max, (len(pick), 1))
        xyz += rng.normal(0, 0.3, xyz.shape)
        rgb = rng.integers(60, 200, xyz.shape).astype(np.uint8)

    write_model(out / "sparse" / "0", cams, frames, xyz, rgb)
    write_ply(out / "sparse" / "0" / "points3D.ply", xyz, rgb)

    centres = np.stack([fr.C for fr, _ in frames])
    path_len = float(np.linalg.norm(np.diff(centres[np.argsort([fr.t_ms for fr, _ in frames])], axis=0), axis=1).sum())
    summary = {
        "recording": str(rec), "frames": len(frames),
        "cameras": {c.side: {"width": c.width, "height": c.height, "fx": c.fx, "fy": c.fy,
                             "cx": c.cx, "cy": c.cy} for c in cams.values()},
        "points": int(len(xyz)), "camera_path_m": round(path_len, 2),
        "world": "COLMAP; y down (gravity), metres, from the headset's tracking space",
    }
    if agree:
        summary["left_right_colour_diff"] = round(float(np.median([a for a, _ in agree])), 1)
        summary["shuffled_colour_diff"] = round(float(np.median([b for _, b in agree])), 1)
    with open(out / "quest_capture.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.check:
        chk = out / "check"
        chk.mkdir(exist_ok=True)
        for old in chk.glob("*.jpg"):
            old.unlink()
        picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
        for i in sorted(set(picks.tolist())):
            fr, cid = frames[i]
            img = cv2.imread(str(img_dir / fr.name))
            uv, z = project(fr, Ks[fr.side], xyz)
            ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] < img.shape[0])
            dots = img.copy()
            far_first = np.argsort(-z[ok])  # nearer points paint over farther ones
            for (u, v), c in zip(uv[ok][far_first].astype(int), rgb[ok][far_first]):
                cv2.circle(dots, (int(u), int(v)), 2, (int(c[2]), int(c[1]), int(c[0])), -1)
            cv2.imwrite(str(chk / f"{Path(fr.name).stem}.jpg"), np.hstack([img, dots]))
        print(f"Check images in {chk}: left is the frame, right has the points drawn on it")

    print()
    print(f"Wrote {out}")
    print(f"  {len(frames)} frames from {len(cams)} camera(s), {len(xyz)} points, "
          f"camera path {path_len:.1f} m, {time.time() - started:.0f} s")
    if agree:
        a, b = summary["left_right_colour_diff"], summary["shuffled_colour_diff"]
        verdict = "poses look consistent" if a < 0.6 * b else "POSES MAY BE WRONG, look at --check images"
        print(f"  left/right colour difference at the same 3D points: {a} (unrelated pixels: {b}) -> {verdict}")
    print(f"  train with: --scene {out.name}={out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", help="OpenQuestCapture folder (YYYYMMDD_hhmmss)")
    ap.add_argument("-o", "--output", help="output workspace (default: <recording>_colmap)")
    ap.add_argument("--cameras", default="left,right", help="which passthrough cameras to use")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth camera frame")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="drop frames whose Laplacian variance is below this (try 30); 0 keeps all")
    ap.add_argument("--png", action="store_true", help="write PNG instead of JPEG")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--depth-every", type=int, default=2, help="use every Nth depth frame")
    ap.add_argument("--depth-stride", type=int, default=4, help="use every Nth depth pixel")
    ap.add_argument("--depth-max", type=float, default=3.0, help="ignore depth beyond this (m)")
    ap.add_argument("--voxel", type=float, default=0.01, help="point cloud voxel size (m)")
    ap.add_argument("--max-points", type=int, default=150000, help="cap on initial points")
    ap.add_argument("--check", action="store_true", help="write overlay images to OUT/check/")
    args = ap.parse_args(argv)
    if args.every < 1 or args.depth_every < 1 or args.depth_stride < 1:
        ap.error("--every, --depth-every and --depth-stride must be at least 1")
    return convert(args)


if __name__ == "__main__":
    sys.exit(main())
