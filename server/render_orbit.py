#!/usr/bin/env python
"""Render a slow horizontal orbit around a finished reconstruction.

    python render_orbit.py --frame final_frame.bin --info final_frame.json --out orbit.mp4

The input is one frame exactly as the live server streams it (16 bytes per Gaussian, the
layout written by `pack_frame` in live_server.py), saved together with the scene's
description. So the video shows the very model that was on screen: nothing is trained
again. scripts/render_orbit.sh fetches both from the running server and runs this.

The camera circles the scene's centre about the scene's up direction, at a distance and
an elevation, and keeps the horizon level. The centre is the middle of the box around the
more opaque Gaussians. Azimuth 0 looks from the side the scan was made from. With
`--push-in SECONDS` the orbit ends by slowing down and moving inside the scene, to where
the scan was made from, and stays there looking at what was scanned. `--stills N` writes
N evenly spaced JPEGs instead of a video, for choosing the view. For a vertical video give
`--width 2160 --height 3840` and a taller `--fov`; `--last 15` renders only the last 15 s.

Needs torch, gsplat, numpy and imageio-ffmpeg.
"""
import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from gsplat.rendering import rasterization

DEVICE = "cuda"


def unpack_frame(path: Path) -> dict:
    """The inverse of pack_frame: positions, scales, rotations (w, x, y, z), colour, opacity."""
    rows = np.fromfile(path, np.uint8).reshape(-1, 16)
    rows = rows[rows[:, 3] > 0]  # opacity 0 marks a Gaussian that is not shown
    rgba = rows[:, 0:4].astype(np.float32) / 255.0
    means = rows[:, 4:10].copy().view(np.float16).astype(np.float32)
    uv = rows[:, 10:12].astype(np.float32) / 255.0 * 2.0 - 1.0
    scales = np.exp((rows[:, 12:15].astype(np.float32) - 1.0) * (21.0 / 254.0) - 12.0)
    angle = rows[:, 15].astype(np.float32) / 255.0 * math.pi
    # the rotation axis, folded into a square (octahedral encoding)
    z = 1.0 - np.abs(uv[:, 0]) - np.abs(uv[:, 1])
    t = np.maximum(-z, 0.0)
    x = uv[:, 0] + np.where(uv[:, 0] >= 0, -t, t)
    y = uv[:, 1] + np.where(uv[:, 1] >= 0, -t, t)
    axis = np.stack([x, y, z], axis=1)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    quats = np.concatenate([np.cos(angle / 2)[:, None], axis * np.sin(angle / 2)[:, None]], axis=1)
    to = lambda array: torch.from_numpy(np.ascontiguousarray(array)).float().to(DEVICE)
    return {"means": to(means), "scales": to(scales), "quats": to(quats),
            "colors": to(rgba[:, :3]), "opacities": to(rgba[:, 3])}


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera matrix of an OpenCV camera (x right, y down, z forward), horizon level."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.stack([right, down, forward], axis=1)
    camera_to_world[:3, 3] = position
    return np.linalg.inv(camera_to_world)


@torch.no_grad()
def render(splats: dict, viewmat: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    image, _, _ = rasterization(
        means=splats["means"], quats=splats["quats"], scales=splats["scales"],
        opacities=splats["opacities"], colors=splats["colors"],
        viewmats=torch.from_numpy(viewmat).float().to(DEVICE)[None],
        Ks=torch.from_numpy(K).float().to(DEVICE)[None], width=width, height=height,
    )
    return (image[0].clamp(0, 1) * 255).round().byte().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frame", required=True, help="the streamed frame, 16 bytes per Gaussian")
    parser.add_argument("--info", required=True, help="JSON with the scene description")
    parser.add_argument("--out", required=True, help="the .mp4, or a folder with --stills")
    parser.add_argument("--distance", type=float, default=16.0, help="metres from the centre")
    parser.add_argument("--elevation", type=float, default=10.0, help="degrees above the horizon")
    parser.add_argument("--azimuth", type=float, default=0.0, help="degrees, where the orbit starts")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("FORWARD", "SIDE", "UP"), help="move the centre, in metres")
    parser.add_argument("--fov", type=float, default=55.0, help="vertical field of view, degrees")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seconds", type=float, default=30.0, help="for one full turn")
    parser.add_argument("--turns", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clockwise", action="store_true")
    parser.add_argument("--push-in", type=float, default=0.0, metavar="SECONDS",
                        help="end by slowing the orbit and moving inside the scene, over this long")
    parser.add_argument("--hold", type=float, default=5.0, metavar="SECONDS",
                        help="after a push-in, how long to stay inside (drifting slowly forward)")
    parser.add_argument("--last", type=float, default=0.0, metavar="SECONDS",
                        help="render only this much of the end of the video")
    parser.add_argument("--stills", type=int, default=0, help="write this many JPEGs instead")
    parser.add_argument("--stills-from", type=float, default=0.0, metavar="SECONDS",
                        help="with --stills: take them from this time on")
    args = parser.parse_args()

    scene = json.loads(Path(args.info).read_text())["scene"]
    splats = unpack_frame(Path(args.frame))
    to_world = np.array(scene["to_world"]) if scene.get("to_world") else np.eye(4)
    metre = 1.0 / np.cbrt(abs(np.linalg.det(to_world[:3, :3])))  # scene units per metre
    up = np.array(scene["up"], dtype=np.float64)
    up /= np.linalg.norm(up)
    # The middle of what is actually visible: the centre of the box that holds the middle
    # 90% of the more opaque Gaussians along each axis. (Their median sits low, near the
    # floor, and the scene would circle around the edge of the picture.)
    solid = splats["means"][splats["opacities"] > 0.3].double().cpu().numpy()
    low, high = np.percentile(solid, 5, axis=0), np.percentile(solid, 95, axis=0)
    centre = (low + high) / 2
    # Horizontal axes: `front` points from the centre towards where the scan was made.
    front = np.array(scene["view"], dtype=np.float64) - centre
    front -= front.dot(up) * up
    front /= np.linalg.norm(front)
    side = np.cross(up, front)
    target = centre + metre * (args.target[0] * front + args.target[1] * side + args.target[2] * up)

    fy = args.height / 2 / math.tan(math.radians(args.fov) / 2)
    K = np.array([[fy, 0, args.width / 2], [0, fy, args.height / 2], [0, 0, 1]])

    def around(azimuth_degrees: float, out: float, height: float) -> np.ndarray:
        """A point `out` metres from the vertical axis through the target, `height` above it."""
        azimuth = math.radians(azimuth_degrees)
        return target + metre * (out * (math.cos(azimuth) * front + math.sin(azimuth) * side) + height * up)

    # The camera path: one (position, point looked at) for every frame.
    sign = -1.0 if args.clockwise else 1.0
    speed = 360.0 / args.seconds  # degrees per second
    out = args.distance * math.cos(math.radians(args.elevation))
    height = args.distance * math.sin(math.radians(args.elevation))
    path = []
    if args.push_in <= 0:
        count = int(round(args.seconds * args.turns * args.fps))
        for i in range(count):
            path.append((around(args.azimuth + sign * speed * i / args.fps, out, height), target))
    else:
        # The orbit runs at its steady speed, then slows to a stop over the push-in while
        # the camera moves to where the scan was made from and turns to look at what was
        # scanned. The steady part is as long as it takes for the orbit to stop exactly on
        # the side it enters from, after about one turn.
        inside = np.array(scene["view"], dtype=np.float64)
        looked_at = np.array(scene["center"], dtype=np.float64)
        offset = (inside - target) / metre
        inside_height = float(offset.dot(up))
        level = offset - inside_height * up
        inside_out = float(np.linalg.norm(level))
        inside_azimuth = math.degrees(math.atan2(level.dot(side), level.dot(front)))
        turned = (sign * (inside_azimuth - args.azimuth)) % 360.0
        while turned < 300.0 * args.turns:
            turned += 360.0
        slowing = speed * args.push_in / 2  # degrees covered while stopping
        steady = int(round((turned - slowing) / speed * args.fps))
        for i in range(steady):
            path.append((around(args.azimuth + sign * speed * i / args.fps, out, height), target))
        azimuth = args.azimuth + sign * speed * steady / args.fps
        pushing = int(round(args.push_in * args.fps))
        for i in range(pushing):
            u = (i + 1) / pushing
            ease = u * u * u * (u * (6 * u - 15) + 10)  # smooth start and stop
            angle = azimuth + sign * speed * args.push_in * (u - u * u / 2)
            position = around(angle, out + (inside_out - out) * ease, height + (inside_height - height) * ease)
            path.append((position, target + (looked_at - target) * ease))
        forward = (looked_at - inside) / np.linalg.norm(looked_at - inside)
        holding = int(round(args.hold * args.fps))
        for i in range(holding):
            u = (i + 1) / holding
            path.append((inside + forward * metre * 0.15 * u * u, looked_at))
        print(f"orbit {steady / args.fps:.1f} s, push-in {args.push_in:.1f} s to "
              f"{inside_out:.1f} m out and {inside_height:.1f} m up, hold {args.hold:.1f} s", flush=True)

    if args.last:
        path = path[-int(round(args.last * args.fps)):]

    def frame(index: int) -> np.ndarray:
        position, looked_at_point = path[index]
        return render(splats, look_at(position, looked_at_point, up), K, args.width, args.height)

    print(f"{len(splats['means'])} Gaussians, 1 m = {metre:.3f} scene units, "
          f"{len(path)} frames ({len(path) / args.fps:.1f} s)", flush=True)
    if args.stills:
        import cv2

        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        first = min(len(path) - 1, int(round(args.stills_from * args.fps)))
        for index in np.linspace(first, len(path) - 1, args.stills).round().astype(int):
            cv2.imwrite(str(folder / f"t{index / args.fps:05.1f}.jpg"), frame(index)[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{args.stills} stills in {folder}", flush=True)
        return

    stream = imageio_ffmpeg.write_frames(
        args.out, (args.width, args.height), fps=args.fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, output_params=["-crf", "16", "-preset", "medium"],
        macro_block_size=2,
    )
    stream.send(None)
    for index in range(len(path)):
        stream.send(frame(index).tobytes())
        if (index + 1) % 150 == 0:
            print(f"frame {index + 1}/{len(path)}", flush=True)
    stream.close()
    print(f"done: {len(path)} frames, {len(path) / args.fps:.1f} s, {args.out}", flush=True)


if __name__ == "__main__":
    main()
