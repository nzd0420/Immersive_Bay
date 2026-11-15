#!/usr/bin/env python
"""Render first-person videos of a scan along the path it was walked.

    python render_walk.py --scene-dir scenes/scan-164255 --path path.json \\
        --gsplat-examples /path/to/gsplat/examples --out out/ --steps 15000

Trains the scene once and writes three videos of the same length and camera motion, so
they can be cut against each other frame for frame:

    learning.mp4  the walk while the model trains: blurry blobs at the start of the walk,
                  the finished reconstruction at its end
    final.mp4     the walk through the finished reconstruction
    photos.mp4    the photos taken along the walk, each held until the next

For one camera move out of the walk, give its frames. This is 8 seconds that start on the
model after 1000 steps and end on the finished one, the step rising evenly in between:

    ... --frames 755:1005 --first-step 1000 --ease 1 --cap 300000

Several values for --ease write one learning video for each (learning-ease1.mp4, ...).

For any other pacing, name the step at chosen frames; in between it rises evenly. This one
forms out of the first 6000 steps over 5.5 seconds, then reaches the finished model within
eight frames and stays there (writes learning-form.mp4; --keys can be repeated):

    ... --frames 755:1005 --cap 300000 --keys form=0:0,60:971,115:3133,165:6000,173:15000,250:15000

Watch the loss it prints. On our arcade scan, runs with --cap 400000 came apart once the
cap was reached (the mean loss climbing from 0.12 to 0.2 and the views turning to fog)
unless --noise-lr was 5e4; lower noise was worse again. With --cap 300000, which is what
the live server uses, the default noise trained cleanly and gave the best finished model.
Each 2160-pixel video being encoded takes about 4 GB of memory, so do not run several of
these side by side on a small machine.

`path.json` comes from scripts/walk_path.py. Needs what live_server.py needs, plus
imageio-ffmpeg.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy

from live_server import DEVICE, TrainConfig, create_splats, load_colmap_scene


def writer(path: Path, shape: tuple, fps: int):
    stream = imageio_ffmpeg.write_frames(
        str(path), shape, fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, output_params=["-crf", "17", "-preset", "medium"],
        macro_block_size=2,
    )
    stream.send(None)
    return stream


@torch.no_grad()
def render(splats, viewmat, K, shape: tuple, near: float = 0.01) -> bytes:
    image, _, _ = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=viewmat[None], Ks=K[None], width=shape[0], height=shape[1], sh_degree=0,
        near_plane=near,
    )
    return (image[0].clamp(0, 1) * 255).round().byte().cpu().numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--gsplat-examples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--size", type=int, default=1440, help="square videos of this many pixels")
    parser.add_argument("--shape", metavar="WIDTHxHEIGHT", help="videos of this shape instead, e.g. 2160x3840")
    parser.add_argument("--fov", type=float,
                        help="degrees seen across the shorter side (default: what the headset's camera sees)")
    parser.add_argument("--cap", type=int, default=400_000,
                        help="Gaussian cap; nothing has to render this on a headset")
    parser.add_argument("--ease", type=float, nargs="+",
                        help="1: the step rises evenly along the video; above 1, more of the "
                             "video is spent on the early, fast-changing steps (default 2.2)")
    parser.add_argument("--keys", action="append", default=[], metavar="NAME=FRAME:STEP,...",
                        help="a learning video whose step is given at some frames (counted from "
                             "the first rendered frame) and rises evenly between them")
    parser.add_argument("--frames", help="FIRST:LAST, the frames of the path to render (both included)")
    parser.add_argument("--first-step", type=int, default=0,
                        help="the training step shown in the first frame of learning.mp4")
    parser.add_argument("--near", type=float,
                        help="leave out Gaussians closer to the camera than this many metres. While "
                             "training, large ones drift past the camera and black out single frames")
    parser.add_argument("--seed", type=int, default=0, help="another one gives another training run")
    parser.add_argument("--noise-lr", type=float, default=5e5,
                        help="how hard MCMC shakes the Gaussians; gsplat's default")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    walk = json.loads(Path(args.path).read_text())
    if args.frames:
        first, last = (int(part) for part in args.frames.split(":"))
        walk["frames"] = walk["frames"][first:last + 1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(scene="walk", max_steps=args.steps, cap=args.cap)
    scene = load_colmap_scene("walk", args.scene_dir, args.gsplat_examples, cfg.max_size)

    # The path is in the workspace's own coordinates; the scene was normalised on loading.
    to_scene = np.linalg.inv(np.array(scene.to_world))
    scale = np.cbrt(np.linalg.det(to_scene[:3, :3]))
    viewmats = []
    for frame in walk["frames"]:
        camtoworld = np.eye(4)
        camtoworld[:3, :3] = to_scene[:3, :3] @ np.array(frame["R"]).reshape(3, 3) / scale
        camtoworld[:3, 3] = to_scene[:3, :3] @ np.array(frame["C"]) + to_scene[:3, 3]
        viewmats.append(np.linalg.inv(camtoworld))
    viewmats = torch.from_numpy(np.stack(viewmats)).float().to(DEVICE)
    shape = tuple(int(v) for v in args.shape.split("x")) if args.shape else (args.size, args.size)
    near = args.near * scale if args.near else 0.01  # in the scene's units; 0.01 is gsplat's default
    focal = min(shape) / 2 / math.tan(math.radians(args.fov or walk["fov"]) / 2)
    K = torch.tensor([[focal, 0, shape[0] / 2], [0, focal, shape[1] / 2], [0, 0, 1.0]], device=DEVICE)
    count, fps = len(viewmats), walk["fps"]
    # The learning videos: for each, the training step shown in every frame.
    eases = args.ease or ([] if args.keys else [2.2])
    span = args.steps - args.first_step
    shown_at = {}
    for ease in eases:
        name = "learning.mp4" if len(eases) == 1 and not args.keys else f"learning-ease{ease:g}.mp4"
        shown_at[name] = [round(args.first_step + span * (f / (count - 1)) ** ease) for f in range(count)]
    for spec in args.keys:
        label, _, pairs = spec.partition("=")
        at, steps = zip(*((int(a), int(b)) for a, b in (pair.split(":") for pair in pairs.split(","))))
        if list(steps) != sorted(steps) or list(at) != sorted(at) or steps[-1] > args.steps:
            raise SystemExit(f"--keys {label}: frames and steps must rise, up to --steps")
        shown_at[f"learning-{label}.mp4"] = [round(v) for v in np.interp(range(count), at, steps)]
    print(f"{count} frames ({count / fps:.0f} s), {len(scene.images)} photos, {args.steps} steps", flush=True)

    splats, optimizers = create_splats(scene, cfg)
    strategy = MCMCStrategy(cap_max=cfg.cap, refine_every=50, refine_stop_iter=int(cfg.max_steps * 0.85),
                            noise_lr=args.noise_lr)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps))
    train_viewmats = torch.linalg.inv(scene.camtoworlds)

    learning = {name: writer(out / name, shape, fps) for name in shown_at}
    frame = dict.fromkeys(shown_at, 0)
    order, started, recent = [], time.time(), []
    for step in range(cfg.max_steps + 1):
        for name, steps in shown_at.items():
            while frame[name] < count and steps[frame[name]] <= step:
                learning[name].send(render(splats, viewmats[frame[name]], K, shape, near))
                frame[name] += 1
        if step == cfg.max_steps:
            break
        if not order:
            order = torch.randperm(len(scene.images)).tolist()  # a film: train on every photo
        i = order.pop()
        pixels = scene.images[i][None].float() / 255.0
        height, width = pixels.shape[1:3]
        renders, _, info = rasterization(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=train_viewmats[i][None], Ks=scene.Ks[i][None],
            width=width, height=height, sh_degree=0, packed=False, absgrad=False,
        )
        strategy.step_pre_backward(params=splats, optimizers=optimizers, state=strategy_state,
                                   step=step, info=info)
        l1 = F.l1_loss(renders, pixels)
        ssim = 1.0 - fused_ssim(renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid")
        loss = l1 * 0.8 + ssim * 0.2
        loss = loss + 0.01 * torch.sigmoid(splats["opacities"]).mean() + 0.01 * torch.exp(splats["scales"]).mean()
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state,
                                    step=step, info=info, lr=scheduler.get_last_lr()[0])
        recent.append(loss.item())
        if (step + 1) % 1000 == 0:
            # The mean over the last 1000 steps: one photo's loss says little. On our scans a
            # run sometimes comes apart after the cap is reached, and this is where it shows.
            print(f"step {step + 1}  {len(splats['means'])} Gaussians  loss {np.mean(recent):.3f}  "
                  f"{time.time() - started:.0f} s", flush=True)
            recent = []
    for stream in learning.values():
        stream.close()

    final = writer(out / "final.mp4", shape, fps)
    for viewmat in viewmats:
        final.send(render(splats, viewmat, K, shape, near))
    final.close()

    # A photo is square and shows what the headset's camera sees: scale it to line up with the
    # renders, centred, cropped or bordered in black where the frame is wider or narrower.
    photos = writer(out / "photos.mp4", shape, fps)
    side = round(2 * focal * math.tan(math.radians(walk["fov"]) / 2))
    current, picture = -1, None
    for item in walk["frames"]:
        if item["photo"] != current:
            current = item["photo"]
            image = cv2.imread(str(Path(args.scene_dir) / "images" / walk["photos"][current]))
            if image is not None or picture is None:  # a missing photo: keep showing the last one
                if image is None:
                    image = np.zeros((side, side, 3), np.uint8)
                image = cv2.resize(image, (side, side), interpolation=cv2.INTER_CUBIC)
                canvas = np.zeros((shape[1], shape[0], 3), np.uint8)
                w, h = min(side, shape[0]), min(side, shape[1])
                x, y, u, v = (shape[0] - w) // 2, (shape[1] - h) // 2, (side - w) // 2, (side - h) // 2
                canvas[y:y + h, x:x + w] = image[v:v + h, u:u + w]
                picture = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes()
        photos.send(picture)
    photos.close()
    print(f"done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
