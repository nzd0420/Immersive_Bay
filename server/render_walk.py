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


def writer(path: Path, size: int, fps: int):
    stream = imageio_ffmpeg.write_frames(
        str(path), (size, size), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, output_params=["-crf", "17", "-preset", "medium"],
        macro_block_size=2,
    )
    stream.send(None)
    return stream


@torch.no_grad()
def render(splats, viewmat, K, size: int) -> bytes:
    image, _, _ = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=viewmat[None], Ks=K[None], width=size, height=size, sh_degree=0,
    )
    return (image[0].clamp(0, 1) * 255).round().byte().cpu().numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--gsplat-examples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--size", type=int, default=1440, help="videos are square")
    parser.add_argument("--cap", type=int, default=400_000,
                        help="Gaussian cap; nothing has to render this on a headset")
    parser.add_argument("--ease", type=float, default=2.2,
                        help="above 1, more of the walk is spent on the early, fast-changing steps")
    args = parser.parse_args()

    walk = json.loads(Path(args.path).read_text())
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
    focal = args.size / 2 / math.tan(math.radians(walk["fov"]) / 2)
    K = torch.tensor([[focal, 0, args.size / 2], [0, focal, args.size / 2], [0, 0, 1.0]], device=DEVICE)
    count, fps = len(viewmats), walk["fps"]
    # The training step shown in each frame of learning.mp4.
    shown_at = [round(args.steps * (f / (count - 1)) ** args.ease) for f in range(count)]
    print(f"{count} frames ({count / fps:.0f} s), {len(scene.images)} photos, {args.steps} steps", flush=True)

    splats, optimizers = create_splats(scene, cfg)
    strategy = MCMCStrategy(cap_max=cfg.cap, refine_every=50, refine_stop_iter=int(cfg.max_steps * 0.85))
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps))
    train_viewmats = torch.linalg.inv(scene.camtoworlds)

    learning = writer(out / "learning.mp4", args.size, fps)
    frame, order, started = 0, [], time.time()
    for step in range(cfg.max_steps + 1):
        while frame < count and shown_at[frame] <= step:
            learning.send(render(splats, viewmats[frame], K, args.size))
            frame += 1
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
        if (step + 1) % 1000 == 0:
            print(f"step {step + 1}  {len(splats['means'])} Gaussians  loss {loss.item():.3f}  "
                  f"{time.time() - started:.0f} s", flush=True)
    learning.close()

    final = writer(out / "final.mp4", args.size, fps)
    for viewmat in viewmats:
        final.send(render(splats, viewmat, K, args.size))
    final.close()

    photos = writer(out / "photos.mp4", args.size, fps)
    current, picture = -1, None
    for item in walk["frames"]:
        if item["photo"] != current:
            current = item["photo"]
            image = cv2.imread(str(Path(args.scene_dir) / "images" / walk["photos"][current]))
            if image is not None or picture is None:  # a missing photo: keep showing the last one
                if image is None:
                    image = np.zeros((args.size, args.size, 3), np.uint8)
                image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
                picture = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()
        photos.send(picture)
    photos.close()
    print(f"done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
