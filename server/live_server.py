#!/usr/bin/env python
"""Live 3D Gaussian Splatting trainer.

Trains a scene with gsplat and streams the Gaussians to viewers every few steps, so a
headset can watch the reconstruction form while it is being optimised.

    python live_server.py --port 8765 --scene succulent=/path/to/colmap/processed

Only the standard library, torch, gsplat, fused_ssim, numpy and OpenCV are needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

SH_C0 = 0.28209479177387814
DEVICE = "cuda"


# ----------------------------------------------------------------------------- data


@dataclass
class Scene:
    """A posed image set, held on the GPU for the whole run."""

    name: str
    images: list  # uint8 [H, W, 3] tensors on the GPU
    camtoworlds: torch.Tensor  # [N, 4, 4], OpenCV convention (x right, y down, z forward)
    Ks: torch.Tensor  # [N, 3, 3]
    points: torch.Tensor  # [M, 3] initial positions
    points_rgb: torch.Tensor  # [M, 3] in 0..1
    scene_scale: float
    train_idx: list
    val_idx: list
    center: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = 1.0
    up: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    view: list = field(default_factory=lambda: [0.0, -1.0, 0.5])  # mean camera position


def _load_images(paths, max_size):
    def load(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = 1.0
        longest = max(image.shape[:2])
        if max_size and longest > max_size:
            scale = max_size / longest
            size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image, scale

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(load, paths))


def _bounds(points: np.ndarray):
    center = np.median(points, axis=0)
    radius = float(np.percentile(np.linalg.norm(points - center, axis=1), 85))
    return center.tolist(), max(radius, 1e-3)


def load_colmap_scene(name: str, data_dir: str, examples_dir: str, max_size: int) -> Scene:
    """Load an undistorted COLMAP workspace (images/ + sparse/) with gsplat's parser."""
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from datasets.colmap import Parser  # gsplat/examples

    parser = Parser(data_dir=data_dir, factor=1, normalize=True, test_every=8)
    loaded = _load_images(parser.image_paths, max_size)
    images, Ks = [], []
    for (image, scale), camera_id in zip(loaded, parser.camera_ids):
        if len(parser.params_dict[camera_id]) > 0:
            raise ValueError("Images must be undistorted (PINHOLE cameras).")
        K = parser.Ks_dict[camera_id].copy()
        K[:2] *= scale
        images.append(torch.from_numpy(image).to(DEVICE))
        Ks.append(K)
    indices = list(range(len(images)))
    center, radius = _bounds(parser.points)
    return Scene(
        name=name,
        images=images,
        camtoworlds=torch.from_numpy(parser.camtoworlds).float().to(DEVICE),
        Ks=torch.from_numpy(np.stack(Ks)).float().to(DEVICE),
        points=torch.from_numpy(parser.points).float(),
        points_rgb=torch.from_numpy(parser.points_rgb / 255.0).float(),
        scene_scale=float(parser.scene_scale) * 1.1,
        train_idx=[i for i in indices if i % 8 != 0],
        val_idx=[i for i in indices if i % 8 == 0],
        center=center,
        radius=radius,
        up=[0.0, 0.0, 1.0],
        view=parser.camtoworlds[:, :3, 3].mean(axis=0).tolist(),
    )


# -------------------------------------------------------------------------- trainer


@dataclass
class TrainConfig:
    scene: str = ""
    max_steps: int = 7000
    strategy: str = "mcmc"  # "mcmc" keeps the Gaussian count under `cap`; "default" is 3DGS
    cap: int = 300_000
    sh_degree: int = 0  # the headset renders base colour only, so train what is shown
    stream_every: int = 20  # publish a frame to viewers every this many steps
    pace: bool = False  # wait for the viewer, so that every one of those frames is shown
    record: bool = True  # also write every one of those frames to a replay file
    ssim_lambda: float = 0.2
    max_size: int = 1280


def _knn_mean_dist(points: torch.Tensor) -> torch.Tensor:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(points.numpy())
    distances, _ = model.kneighbors(points.numpy())
    return torch.from_numpy(np.sqrt((distances[:, 1:] ** 2).mean(axis=-1))).float()


def create_splats(scene: Scene, cfg: TrainConfig):
    mcmc = cfg.strategy == "mcmc"
    points, rgbs = scene.points, scene.points_rgb
    dist = _knn_mean_dist(points).clamp_min(1e-7)
    scales = torch.log(dist * (0.1 if mcmc else 1.0)).unsqueeze(-1).repeat(1, 3)
    count = points.shape[0]
    colors = torch.zeros((count, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = (rgbs - 0.5) / SH_C0
    params = [
        ("means", points, 1.6e-4 * scene.scene_scale),
        ("scales", scales, 5e-3),
        ("quats", torch.rand((count, 4)), 1e-3),
        ("opacities", torch.logit(torch.full((count,), 0.5 if mcmc else 0.1)), 5e-2),
        ("sh0", colors[:, :1, :], 2.5e-3),
        ("shN", colors[:, 1:, :], 2.5e-3 / 20),
    ]
    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value) for name, value, _ in params}
    ).to(DEVICE)
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
        for name, _, lr in params
    }
    return splats, optimizers


@torch.no_grad()
def pack_frame(splats, min_alpha: float = 0.02) -> tuple[bytes, int]:
    """Encode the Gaussians in Spark's PackedSplats layout: 16 bytes each, ready to upload.

    Per Gaussian, little-endian: rgba (4 x u8), centre (3 x float16), rotation axis as
    octahedral u, v (2 x u8), log-scales (3 x u8), rotation angle (u8). This mirrors
    `setPackedSplat` in spark.module.js with its default encoding.
    """
    # Every Gaussian is sent, in training order, so that consecutive frames line up
    # index for index and can be sent as differences.
    opacity = torch.sigmoid(splats["opacities"])
    opacity = torch.where(opacity > min_alpha, opacity, torch.zeros_like(opacity))
    rgb = (splats["sh0"][:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    rgba = (torch.cat([rgb, opacity[:, None]], dim=-1) * 255).round().to(torch.uint8)
    centre = splats["means"].clamp(-6e4, 6e4).half().contiguous().view(torch.uint8)
    centre = centre.reshape(-1, 6)
    # log-scale range [-12, 9] mapped to 1..255 (0 is reserved for a flat axis)
    scale = ((splats["scales"] + 12.0) * (254.0 / 21.0)).round() + 1
    scale = scale.clamp(1, 255).to(torch.uint8)

    quat = F.normalize(splats["quats"], dim=-1)  # w, x, y, z
    quat = torch.where(quat[:, :1] < 0, -quat, quat)
    theta = 2 * torch.acos(quat[:, 0].clamp(-1, 1))
    xyz = quat[:, 1:]
    norm = xyz.norm(dim=-1, keepdim=True)
    unit_x = torch.tensor([1.0, 0.0, 0.0], device=xyz.device).expand_as(xyz)
    axis = torch.where(norm < 1e-6, unit_x, xyz / norm.clamp_min(1e-12))
    p = axis[:, :2] / axis.abs().sum(-1, keepdim=True)
    sign = torch.where(p >= 0, 1.0, -1.0)
    folded = (1 - p.flip(-1).abs()) * sign
    p = torch.where(axis[:, 2:3] < 0, folded, p)
    uv = ((p * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
    angle = (theta * (255.0 / math.pi)).round().clamp(0, 255).to(torch.uint8)

    packed = torch.cat([rgba, centre, uv, scale, angle[:, None]], dim=-1)  # [N, 16]
    return packed.cpu().numpy().tobytes(), int(packed.shape[0])


def message(header: dict, payload: bytes = b"") -> bytes:
    """One stream message: `LSPL`, a u32 header length, a JSON header, then the payload."""
    head = json.dumps(header).encode()
    return b"LSPL" + len(head).to_bytes(4, "little") + head + payload


def delta_payload(current: np.ndarray, base: np.ndarray, level: int = 1) -> bytes:
    """The change from `base` to `current` ([count, 16] uint8 arrays), compressed.

    Rows missing from `base` count as zero. The byte-wise difference is stored column by
    column and deflated, which is about a third of the size of the raw frame.
    """
    previous = np.zeros_like(current)
    shared = min(len(current), len(base))
    previous[:shared] = base[:shared]
    planes = np.ascontiguousarray((current - previous).T)
    return zlib.compress(planes.tobytes(), level)


class Recorder:
    """Writes a run to `<name>.lspl`: the same messages a live viewer receives, with every
    published frame included, so the file can be played back at any speed later."""

    def __init__(self, directory: Path, name: str, intro: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.final_path = directory / f"{name}.lspl"
        self.part_path = directory / f"{name}.lspl.part"
        self.frames = 0
        self.queue: queue.Queue = queue.Queue()
        self.file = open(self.part_path, "wb")
        self.file.write(message({"type": "intro", "bytes": 0, **intro}))
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def add(self, frame):
        self.queue.put(frame)

    def close(self, keep: bool):
        self.queue.put(None)
        self.thread.join()
        self.file.close()
        if keep and self.frames > 1:
            self.part_path.rename(self.final_path)
        else:
            self.part_path.unlink(missing_ok=True)

    def _work(self):
        base = np.zeros((0, 16), np.uint8)
        while (frame := self.queue.get()) is not None:
            current = np.frombuffer(frame.data, np.uint8).reshape(-1, 16)
            payload = delta_payload(current, base)
            header = {"seq": frame.seq, "step": frame.step, "gaussians": frame.count,
                      "psnr": frame.psnr, "loss": frame.loss, "elapsed": frame.elapsed,
                      "reset": len(base) == 0, "bytes": len(payload)}
            self.file.write(message(header, payload))
            base = current
            self.frames += 1


@dataclass
class Frame:
    seq: int
    step: int
    data: bytes
    count: int
    psnr: float
    loss: float
    elapsed: float


class Session:
    """One training run. `Session.current` is what the HTTP handlers talk to."""

    def __init__(self, cfg: TrainConfig, loader, run: int = 1, viewers=lambda: 0,
                 runs_dir: Path | None = None):
        self.cfg = cfg
        self.runs_dir = runs_dir
        self.recorder: Recorder | None = None
        self.recording = ""  # name of the replay file, once the run has finished
        self.loader = loader
        self.viewers = viewers  # how many viewers are connected right now
        self.run = run
        self.state = "loading"  # loading -> training -> done | stopped | error
        self.error = ""
        self.step = 0
        self.count = 0
        self.loss = 0.0
        self.steps_per_sec = 0.0
        self.started = time.time()
        self.finished = None
        self.scene_info = {}
        self.frame: Frame | None = None  # newest frame; viewers that fall behind skip ahead
        self.history: list[dict] = []
        self.cond = threading.Condition()
        self.want_frame = True
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ---- called from HTTP threads

    def status(self):
        return {
            "state": self.state,
            "run": self.run,
            "error": self.error,
            "config": asdict(self.cfg),
            "step": self.step,
            "max_steps": self.cfg.max_steps,
            "gaussians": self.count,
            "loss": self.loss,
            "steps_per_sec": self.steps_per_sec,
            "elapsed": (self.finished or time.time()) - self.started,
            "scene": self.scene_info,
            "recording": self.recording,
            "history": self.history[-400:],
        }

    def wait_frame(self, after: int, timeout: float):
        """Return the newest frame once its seq is past `after`, else None on timeout."""
        deadline = time.time() + timeout
        with self.cond:
            while True:
                if self.frame and self.frame.seq > after:
                    return self.frame
                if not self.want_frame:
                    self.want_frame = True
                    self.cond.notify_all()  # a paced trainer may be waiting for this
                remaining = deadline - time.time()
                if remaining <= 0 or self.state in ("done", "stopped", "error"):
                    return None
                self.cond.wait(remaining)

    def stop(self):
        self.stop_flag = True

    # ---- training thread

    def _publish(self, step, splats, loss, psnr, live: bool):
        """Pack the current Gaussians; record them, and offer them to live viewers."""
        data, count = pack_frame(splats)
        elapsed = time.time() - self.started
        with self.cond:
            seq = len(self.history) + 1
            frame = Frame(seq, step, data, count, psnr, loss, elapsed)
            self.history.append({"step": step, "gaussians": count, "psnr": psnr,
                                 "loss": loss, "elapsed": elapsed})
            if live:
                self.frame = frame
                self.want_frame = False
                self.cond.notify_all()
        if self.recorder:
            self.recorder.add(frame)

    @torch.no_grad()
    def _psnr(self, scene: Scene, splats, viewmats) -> float:
        values = []
        for i in scene.val_idx[:: max(1, len(scene.val_idx) // 2)][:2]:
            pixels = scene.images[i].float() / 255.0
            height, width = pixels.shape[:2]
            render, _, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                viewmats=viewmats[i][None], Ks=scene.Ks[i][None],
                width=width, height=height, sh_degree=self.cfg.sh_degree,
            )
            mse = F.mse_loss(render[0].clamp(0, 1), pixels)
            values.append(-10.0 * math.log10(max(mse.item(), 1e-10)))
        return float(np.mean(values)) if values else 0.0

    def _run(self):
        try:
            self._train()
        except Exception:
            self.error = traceback.format_exc()
            self.state = "error"
            print(self.error, flush=True)
        finally:
            self.finished = time.time()
            if self.recorder:
                self.recorder.close(keep=self.state == "done")
                if self.state == "done":
                    self.recording = self.recorder.name
            with self.cond:
                self.cond.notify_all()
            torch.cuda.empty_cache()

    def _train(self):
        cfg = self.cfg
        torch.manual_seed(42)
        scene: Scene = self.loader(cfg)
        self.scene_info = {
            "name": scene.name, "images": len(scene.images),
            "center": scene.center, "radius": scene.radius, "up": scene.up,
            "view": scene.view,
        }
        splats, optimizers = create_splats(scene, cfg)
        mcmc = cfg.strategy == "mcmc"
        if mcmc:
            # Grow every 50 steps so the cap is reached well before the end of a short run.
            strategy = MCMCStrategy(cap_max=cfg.cap, refine_every=50,
                                    refine_stop_iter=int(cfg.max_steps * 0.85))
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state()
        else:
            strategy = DefaultStrategy(refine_stop_iter=int(cfg.max_steps * 0.75),
                                       reset_every=3000)
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
        )
        viewmats = torch.linalg.inv(scene.camtoworlds)

        self.started = time.time()
        self.state = "training"
        self.count = len(splats["means"])
        if cfg.record and self.runs_dir:
            name = f"{scene.name}-{time.strftime('%m%d-%H%M%S')}"
            self.recorder = Recorder(self.runs_dir, name,
                                     {"scene": self.scene_info, "config": asdict(cfg)})
        psnr = self._psnr(scene, splats, viewmats)
        self._publish(0, splats, 0.0, psnr, live=True)

        loss_ema = None
        tick, tick_step = time.time(), 0
        order = []
        for step in range(cfg.max_steps):
            if self.stop_flag:
                self.state = "stopped"
                return
            if not order:
                order = [scene.train_idx[j] for j in torch.randperm(len(scene.train_idx))]
            i = order.pop()
            pixels = scene.images[i][None].float() / 255.0
            height, width = pixels.shape[1:3]
            sh_degree = min(step // 1000, cfg.sh_degree)

            renders, alphas, info = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                viewmats=viewmats[i][None], Ks=scene.Ks[i][None],
                width=width, height=height, sh_degree=sh_degree,
                packed=False, absgrad=False,
            )
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=step, info=info,
            )
            l1 = F.l1_loss(renders, pixels)
            ssim = 1.0 - fused_ssim(
                renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1 * (1.0 - cfg.ssim_lambda) + ssim * cfg.ssim_lambda
            if mcmc:
                loss = loss + 0.01 * torch.sigmoid(splats["opacities"]).mean()
                loss = loss + 0.01 * torch.exp(splats["scales"]).mean()
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if mcmc:
                strategy.step_post_backward(
                    params=splats, optimizers=optimizers, state=strategy_state,
                    step=step, info=info, lr=scheduler.get_last_lr()[0],
                )
            else:
                strategy.step_post_backward(
                    params=splats, optimizers=optimizers, state=strategy_state,
                    step=step, info=info, packed=False,
                )

            done = step + 1
            if done % 10 == 0:
                value = loss.item()
                loss_ema = value if loss_ema is None else 0.9 * loss_ema + 0.1 * value
                self.loss = loss_ema
                self.step = done
                self.count = len(splats["means"])
            if done % 100 == 0:
                now = time.time()
                self.steps_per_sec = (done - tick_step) / max(now - tick, 1e-6)
                tick, tick_step = now, done
            final = done == cfg.max_steps
            boundary = done % cfg.stream_every == 0
            if boundary and cfg.pace and not final:
                # Hold training until a viewer is ready for this frame (at most 5 s).
                deadline = time.time() + 5.0
                with self.cond:
                    while (not self.want_frame and self.viewers() > 0
                           and not self.stop_flag and time.time() < deadline):
                        self.cond.wait(0.1)
            live = final or (self.want_frame and boundary)
            if live or (boundary and self.recorder):
                self.step = done
                self.count = len(splats["means"])
                if final or done % 100 == 0:  # two extra renders, so not on every frame
                    psnr = self._psnr(scene, splats, viewmats)
                self._publish(done, splats, self.loss, psnr, live)
        self.state = "done"


# ----------------------------------------------------------------------------- http


class App:
    def __init__(self, scenes: dict, examples_dir: str, static_dir: str, runs_dir: str):
        self.runs_dir = Path(runs_dir).resolve()
        self.scenes = scenes
        self.examples_dir = examples_dir
        self.static_dir = Path(static_dir).resolve() if static_dir else None
        self.session: Session | None = None
        self.runs = 0
        self.lock = threading.Lock()
        # Flow control for viewers: each one acknowledges the frames it has drawn.
        self.acks: dict = {}  # viewer id -> (run, seq) last acknowledged
        self.viewers = 0  # open streams
        self.ack_cond = threading.Condition()
        self.scene_cache: dict = {}

    def _load(self, cfg: TrainConfig) -> Scene:
        key = (cfg.scene, cfg.max_size)
        if key not in self.scene_cache:
            self.scene_cache.clear()  # one scene on the GPU at a time
            torch.cuda.empty_cache()
            self.scene_cache[key] = load_colmap_scene(
                cfg.scene, self.scenes[cfg.scene], self.examples_dir, cfg.max_size
            )
        return self.scene_cache[key]

    def recordings(self) -> list:
        """Finished replay files, newest first."""
        files = sorted(self.runs_dir.glob("*.lspl"), key=lambda f: f.stat().st_mtime,
                       reverse=True)
        return [{"name": f.stem, "bytes": f.stat().st_size} for f in files]

    def start(self, options: dict) -> Session:
        with self.lock:
            if self.session is not None:
                self.session.stop()
                self.session.thread.join(timeout=30)
            known = {k: v for k, v in options.items() if k in TrainConfig.__dataclass_fields__}
            cfg = TrainConfig(**known)
            if not cfg.scene:
                cfg.scene = next(iter(self.scenes))
            if cfg.scene not in self.scenes:
                raise KeyError(f"Unknown scene {cfg.scene!r}; have {list(self.scenes)}")
            self.runs += 1
            self.session = Session(cfg, self._load, self.runs, lambda: self.viewers,
                                   self.runs_dir)
            self.session.thread.start()
            return self.session


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body=b"", content_type="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code=200):
            self._send(code, json.dumps(payload).encode())

        def do_OPTIONS(self):
            self._send(204, headers={"Access-Control-Allow-Headers": "Content-Type",
                                     "Access-Control-Allow-Methods": "GET, POST"})

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            session = app.session
            if url.path == "/api/status":
                payload = session.status() if session else {"state": "idle"}
                payload["scenes"] = list(app.scenes)
                return self._json(payload)
            if url.path == "/api/recordings":
                return self._json(app.recordings())
            if url.path.startswith("/api/recording/"):
                return self._recording(url.path.rsplit("/", 1)[1])
            if url.path == "/api/stream":
                return self._stream(query.get("id", [""])[0] or str(id(self)))
            return self._static(url.path)

        def _stream(self, viewer):
            """Push frames for as long as the viewer stays connected.

            Each message is `LSPL`, a u32 header length, a JSON header, then `bytes` of
            payload. A header with no payload is a heartbeat.

            The payload is the change since the frame this viewer last received: the
            byte-wise difference of the two [count, 16] arrays (missing rows count as
            zero), stored column by column and deflated. That is about a third of the
            size of the raw frame, which matters on a slow link.

            The viewer acknowledges each frame it has drawn. Frames are only sent while
            the unacknowledged bytes fit in a window, and the window is steered so that
            a frame is on screen about a second after it was made: a slow link gets
            fewer, newer frames rather than a growing backlog.
            """
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            run, seq = 0, 0
            base = np.zeros((0, 16), np.uint8)
            flight = []  # [(run, seq), bytes, time sent] for frames not yet acknowledged
            window = 600_000  # bytes allowed in flight
            with app.ack_cond:
                app.acks[viewer] = (0, 0)
                app.viewers += 1

            def settle():
                """Drop acknowledged frames from `flight` and adapt the window."""
                nonlocal flight, window
                acked = app.acks[viewer]
                now = time.time()
                for mark, _, sent_at in flight:
                    if mark <= acked:
                        delay = now - sent_at
                        if delay < 1.2:
                            window = min(window * 1.15, 16_000_000)
                        elif delay > 2.0:
                            window = max(window * 0.8, 150_000)
                flight = [item for item in flight if item[0] > acked]
                return len(flight) >= 2 and sum(item[1] for item in flight) >= window

            try:
                while True:
                    with app.ack_cond:
                        full = settle()
                        if full:
                            app.ack_cond.wait(1.0)
                            full = settle()
                    if full:
                        self._heartbeat(run)
                        continue
                    session = app.session
                    if session is None:
                        time.sleep(0.3)
                        frame = None
                    else:
                        if session.run != run:
                            run, seq = session.run, 0
                            base = np.zeros((0, 16), np.uint8)
                            flight = []
                        frame = session.wait_frame(seq, 1.0)
                        if frame is None and session.state != "training":
                            time.sleep(0.3)
                    if frame is None:
                        self._heartbeat(run)
                        continue
                    seq = frame.seq
                    current = np.frombuffer(frame.data, np.uint8).reshape(-1, 16)
                    payload = delta_payload(current, base)
                    header = {"run": run, "seq": seq, "step": frame.step,
                              "gaussians": frame.count, "psnr": frame.psnr,
                              "loss": frame.loss, "elapsed": frame.elapsed,
                              "reset": len(base) == 0, "bytes": len(payload)}
                    base = current
                    self.wfile.write(message(header, payload))
                    self.wfile.flush()
                    flight.append(((run, seq), len(payload), time.time()))
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with app.ack_cond:
                    app.viewers -= 1

        def _recording(self, name):
            """Send a replay file; `Range: bytes=N-` resumes an interrupted download."""
            path = (app.runs_dir / f"{name}.lspl").resolve()
            if path.parent != app.runs_dir or not path.is_file():
                return self._send(404)
            size = path.stat().st_size
            start = 0
            requested = self.headers.get("Range", "")
            if requested.startswith("bytes=") and requested.endswith("-"):
                start = min(int(requested[6:-1]), size)
            self.send_response(206 if start else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size - start))
            self.send_header("X-Total-Bytes", str(size))
            self.end_headers()
            with open(path, "rb") as file:
                file.seek(start)
                try:
                    while chunk := file.read(1 << 16):
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        def _heartbeat(self, run):
            self.wfile.write(message({"run": run, "bytes": 0}))
            self.wfile.flush()

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                if url.path == "/api/ack":
                    query = parse_qs(url.query)
                    viewer = query["id"][0]
                    mark = (int(query["run"][0]), int(query["seq"][0]))
                    with app.ack_cond:
                        app.acks[viewer] = max(app.acks.get(viewer, (0, 0)), mark)
                        app.ack_cond.notify_all()
                    return self._json({"ok": True})
                options = json.loads(raw) if raw else {}
                if url.path == "/api/start":
                    session = app.start(options)
                    return self._json({"ok": True, "run": session.run,
                                       "config": asdict(session.cfg)})
                if url.path == "/api/stop":
                    if app.session:
                        app.session.stop()
                    return self._json({"ok": True})
            except Exception as error:  # report bad requests to the caller
                return self._json({"ok": False, "error": str(error)}, 400)
            self._send(404)

        def _static(self, path):
            if app.static_dir is None:
                return self._send(404)
            target = (app.static_dir / path.lstrip("/")).resolve()
            if target.is_dir():
                target = target / "index.html"
            if app.static_dir not in target.parents or not target.is_file():
                return self._send(404)
            self._send(200, target.read_bytes(),
                       CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", action="append", default=[], metavar="NAME=DIR",
                        help="COLMAP workspace with images/ and sparse/; repeatable")
    parser.add_argument("--gsplat-examples", required=True,
                        help="path to gsplat's examples/ directory (for its COLMAP parser)")
    parser.add_argument("--static", default="", help="directory of viewer files to serve")
    parser.add_argument("--runs", default="runs", help="directory for replay files")
    args = parser.parse_args()
    scenes = dict(item.split("=", 1) for item in args.scene)
    if not scenes:
        parser.error("give at least one --scene NAME=DIR")
    app = App(scenes, args.gsplat_examples, args.static, args.runs)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"Serving on http://{args.host}:{args.port}  scenes: {list(scenes)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
