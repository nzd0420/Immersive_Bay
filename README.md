# Live Splat

English | [简体中文](README.zh-CN.md)

Watch a 3D Gaussian Splatting reconstruction form in front of you.

A scene is trained on a GPU server while a Meta Quest shows the Gaussians sharpening, every
20 training steps, from a sparse point cloud into the finished scene: a hologram floating
in passthrough AR. When the run ends it plays back in a loop, so the whole thirty seconds
of "learning" can be watched again at a steady frame rate.

Built at [Immerse the Bay 2026](https://immersethebay.org/), Stanford XR's hackathon.

## Why

3D Gaussian Splatting is usually a black box: you upload photos, wait, and get a scene.
The optimisation itself is the interesting part. Tens of thousands of blurry blobs drift,
split and sharpen until they agree with every photo at once. Live Splat puts that process
at room scale, where you can walk around it while it happens.

## How it works

```text
Quest browser (WebXR)  ──USB──►  Mac: scripts/dev_proxy.py :8090  ──SSH──►  GPU: server/live_server.py :8765
  viewer/ (three.js + Spark)       serves viewer/ and replays,              trains with gsplat,
                                   forwards /api                            streams and records frames
```

**Training.** `server/live_server.py` runs a compact [gsplat](https://github.com/nerfstudio-project/gsplat)
training loop. The MCMC densification strategy keeps the number of Gaussians under a cap
(300,000 by default), which is what a standalone headset can render.

**Streaming.** Every 20 steps the server packs all Gaussians on the GPU into the
16-byte-per-Gaussian layout that the [Spark](https://sparkjs.dev/) renderer uses
internally, so the viewer uploads it without parsing. What goes over the wire is the
difference from the previous frame, column by column and deflated: about a third of the
raw size.

**Staying live on a slow link.** The viewer acknowledges each frame it has drawn. The
server lets about one second of frames be in flight and adapts that window to the link,
so a slow connection shows fewer, newer frames instead of falling behind the training.

**Replay.** Every run is also recorded on the GPU machine with all of its frames. The Mac
proxy copies finished recordings into `recordings/` in the background (pausing while a
run is training) and the viewer replays them locally at 15 frames per second. Replays do
not need the GPU machine at all, so they double as an offline fallback for a demo.

**Viewing.** `viewer/` is a single WebXR page. On a desktop you orbit with the mouse. In
the headset the scene appears at table height in passthrough; grip moves it, both grips
resize it, and the trigger restarts training.

## Measured

On an RTX 3080 Ti, with a 100-image test scene at 1280×717 and a link of roughly
1.2 MB/s between the viewer and the GPU machine:

| | |
|---|---|
| Training speed | about 100 steps per second; 3000 steps in 29 s |
| Quality after 3000 steps | about 29 dB PSNR on held-out views |
| Frame size on the wire | 30–43% of the raw frame |
| Live updates, first 1000 steps | 5 per second (every 20 steps) |
| Live updates, later | 1.5–2 per second, limited by the link |
| Recording of a 3000-step run | 151 frames, 69 MB |
| Replay | steady 15 frames per second |

## Run it

You need a machine with an NVIDIA GPU that has PyTorch, gsplat 1.5 and `fused-ssim`
installed, plus gsplat's `examples/` folder (for its COLMAP parser), and a scene as an
undistorted COLMAP workspace (`images/` and `sparse/`).

On the GPU machine:

```bash
python server/live_server.py --port 8765 \
  --scene myscene=/path/to/colmap/workspace \
  --gsplat-examples /path/to/gsplat/examples
```

On the Mac, in two terminals:

```bash
scripts/tunnel.sh                # Mac :8765 -> GPU :8765
```

```bash
python3 scripts/dev_proxy.py     # viewer on http://localhost:8090
```

`scripts/deploy.sh` copies `server/` to our GPU machine and starts it; its paths are
specific to our setup. `scripts/tunnel.sh` reuses an SSH connection that is already open,
because that machine uses password login.

On the headset, with developer mode on and a USB cable connected:

```bash
adb reverse tcp:8090 tcp:8090
```

Then open `http://localhost:8090` in the Quest browser and press **Enter AR**.
`localhost` counts as a secure context, so WebXR works without certificates.

## Repository layout

```text
server/live_server.py   training loop, frame packing, stream, recorder, HTTP API
viewer/index.html       control panel
viewer/main.js          stream reader, in-place mesh updates, replay, WebXR controls
viewer/vendor/          three.js r180 and Spark 2.4.0, vendored so the headset needs no CDN
scripts/dev_proxy.py    serves the viewer, forwards /api, copies and serves recordings
scripts/tunnel.sh       SSH port forward to the GPU machine
scripts/deploy.sh       copy and restart the server on our GPU machine
```

## API

| Route | What it does |
|---|---|
| `GET /api/status` | State, step, Gaussian count, loss, and per-frame history |
| `POST /api/start` | `{"scene": "...", "max_steps": 7000, "cap": 300000, "pace": false}` |
| `POST /api/stop` | Stop the current run |
| `GET /api/stream?id=V` | Frame stream for viewer `V` |
| `POST /api/ack?id=V&run=R&seq=N` | Viewer `V` has drawn frame `N` of run `R` |
| `GET /api/recordings` | Finished recordings (the Mac proxy adds copy progress) |
| `GET /api/recording/NAME` | Download a recording from the GPU machine (supports `Range`) |
| `GET /recordings/NAME.lspl` | Mac proxy only: a local recording, for playback |

A stream, and a recording file, is a sequence of messages: the bytes `LSPL`, a
little-endian u32 header length, a JSON header, then `bytes` of payload. A payload holds
16 columns of `gaussians` bytes each, deflated; adding them (mod 256) to the previous
frame's rows gives the new frame. `"pace": true` makes training wait for the viewer so
that every 20th step is shown live, at the cost of a run about three times longer.

## Status

Working, checked in a desktop browser:

- training server, live frame stream, flow control
- recording, background copy to the Mac, looping replay

Not done yet:

- **Headset test.** The WebXR path (AR session, grab, resize, trigger) is written but has
  not been run on the Quest.
- **Training from a Quest scan.** The plan is to capture with the headset's passthrough
  cameras and use the headset's own tracking for camera poses, which removes the COLMAP
  step. Today the server only reads COLMAP workspaces.
- Recordings of long runs are large. A 7000-step run has not been measured.

## Credits

- [gsplat](https://github.com/nerfstudio-project/gsplat) for differentiable Gaussian rasterisation and densification
- [Spark](https://github.com/sparkjsdev/spark) for Gaussian splat rendering in three.js and WebXR
- [three.js](https://threejs.org/) (MIT)

All code in this repository was written during the hackathon. The gsplat environment on
the GPU machine and the three test scenes used for the measurements above existed before
it; neither is part of this repository.

## License

[MIT](LICENSE). The vendored three.js and Spark in `viewer/vendor/` keep their own MIT licenses.
