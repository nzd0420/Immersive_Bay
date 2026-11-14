# Live Splat

Watch a 3D Gaussian Splatting reconstruction form in front of you. A scene is trained on a
GPU server while a Meta Quest shows the Gaussians sharpening from a sparse point cloud into
the finished scene, as a hologram in passthrough AR.

Built at Immerse the Bay 2026.

## How it fits together

```text
Quest browser (WebXR)  ──USB──►  Mac: scripts/dev_proxy.py :8090  ──SSH──►  GPU: server/live_server.py :8765
   viewer/ (three.js + Spark)        serves viewer/, forwards /api          trains with gsplat, publishes snapshots
```

- `server/live_server.py` trains with [gsplat](https://github.com/nerfstudio-project/gsplat)
  and pushes a frame to each viewer every 20 steps over one streaming connection. The MCMC
  strategy keeps the Gaussian count under a cap (300k by default) so the headset can
  render it.
- A frame is sent in Spark's own 16-byte-per-Gaussian layout, as the difference from the
  previous frame, deflated: about a third of the raw size. Viewers acknowledge frames, and
  the server keeps only about a second of frames in flight, so a slow link shows fewer,
  newer frames instead of falling behind.
- `viewer/` writes each frame straight into one Spark mesh. Desktop: orbit with the mouse.
  Headset: grip moves the hologram, both grips resize it, trigger restarts training.
- Every run is also recorded on the GPU machine with all of its frames (every 20 steps).
  The Mac proxy copies finished recordings into `recordings/` in the background and the
  viewer replays them from there at 15 frames per second, in a loop. Replays do not need
  the GPU machine, so they also work as an offline fallback for a demo.
- `scripts/` holds the glue: deploy to the GPU machine, SSH tunnel, local proxy.

## Run it

```bash
scripts/deploy.sh                # copy server/ to the GPU machine and start it
scripts/tunnel.sh                # Mac :8765 -> GPU :8765 (keep running)
python3 scripts/dev_proxy.py     # viewer on http://localhost:8090
```

The GPU machine uses password login, so `tunnel.sh` reuses an already-open `ssh autodl_3080TI`
connection. If it reports that there is none, run `ssh autodl_3080TI` in a terminal first.

On the headset (developer mode on, USB cable connected):

```bash
adb reverse tcp:8090 tcp:8090
```

Then open `http://localhost:8090` in the Quest browser and press **Enter AR**.

## API

| Route | What it does |
|---|---|
| `GET /api/status` | State, step, Gaussian count, loss, and per-frame history |
| `GET /api/stream?id=V` | Frame stream for viewer `V` (format in `_stream`'s docstring) |
| `POST /api/ack?id=V&run=R&seq=N` | Viewer `V` has drawn frame `N` of run `R` |
| `POST /api/start` | `{"scene": "...", "max_steps": 7000, "cap": 300000, "pace": false}` |
| `POST /api/stop` | Stop the current run |
| `GET /api/recordings` | Finished recordings (the Mac proxy adds copy progress) |
| `GET /api/recording/NAME` | Download a recording from the GPU machine (supports `Range`) |
| `GET /recordings/NAME.lspl` | Mac proxy only: a local recording, for playback |

## Status

- Working: training server, frame stream, desktop viewer (3000 steps in about 30 s on an
  RTX 3080 Ti, held-out PSNR about 29 dB on a 100-image test scene).
- Update rate depends on the link to the GPU machine. Measured at about 1.2 MB/s: 5 frames
  per second for the first 1000 steps, 1.5 to 2 per second once frames grow. With
  `"pace": true` training waits for the viewer, so every 20th step is shown, at the cost
  of a run about three times longer.
- Replay: a 3000-step run records 151 frames in 69 MB, copied to the Mac in about half a
  minute, then played back at a steady 15 frames per second. Longer runs with more
  Gaussians make larger files; a 7000-step run has not been measured yet.
- Not yet tested on the headset.
- Not yet built: training from Quest captures (poses from the headset instead of COLMAP).
