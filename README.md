# Darkroom

English | [简体中文](README.zh-CN.md)

Scan the room you are standing in with a Meta Quest, then watch a 3D Gaussian Splatting
reconstruction of it take shape around you, in place and at real size.

You pull the trigger and look around. The headset photographs through its passthrough
cameras and remembers where your head was for each photo. A GPU server starts training,
and every 20 steps the Gaussians are sent back to the headset: a cloud of blurry blobs
appears where the real table, chairs and walls are, and sharpens into them. The real room
fades to black, so what you are standing in is only what the model has learned so far.
Press B and the real room is back, in the same place, to compare.

Everything on the headset is one web page. There is no app to install and no cable.

Made by Yann and Zidong at [Immerse the Bay 2026](https://immersethebay.org/), Stanford
XR's hackathon, in 36 hours.

## Why

3D Gaussian Splatting is usually a black box: you upload photos, wait, and get a scene.
The optimisation itself is the interesting part. Tens of thousands of blurry blobs drift,
split and sharpen until they agree with every photo at once. Darkroom puts that process
at room scale, around the person who scanned it, while it happens.

## How it works

```text
Quest browser (WebXR page)  ──►  a computer: scripts/dev_proxy.py  ──SSH──►  GPU machine: server/live_server.py
  scans, shows the result        builds the training scene,                 trains with gsplat,
                                 forwards /api                              streams the Gaussians
```

**Scanning.** Inside the AR session the page reads both passthrough cameras with
`getUserMedia` (1728×1728, 30 frames per second) and takes a photo with each of them
every half second or so. Every AR frame it also records the head pose, about 90 times a
second. Photos are skipped while the head turns fast or while tracking is paused.

**From photos to a training scene, without COLMAP.** `scripts/webxr_scan.py` gives each
photo its camera pose: the head pose interpolated to the photo's timestamp, combined with
that camera's lens position and intrinsics, which the headset itself reports
(`scripts/quest_cameras.json`). The result is written as a COLMAP workspace and uploaded.
No structure-from-motion step runs at any point.

**In place, with nothing to line up.** The poses come from the same AR session that later
displays the result, so the trained Gaussians are already in the viewer's coordinates.
They appear where the real things are, at real size.

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

[docs/pipeline.md](docs/pipeline.md) goes through each stage in detail.

## In the headset

| Control | What it does |
|---|---|
| Trigger | Start a scan; pull again to finish and train |
| B / Y | Switch between the real room and the reconstruction, with a fade through black |
| A / X | Switch between room scale and a table-top hologram |
| Thumbstick press | Reset: stop training, clear everything, ready for a new scan |
| Grip | Slide the scene (room scale) or move the hologram |
| Both grips | Turn the scene (room scale) or resize the hologram |

While scanning, a readout in front of the eyes counts the photos and small markers show
where each one was taken. When the first frame of the reconstruction arrives the real
room is hidden, and from then on the two are never shown on top of each other. The room
is hidden by a black sphere around the viewer, because three.js always clears to
transparent in an AR session.

The same page works in a desktop browser: orbit with the mouse, or follow the path that
was walked during the scan in first person.

## Measured

Training on an RTX 3080 Ti. The GPU machine was on another continent, on a link of roughly
1.2 MB/s.

| | |
|---|---|
| Scan | 75 s of walking gives about 105 photos; 2 lacked a head pose |
| From finishing a scan to the first Gaussians | roughly half a minute, mostly upload (not timed precisely) |
| Training speed | 50 to 100 steps per second, depending on the Gaussian count |
| A scan of 138 photos, 15,000 steps | 26.1 dB PSNR on views held out from training |
| A walked room scan (about 4.7 × 2.5 m) | 23.6 dB after 10,000 steps; 18.9 dB after 1,500 |
| Frame size on the wire | 30–43% of the raw frame |
| Live updates, first 1000 steps | 5 per second (every 20 steps) |
| Live updates, later | 1.5–2 per second, limited by the link |

The delay between a photo's timestamp and the head pose it belongs to was measured with
`scripts/calibrate_scan.py`: held-out PSNR is flat for an offset of 0 to 30 ms and falls
beyond that (17.8 dB at 60 ms, 16.9 dB at 120 ms, against 18.9 dB at 0).

## Run it

Three machines are involved: the headset, a computer next to you, and a machine with an
NVIDIA GPU that the computer can reach over SSH.

**On the GPU machine**, make a project folder that contains

- `activate.sh`, a script that activates a Python environment with PyTorch, gsplat 1.5
  and `fused-ssim`;
- `gsplat/examples`, gsplat's examples folder (for its COLMAP parser).

**On the computer**, say where that is:

```bash
cp scripts/local.env.example scripts/local.env     # then set GPU_HOST and GPU_DIR in it
```

`scripts/local.env` is not tracked. Then, in three terminals:

```bash
scripts/deploy.sh                # copy server/ to the GPU machine and start it
```

```bash
scripts/tunnel.sh                # local port 8765 -> the training server
```

```bash
python3 scripts/dev_proxy.py     # the viewer, on http://localhost:8090
```

**The headset** needs the page from a secure origin. Either connect it by USB, with
developer mode on:

```bash
adb reverse tcp:8090 tcp:8090
```

and open `http://localhost:8090` in the Quest browser, or run it untethered through a
public HTTPS tunnel:

```bash
cloudflared tunnel --url http://localhost:8090
```

and open the address it prints. Anyone who has that address can use the page, so close
the tunnel when you are done. On the page press **Enable scanning** and allow the camera,
then **Enter AR**.

Set the headset's boundary to room scale and draw it generously. With the small
stationary boundary, stepping outside pauses the AR session, and the photos taken during
the pause have no head pose.

`scripts/quest_cameras.json` holds the lens calibration of our headset. To read another
one's:

```bash
adb shell dumpsys media.camera
```

and copy `android.lens.intrinsicCalibration`, `poseTranslation` and `poseRotation` of
cameras 50 and 51 into it.

### Other ways to get a scene

Any undistorted COLMAP workspace (`images/` and `sparse/`) can be trained and watched:
put it in the `scenes` folder on the GPU machine, or name it in `EXTRA_SCENES`. Scenes
that were not scanned in the session are shown as a table-top hologram, or at room scale
and lined up by hand.

A scene can also be recorded with [OpenQuestCapture](https://github.com/samuelm2/OpenQuestCapture)
on a Quest 3 or 3S and converted on a computer with Python, `numpy` and `opencv-python`:

```bash
scripts/pull_capture.sh                                   # newest recording -> captures/
```

```bash
python scripts/quest_to_colmap.py captures/20261114_153012 --check
```

The converter takes poses from the headset's tracking and starting points from its depth
maps, with no COLMAP run. At the end it compares the colour that the left and the right
camera see at the same 3D points; `--check` also draws the points onto a few frames.

### Recording and replay

Tick "Record each run and replay it when it finishes" on the page. The run is then
recorded on the GPU machine with all of its frames, copied to `recordings/` in the
background, and replayed from there at 15 frames per second. A replay needs no GPU
machine, so it works as an offline fallback. A 3000-step recording is 70 to 120 MB.

## Repository layout

```text
server/
  live_server.py        training loop, frame packing, stream, recorder, HTTP API (GPU machine)
viewer/
  index.html, main.js   the WebXR page: scanning, live view, replay, controls
  camtest.html          checks what a headset's browser offers: cameras in AR, depth
  vendor/               three.js r180 and Spark 2.4.0, so the headset needs no CDN
scripts/                what runs on the computer in between
  dev_proxy.py          serves the viewer, forwards /api, receives scans, serves recordings
  webxr_scan.py         photos and head poses -> COLMAP workspace -> upload -> start training
  quest_cameras.json    lens calibration of a Quest 3S
  deploy.sh, tunnel.sh  start the server on the GPU machine; forward its port
  config.sh             shared settings, read from local.env (see local.env.example)
  calibrate_scan.py     measures the camera-to-pose time offset on a recorded scan
  pull_capture.sh       copy an OpenQuestCapture recording off the headset
  quest_to_colmap.py    OpenQuestCapture recording -> COLMAP workspace
docs/
  pipeline.md           the pipeline, stage by stage
demo/                   how the demo video was made: storyboard, offline renders, the cut
```

`demo/` is not needed to run anything; see [demo/README.md](demo/README.md).

## API

Training server:

| Route | What it does |
|---|---|
| `GET /api/status` | State, step, Gaussian count, loss, and per-frame history |
| `POST /api/start` | `{"scene": "...", "max_steps": 7000, "cap": 300000, "pace": false, "record": false}` |
| `POST /api/stop` | Stop the current run |
| `POST /api/reset` | Stop, and drop the run so viewers have nothing left to show |
| `GET /api/stream?id=V` | Frame stream for viewer `V` |
| `POST /api/ack?id=V&run=R&seq=N` | Viewer `V` has drawn frame `N` of run `R` |
| `GET /api/recordings`, `GET /api/recording/NAME` | Finished recordings; download one (supports `Range`) |

`scripts/dev_proxy.py`, in addition:

| Route | What it does |
|---|---|
| `POST /scan/frame?id=&side=&n=&t=&w=&h=` | One photo of a scan (JPEG body) |
| `POST /scan/finish?id=` | The head pose history; starts conversion, upload and training |
| `GET /scan/status`, `POST /scan/reset` | Where the scan is in that pipeline; forget it |
| `GET /recordings/NAME.lspl` | A local recording, for playback |

A stream, and a recording file, is a sequence of messages: the bytes `LSPL`, a
little-endian u32 header length, a JSON header, then `bytes` of payload. A payload holds
16 columns of `gaussians` bytes each, deflated; adding them (mod 256) to the previous
frame's rows gives the new frame. `"pace": true` makes training wait for the viewer so
that every 20th step is shown live, at the cost of a run about three times longer.

## Limitations

- **Depth is not used.** Reading the headset's depth sensor during a scan, to show a live
  point cloud and to seed training, is written but switched off (`?depth` in the address
  turns it on): asking for depth sensing kept the AR session from starting on our
  headset. Training starts from random points along the view rays instead.
- **Quality.** Scans reach 23 to 26 dB: recognisable, text readable up close, soft in the
  details. Head poses are used as tracked, with no refinement.
- **Distance to the GPU.** Ours was on another continent. Photos took most of a minute to
  reach it, and late in training only one or two frames a second came back.
- **One session.** A scan lines up by itself only within the AR session it was made in.
  After leaving and re-entering AR it has to be lined up by hand.
- It has been run on one Quest 3S.

## Credits

- [gsplat](https://github.com/nerfstudio-project/gsplat) for differentiable Gaussian rasterisation and densification
- [Spark](https://github.com/sparkjsdev/spark) for Gaussian splat rendering in three.js and WebXR
- [three.js](https://threejs.org/)
- [OpenQuestCapture](https://github.com/samuelm2/OpenQuestCapture), whose recordings and lens-pose convention the first working pipeline was built on

All code in this repository was written during the hackathon. Early tests used a gsplat
installation that already existed on the GPU machine and three phone-captured test
scenes from before it; neither is in the repository.

## License

[MIT](LICENSE). The vendored three.js and Spark in `viewer/vendor/` keep their own MIT licenses.
