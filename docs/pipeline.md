# Technical pipeline

English | [简体中文](pipeline.zh-CN.md)

How a scan becomes a reconstruction that forms around the person who made it. The numbers
come from the code and from the "Measured" section of the README.

## Overview

```text
Quest 3S browser (one WebXR page)
   │  1. JPEG photos + head poses            ▲ 4. Gaussian frames (difference, deflated)
   ▼  HTTPS (public tunnel, or USB)          │
Computer in between (scripts/dev_proxy.py: web server, conversion, forwarding)
   │  2. rsync over SSH                      ▲ 3. HTTP stream, through an SSH port forward
   ▼                                         │
GPU machine (server/live_server.py: gsplat training)
```

Each machine does one thing. The headset captures and displays, the computer in between
converts and relays, the GPU machine trains.

## 1. Capturing on the headset

- **Photos.** During the AR session the page opens both passthrough cameras with
  `getUserMedia` (1728×1728, 30 frames per second). From each camera it takes a frame
  every 450 ms, scales it to 1024×1024 and encodes it as JPEG (quality 0.88), up to 240
  photos per scan.
- **Poses.** On every XR frame, about 90 times a second, the page records the head pose,
  stamped with `frame.predictedDisplayTime`.
- **Filtering.** No photo is taken while the head turns faster than 50° per second, or
  when there has been no pose for 150 ms. That avoids blurred photos and photos that
  cannot be placed.
- **Upload.** Photos are sent as they are taken (`POST /scan/frame`); the pose history is
  sent once, when the scan ends (`POST /scan/finish`).

## 2. The computer in between

`scripts/dev_proxy.py` plays four parts.

- **Web server.** WebXR and the cameras both require a secure origin. The headset reaches
  port 8090 either through an HTTPS tunnel (wireless) or through `adb reverse` (USB, which
  makes it `localhost`).
- **Receiving a scan.** Photos go to `captures/<scan>/`, each with a line of metadata:
  which camera, its number, its timestamp.
- **Building the training scene.** This is what replaces COLMAP.
  - For each photo, the head pose is interpolated to the photo's timestamp minus 15 ms
    (position linearly, rotation spherically). The 15 ms is the camera's delay, measured
    with `scripts/calibrate_scan.py`.
  - Applying that camera's position and orientation relative to the head gives the camera
    pose; the intrinsics are scaled to 1024 pixels. This calibration is read out of the
    headset with `adb shell dumpsys media.camera`.
  - Coordinates go from WebXR (y up, z towards the viewer) to COLMAP (y down, z forward).
  - The result is written directly as a COLMAP workspace. No feature matching and no
    structure-from-motion run. The starting points are scattered at random along the
    cameras' view rays.
- **The only way to the GPU machine.** The GPU machine needs no public port, only SSH from
  this computer. The workspace is copied with `rsync` into the server's scenes folder,
  and training is started through an SSH port forward (local 8765 to the server's 8765).
  From then on every `/api/*` request of the headset is forwarded the same way.

## 3. Training

- gsplat 1.5 with the MCMC densification strategy, at most 300,000 Gaussians (what the
  headset can render), spherical harmonics of degree 0, 7000 steps by default.
- The loss is 0.8·L1 + 0.2·SSIM, plus regularisers on opacity and scale.
- 50 to 100 steps per second, depending on how many Gaussians there are.

## 4. Streaming back and showing it live

A frame is published every 20 steps. The problem: a raw frame is close to 5 MB, and our
link carried about 1.2 MB/s.

- **Packed on the GPU.** All Gaussians are encoded into the 16-byte layout the Spark
  renderer uses internally: colour and opacity in 4 bytes, the centre as three float16,
  the rotation in 3 bytes, the log-scales in 3 bytes. 300,000 Gaussians are 4.8 MB. The
  headset writes a frame into GPU memory without parsing it.
- **Only differences are sent.** Each frame is subtracted byte by byte from the previous
  one, rearranged by column so that like fields sit together, and deflated. Twenty steps
  change little, so a frame ends up at 30–43% of its raw size.
- **Flow control by acknowledgement.** The headset acknowledges each frame it has drawn.
  The server sends a new frame only while the unacknowledged bytes are below a window,
  which starts at 600 KB and grows or shrinks by 15–20% according to how long
  acknowledgements take. On a slow link the effect is that intermediate frames are
  skipped and the newest is sent, instead of a backlog building up.
- **Updated in place.** The browser inflates the frame with `DecompressionStream`, adds it
  to the previous one, overwrites the same `PackedSplats` buffer and calls
  `mesh.updateVersion()` so that Spark uploads the texture again. There is one mesh for
  the whole session; nothing is created or destroyed.

### Update rate, measured

| Phase | Model updates | Limited by |
|---|---|---|
| First 1000 steps | 5 per second (every 20-step frame arrives) | training speed |
| After that | 1.5–2 per second | the network (more Gaussians, larger frames) |

Two rates are easy to confuse here. One or two per second is how often the *model*
changes. The headset's own rendering rate is unaffected: between two updates, turning the
head or walking redraws the current Gaussians at the display's refresh rate.

## Why the Gaussians come back in place

The camera poses come from the head tracking of the same AR session, so the trained scene
is in the headset's coordinate system from the start. The server normalises a scene when
it loads it and sends the inverse transform (`to_world`) along with the scene's
description; with that applied, the Gaussians land where the real things are, at 1:1
scale, with no registration step.
