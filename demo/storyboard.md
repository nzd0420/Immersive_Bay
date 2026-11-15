# Storyboard

The video is 29.5 seconds, vertical (1080x1920), with captions and music and no voice.
It has one job: someone who has never heard of Gaussian Splatting should understand, in
the first few seconds, that a person scanned a room and the room then rebuilt itself
around them.

The shot the whole video leads to is the reconstruction going from fog to a sharp scene.
Everything else is kept short.

| Time | Picture | Caption | Where it comes from |
|---|---|---|---|
| 0–3 s | Opening shot, filmed on a phone | **Live Splat** · Scan a room. Watch it rebuild around you. | phone clip |
| 3–6 s | A reconstruction seen from outside: a cloud of Gaussians, slowly turning | **No app. No cable.** One headset. One web page. | `render_orbit.sh` with `--push-in` |
| 6–10 s | Looking around a room in passthrough; a green marker appears for every photo | **01 SCAN** · Look around. Each green frame: one photo + head pose. | headset recording, 2× speed |
| 10–14 s | The real room goes black; blurry blobs appear where the room was and sharpen | **02 TRAIN** · Training, live. Real-time. Not a replay. | headset recording, 2× speed |
| 14–20 s | An outdoor scene forming along a walk, with the training step counting up to 15,000 | **ANY PLACE** · Outdoors too. From random points to a real place. | `render_walk.sh`, time-lapse of one training run |
| 20–26 s | The real room, a fade through black, the reconstruction in the same place | **03 COMPARE** · Reality / Gaussian splat. Same place. Real scale. | headset recording |
| 26–29.5 s | Inside the finished reconstruction | **Live Splat** · Real-time 3D Gaussian Splatting. Fully wireless. One web page on Quest 3S. | end of the orbit render |

Every cut falls on a bar line of the music, which runs at 120 beats a minute, except the
fade through black at 3 s. The exact clips, start times and speeds are the `SEGMENTS`
list at the top of `edit/make_demo.py`, and the caption texts are in `edit/make_text.py`;
if this table and those two ever disagree, the scripts are right.

## What is real time and what is not

- The scanning and the training in the headset recording are shown as they happened,
  played at twice the speed. The reconstruction on screen is the model being trained at
  that moment, streamed to the headset every 20 steps.
- The outdoor scene is an offline render of a scan made the same way. It compresses one
  training run of 15,000 steps into six seconds and is labelled as a time-lapse.
- The opening and closing orbit is an offline render of a model that was trained live:
  the exact Gaussians that were on screen in the headset, saved and rendered from a
  camera path the headset could not fly.
