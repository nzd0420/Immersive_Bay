# Storyboard

This is the cut that `edit/make_demo.py` produces: 29.5 seconds, vertical (1080x1920),
with captions and music and no voice. It has one job: someone who has never heard of Gaussian Splatting should understand, in
the first few seconds, that a person scanned a room and the room then rebuilt itself
around them.

The shot the whole video leads to is the reconstruction going from fog to a sharp scene.
Everything else is kept short.

| Time | Picture | Caption | Where it comes from |
|---|---|---|---|
| 0–3 s | Putting the headset on, filmed on a phone; the camera pushes into the headset | **Darkroom** | phone clip |
| 3–6 s | A dissolve into a reconstruction seen from outside, a cloud of Gaussians; the camera flies in and lands in the scanned corner | **No app. No cable.** | `render_orbit.sh` with `--push-in` |
| 6–10 s | Looking around a room in passthrough; a green marker appears for every photo | **01 SCAN** · Look around. | headset recording, 2× speed |
| 10–14 s | Two moments of one training run, one above the other. Below: the real room goes black and points appear where it was and gather into the room. Above: a minute and a half later, near step 9,000, looking around inside the reconstruction | **02 TRAIN** · Training, live. (between the two pictures) | headset recording, 2× speed |
| 14–20 s | A turn through the room while the view switches seven times between the real room and the reconstruction; a B button under the picture goes grey at each press | **03 COMPARE** · Reality / Gaussian splat | headset recording, 1.4× speed |
| 20–26 s | An outdoor scene forming, a third-person shot of the person scanning it, the scene finished; the training step counts up to 15,000 | **ANY PLACE** · Outdoors too. | `render_walk.sh`; the middle 2.4 s is from an earlier rough cut, filmed on a phone |
| 26–29.5 s | Inside the finished reconstruction, dimmed | **Darkroom** · Yann Wu & Zidong Ning · All headset footage is a live screen recording, rendered in real time. Not a replay. | end of the orbit render |

Captions are white with a narrow black outline and sit above the picture (between the
two pictures of the training stretch); nothing is written below it. Every cut falls on a bar line of the music, which runs at 120 beats a
minute; the opening dissolve is centred on 3 s. The exact clips, start times and speeds
are the `SEGMENTS` list at the top of `edit/make_demo.py`, and the caption texts are in
`edit/make_text.py`; if this table and those two ever disagree, the scripts are right.

## What is real time and what is not

- The scanning and the training in the headset recording are shown as they happened,
  played at twice the speed. The reconstruction on screen is the model being trained at
  that moment, streamed to the headset every 20 steps.
- The outdoor scene is an offline render of a scan made the same way. It compresses one
  training run of 15,000 steps into a few seconds. Nothing on screen says so: the end card
  makes its claim about headset footage only, and this is not headset footage.
- The opening and closing orbit is an offline render of a model that was trained live:
  the exact Gaussians that were on screen in the headset, saved and rendered from a
  camera path the headset could not fly.
