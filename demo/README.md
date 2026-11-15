# Demo video

Everything that exists only to make demo footage of Live Splat: offline renders of
scanned scenes, and a scripted 30-second vertical cut. None of it is needed to run the
project: that is `server/`, `viewer/` and `scripts/` at the top level.

The video itself, the clips it is cut from and the intermediate renders are large and are
not in the repository. The folders they live in locally are listed below so that the
scripts' paths make sense.

## What is here

```text
demo/
  storyboard.md      what the scripted cut shows, second by second
  render/            offline renders of a scanned scene, made on the GPU machine
    walk_path.py       a scan's recorded head path as a smooth camera path
    render_walk.sh     first-person videos along that path: while training, finished, photos
    render_walk.py     (the renderer render_walk.sh runs on the GPU machine)
    render_orbit.sh    a slow orbit around a finished reconstruction, optionally pushing in
    render_orbit.py    (the renderer render_orbit.sh runs on the GPU machine)
  edit/              cutting the final video
    make_text.py       caption sprites as PNGs
    make_music.py      the soundtrack, synthesised from scratch
    make_demo.py       lays out the clips, captions and music as a 1080x1920 video

  footage/           not tracked: headset recordings, phone clips, finished renders
  renders/           not tracked: per-scan camera paths, saved models, rendered videos
  out/               not tracked: finished cuts
```

## How the footage and the cut are made

**1. Footage from the headset.** The Quest's own screen recorder captures a whole session:
scanning in passthrough, the upload, the reconstruction forming, and switching between
the real room and the reconstruction with B. Stretches of that recording are the
landscape clips in the video.

**2. Renders of the same scans.** A scan keeps its photos and head path, so the scene can
be rendered again offline, from any camera, at any resolution.

```bash
demo/render/render_walk.sh scan-164255 15000
```

trains the scan once and writes three videos with identical camera motion, so they can be
cut against each other frame for frame: the walk while the model trains (`learning.mp4`),
the walk through the finished model (`final.mp4`), and the photos taken along the way
(`photos.mp4`). Extra arguments select one camera move out of the walk and control how
the training step advances over it; `render_walk.py` documents them.

```bash
demo/render/render_orbit.sh pushin --distance 12 --elevation 9 --azimuth 10 --seconds 40 --push-in 8
```

renders the model the training server is showing right now, without training again: a
level orbit around the scene that ends by flying inside it. `--stills 8` writes preview
images instead, for choosing the view; `--width 1080 --height 1920` gives a vertical video.

Both wrappers read the GPU machine's address from `scripts/local.env`, like the rest of
the project.

**3. The cut.** On a Mac, `make_text.py` draws the captions with the system fonts.
`make_music.py` writes the soundtrack: 120 beats a minute, so a bar is two seconds and
every cut in the video sits on a bar line. `make_demo.py` then reads the clips, lays every
frame out at 1080x1920 (landscape footage as a cut-out over a blurred copy of itself),
adds the captions and writes the H.264 video.

```bash
python3 demo/edit/make_text.py                       # writes text/*.png
python demo/edit/make_music.py music.wav
python demo/edit/make_demo.py --src src --text text --out live-splat-demo.mp4
```

`make_demo.py` and the renderers need `imageio-ffmpeg` and OpenCV; they were run on the
GPU machine, where those are installed. The clip names and timings at the top of
`make_demo.py` are the cut list.
