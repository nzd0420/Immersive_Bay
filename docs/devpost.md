# Devpost 提交文案(草稿)

> 给队友的说明:下面英文部分可以直接粘贴到 Devpost 对应的栏目里。方括号里的内容需要你们填写或确认。
> 截至周六下午,Devpost 页面上的提交要求和评审标准还是占位符,请以 live dashboard 的 "Tracks & Submit" 为准。
> 文中的数字都来自我们自己的实测,出处见 README 的 "Measured" 一节。演示前如果重新测了,请同步改这里。

---

## Project name

Live Splat

## Tagline

Scan the room you are in with a Quest, then stand inside it while a 3D reconstruction of it learns itself into focus around you.

## Tracks

- AI Horizons (Best Use of AI with XR)
- Creative Zenith (Best of Creative XR)

[确认是否报这两个。我们没有用 smart glasses,也没有用 Orbis,所以不符合 ManageXR 和 Visko 赛道。]

## Built with

webxr, three.js, spark, gaussian-splatting, gsplat, pytorch, python, javascript, meta-quest, cloudflare-tunnel

---

## Inspiration

3D Gaussian Splatting turns a handful of photos into a scene you can walk through, and it has become the default way to capture real places. But you only ever see the result. You upload photos, a progress bar fills, and a finished scene comes back.

The part nobody sees is the best part. Training starts as a cloud of blurry blobs. Over a few thousand steps they drift, split and sharpen until every one of them agrees with every photo at once. It looks like a room condensing out of fog.

We wanted to stand inside that. Not a recording of it on a monitor: the room you are actually in, rebuilding itself around you, at real size, while the optimiser is still working.

## What it does

1. **Scan.** Put on a Quest 3S, open a web page, enter AR and pull the trigger. Look around the room for half a minute to a minute. The page photographs through the headset's two passthrough cameras and records where your head was for each photo. Small markers appear where each photo was taken.
2. **Train.** Pull the trigger again. The photos go to a GPU server and training starts. No structure-from-motion step, no waiting for COLMAP.
3. **Watch it learn, in place.** Every 20 training steps the current Gaussians are streamed back into the same AR session. They appear exactly where the real table, chairs and walls are, at real size. The real room fades to black, so everything you see is what the model has learned so far. First a haze of coloured blobs, then shapes, then the room.
4. **Compare.** Press B to bring the real room back and check the reconstruction against reality, or switch to a table-top hologram of the scene that you can pick up and resize.

Everything on the headset is a single WebXR page. There is no app to install.

## How we built it

**Scanning in the browser.** The Quest browser exposes the passthrough cameras to a web page through `getUserMedia`, at 1728×1728 and 30 frames per second, and keeps them running during an AR session. It does not tell you where each camera was. So the page records the head pose on every AR frame, about 90 times a second, and timestamps every photo. On the Mac, each photo gets a camera pose by interpolating the head pose to the photo's timestamp and applying that camera's lens position and intrinsics, which we read out of the headset itself.

**No alignment step.** Because the poses come from the same AR session that later shows the result, the trained scene is already in the viewer's coordinate system. Nothing has to be registered or lined up: the Gaussians come back where the real things are.

**Training.** A compact training loop on [gsplat](https://github.com/nerfstudio-project/gsplat), using the MCMC densification strategy so the number of Gaussians stays under a cap (300,000) that a standalone headset can render.

**Streaming a model that is still training.** Every 20 steps the server packs all Gaussians, on the GPU, into the 16-byte layout that the [Spark](https://sparkjs.dev/) renderer uses internally, so the headset can upload a frame to its GPU without parsing anything. Only the difference from the previous frame is sent, stored column by column and deflated, which is about a third of the raw size. The viewer acknowledges every frame it has drawn and the server keeps about one second of frames in flight, so a slow link shows fewer, newer frames instead of falling behind.

**Rendering.** three.js and Spark in WebXR. One mesh for the whole session, overwritten in place as frames arrive.

## Challenges we ran into

- **Two apps, two coordinate systems.** Our first pipeline scanned with a native capture app and viewed in the browser. The reconstruction was the right size but floated in the wrong place, because each app has its own tracking origin and there is no anchor the two can share. Moving the scan into the same AR session as the viewer removed the problem instead of solving it.
- **Camera timing.** A photo and a head pose are stamped by different parts of the system. We built a small tool that rebuilds a recorded scan with different time offsets and trains each briefly. Quality is flat for an offset of 0 to 30 ms and falls off beyond that, so we could stop guessing.
- **A GPU on another continent.** Our only GPU was reachable at about 1.2 MB/s. A raw frame of 300,000 Gaussians is almost 5 MB. Difference coding and flow control are what made a live view possible at all.
- **Overwritten photos.** For a while scans trained to mush. The page had opened each camera several times, the streams all numbered their photos from zero, and photos overwrote each other and were paired with the wrong poses. A 144-photo scan held 42 distinct files. Finding that took us from 16 dB to 23 dB.
- **Hiding reality.** three.js always clears to transparent in an AR session, so setting an opaque background does nothing. The real room is hidden by a black sphere around the viewer.
- **The boundary.** With the headset's small stationary boundary, stepping outside pauses the AR session and every photo taken during the pause is useless. The fix is a room-scale boundary drawn generously.

## Accomplishments that we're proud of

- Scan, train and view happen in one AR session, in one web page, with no native app and no structure-from-motion.
- The reconstruction appears at real size in the place it was scanned, with nothing to line up.
- You watch a model train from the inside, live, a frame every 20 steps.
- A walked scan of a room about 4.7 × 2.5 m reaches 23.6 dB PSNR on views held out from training.

## What we learned

- The Quest browser is a far more capable sensor platform than we expected: stereo camera frames, 90 Hz head poses and calibration data are all within reach of a web page.
- Sharing a coordinate system is worth more than any registration algorithm. The cheapest way to align two things is to never let them come apart.
- When a reconstruction is bad, look at the data before the optimiser. Our worst quality problem was a file-naming collision, not a training setting.

## What's next

- **Depth.** The headset has a depth sensor, and the code to read it during a scan, show it as a live point cloud and start training from it is written. It is switched off, because asking for depth sensing kept the AR session from starting on our headset. Training currently starts from random points.
- **Sharper results.** Refine the head poses during training instead of trusting the tracker, and handle the cameras' changing exposure.
- **A closer GPU.** Most of the waiting is network. On a GPU in the same building, photos would arrive as they are taken and training could start while you are still scanning.
- **Scan while training.** Keep adding photos to a model that is already learning, so the room fills in where you look.

---

## Links

- GitHub: https://github.com/nzd0420/Immersive_Bay  [提交前需要仓库主人把仓库设为公开]
- Demo video: [待填]

## 提交前检查

- [ ] 仓库已设为公开(Settings → General → Danger Zone → Change visibility)
- [ ] 上传了演示视频
- [ ] 至少 3 张截图或 GIF:扫描中的标记、训练中期的模糊状态、训练完成后的清晰状态
- [ ] 选好了赛道
- [ ] 两位队员都已加入 Devpost 项目
- [ ] 8:30 前提交,留出余量
