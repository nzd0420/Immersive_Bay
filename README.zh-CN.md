# Live Splat

[English](README.md) | 简体中文

在你眼前看着 3D Gaussian Splatting 一点点把场景重建出来。

场景在 GPU 服务器上训练,同时 Meta Quest 头显里每 20 个训练步刷新一次画面:高斯点从稀疏的点云逐渐变清晰,最后成为完整的场景,以全息模型的形式悬浮在透视(passthrough)AR 里。训练结束后会自动循环回放,可以用稳定的帧率把这三十秒的"学习过程"再看一遍。

本项目诞生于 Stanford XR 主办的黑客松 [Immerse the Bay 2026](https://immersethebay.org/)。

## 为什么做这个

3D Gaussian Splatting 通常是个黑盒:上传照片,等待,得到一个场景。但最有意思的其实是优化过程本身。几万个模糊的小团块不断移动、分裂、变锐利,直到同时与每一张照片吻合。Live Splat 把这个过程放到房间尺度上,让你可以在它发生的时候绕着它走。

## 工作原理

```text
Quest 浏览器 (WebXR)  ──USB──►  Mac: scripts/dev_proxy.py :8090  ──SSH──►  GPU: server/live_server.py :8765
  viewer/ (three.js + Spark)      提供观看页面和回放,                       用 gsplat 训练,
                                  转发 /api                                推流并录制每一帧
```

**训练。** `server/live_server.py` 是一个精简的 [gsplat](https://github.com/nerfstudio-project/gsplat) 训练循环。使用 MCMC 致密化策略把高斯数量控制在上限以内(默认 30 万),这是一体机头显能渲染的规模。

**推流。** 每 20 步,服务器在 GPU 上把全部高斯打包成 [Spark](https://sparkjs.dev/) 渲染器内部使用的格式(每个高斯 16 字节),观看端收到后不用解析就能直接上传到显卡。实际传输的是与上一帧的差值,按列排列后压缩,体积约为原始帧的三分之一。

**链路慢也不滞后。** 观看端每画完一帧就向服务器确认。服务器只允许大约一秒的帧在途,并根据链路状况调整这个窗口。所以连接慢的时候,看到的是更少但更新的帧,画面不会越来越落后于训练。

**回放。** 每次训练的所有帧同时录制在 GPU 服务器上。Mac 上的代理在后台把录完的文件拷到 `recordings/`(训练进行中会暂停,不抢带宽),观看端从本地以每秒 15 帧回放。回放完全不依赖 GPU 服务器,所以也能当作演示时的离线保底。

**观看。** `viewer/` 是一个 WebXR 网页。桌面上用鼠标拖动旋转。头显里场景出现在桌面高度的透视画面中:握把移动,双手握把缩放,扳机重新开始训练。

## 实测数据

RTX 3080 Ti,100 张 1280×717 图片的测试场景,观看端到 GPU 服务器的链路约 1.2 MB/s:

| | |
|---|---|
| 训练速度 | 约 100 步/秒;3000 步用时 29 秒 |
| 3000 步后的画质 | 留出视角 PSNR 约 29 dB |
| 传输的帧大小 | 原始帧的 30–43% |
| 实时刷新,前 1000 步 | 每秒 5 次(每 20 步一帧) |
| 实时刷新,之后 | 每秒 1.5–2 次,受链路限制 |
| 3000 步训练的录像 | 151 帧,69 MB |
| 回放 | 稳定每秒 15 帧 |

## 运行方法

需要一台带 NVIDIA 显卡的机器,装好 PyTorch、gsplat 1.5 和 `fused-ssim`,并有 gsplat 的 `examples/` 目录(用它的 COLMAP 解析器)。场景需要是去畸变后的 COLMAP 工作区(包含 `images/` 和 `sparse/`)。

在 GPU 机器上:

```bash
python server/live_server.py --port 8765 \
  --scene myscene=/path/to/colmap/workspace \
  --gsplat-examples /path/to/gsplat/examples
```

在 Mac 上开两个终端:

```bash
scripts/tunnel.sh                # Mac :8765 -> GPU :8765
```

```bash
python3 scripts/dev_proxy.py     # 观看页面在 http://localhost:8090
```

`scripts/deploy.sh` 把 `server/` 拷到我们的 GPU 机器并启动,里面的路径是我们自己环境的。`scripts/tunnel.sh` 复用一条已经打开的 SSH 连接,因为那台机器是密码登录。

头显开启开发者模式并用 USB 线连接后:

```bash
adb reverse tcp:8090 tcp:8090
```

然后在 Quest 浏览器里打开 `http://localhost:8090`,点 **Enter AR**。`localhost` 属于安全上下文,所以 WebXR 不需要证书就能用。

### 用 Quest 扫描的场景训练

先在 Quest 3 或 3S 上用 [OpenQuestCapture](https://github.com/samuelm2/OpenQuestCapture) 录一段场景(1–2 分钟,动作放慢),然后在装有 Python、`numpy` 和 `opencv-python` 的电脑上运行:

```bash
scripts/pull_capture.sh                                   # 最新的录制 -> captures/
```

```bash
python scripts/quest_to_colmap.py captures/20261114_153012 --check
```

转换脚本会生成 `captures/20261114_153012_colmap/`,这是一个 COLMAP 工作区,服务端可以像读其他场景一样直接读取:把它拷到 GPU 机器上,启动时传 `--scene myscan=/path/to/20261114_153012_colmap`。整个过程不需要运行 COLMAP。相机位姿来自头显自身的追踪(按每帧的时间戳插值,再加上每个透视相机镜头相对头部的位姿),所以场景以米为单位,重力沿世界坐标的 y 轴。初始点云来自头显的深度图,反投影后用时间最近的一帧上色。图像存成 JPEG,方便上传。

脚本最后会比较左右两个相机在同一批 3D 点上看到的颜色。如果差异远小于随机像素之间的差异,说明位姿是一致的;加 `--check` 还会把点画到几帧图像上,存在 `check/` 里。常用选项:`--every 2`(只用一半的帧)、`--min-sharpness 30`(丢掉运动模糊的帧)、`--cameras left`(只用左相机)。

## 目录结构

```text
server/live_server.py   训练循环、帧打包、推流、录制、HTTP API
viewer/index.html       控制面板
viewer/main.js          读流、原地更新网格、回放、WebXR 交互
viewer/vendor/          three.js r180 和 Spark 2.4.0,打包在仓库里,头显不依赖 CDN
scripts/dev_proxy.py    提供观看页面、转发 /api、下载并提供录像
scripts/tunnel.sh       到 GPU 机器的 SSH 端口转发
scripts/deploy.sh       把服务端拷到我们的 GPU 机器并重启
scripts/pull_capture.sh 通过 adb 把 OpenQuestCapture 的录制从头显拷到电脑
scripts/quest_to_colmap.py  把 OpenQuestCapture 录制转成可训练的 COLMAP 工作区
```

## API

| 路由 | 作用 |
|---|---|
| `GET /api/status` | 状态、步数、高斯数量、loss,以及每一帧的历史记录 |
| `POST /api/start` | `{"scene": "...", "max_steps": 7000, "cap": 300000, "pace": false}` |
| `POST /api/stop` | 停止当前训练 |
| `GET /api/stream?id=V` | 观看端 `V` 的帧流 |
| `POST /api/ack?id=V&run=R&seq=N` | 观看端 `V` 已画完第 `R` 次训练的第 `N` 帧 |
| `GET /api/recordings` | 已完成的录像(Mac 代理会附上下载进度) |
| `GET /api/recording/NAME` | 从 GPU 机器下载录像(支持 `Range` 断点续传) |
| `GET /recordings/NAME.lspl` | 仅 Mac 代理:本地录像,用于回放 |

帧流和录像文件都是一串消息:四个字节 `LSPL`,小端 u32 的头部长度,JSON 头部,然后是 `bytes` 字节的数据。数据解压后是 16 列、每列 `gaussians` 个字节;把它们逐字节加到上一帧对应的行上(模 256),就得到新的一帧。`"pace": true` 会让训练等待观看端,保证每 20 步都实时显示,代价是训练时间大约变成三倍。

## 当前状态

已经可用(在桌面浏览器里验证过):

- 训练服务、实时帧流、流量控制
- 录制、后台下载到 Mac、循环回放

还没做完:

- **头显实测。** WebXR 部分(AR 会话、抓取、缩放、扳机)已经写好,但还没在 Quest 上运行过。
- **用真实的 Quest 扫描训练。** `scripts/quest_to_colmap.py` 已经写好,并通过了合成数据的往返测试(用渲染出的假录制,包含带填充的 YUV 行跨度、裁剪过的传感器和不对称的深度视场:位姿重投影误差在 0.1 像素以内,深度点落在真实表面上)。从相机的 Android 镜头位姿到头部坐标系的换算沿用了 OpenQuestCapture 自己的导出代码,还没有用真实录制验证过。
- 长时间训练的录像文件很大,7000 步的训练还没实测过。

## 致谢

- [gsplat](https://github.com/nerfstudio-project/gsplat):可微分的高斯光栅化和致密化
- [Spark](https://github.com/sparkjsdev/spark):在 three.js 和 WebXR 中渲染高斯
- [three.js](https://threejs.org/)(MIT)

本仓库中的全部代码都是在黑客松期间编写的。GPU 机器上的 gsplat 环境,以及上面实测所用的三个测试场景,在比赛之前就已存在,两者都不在本仓库中。

## 许可证

[MIT](LICENSE)。`viewer/vendor/` 中内置的 three.js 和 Spark 沿用它们各自的 MIT 许可证。
