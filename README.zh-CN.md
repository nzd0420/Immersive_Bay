# Live Splat

[English](README.md) | 简体中文

用 Meta Quest 扫描你所在的房间,然后看着它的 3D Gaussian Splatting 重建在你周围原位、按真实大小一点点成形。

扣下扳机,四处看看。头显通过透视相机拍照,并记下每张照片拍摄时头部的位置。GPU 服务器开始训练,每 20 步把高斯传回头显:一团模糊的光斑出现在真实的桌子、椅子和墙所在的位置,然后逐渐变清晰。真实房间被压成黑色,所以你身处其中的,只是模型到目前为止学到的东西。

本项目诞生于 Stanford XR 主办的黑客松 [Immerse the Bay 2026](https://immersethebay.org/)。

## 为什么做这个

3D Gaussian Splatting 通常是个黑盒:上传照片,等待,得到一个场景。但最有意思的其实是优化过程本身。几万个模糊的小团块不断移动、分裂、变锐利,直到同时与每一张照片吻合。Live Splat 把这个过程放到房间尺度上,放在扫描者本人周围,让它在发生的时候就能被看到。

## 工作原理

```text
Quest 浏览器 (WebXR 网页)  ──►  Mac: scripts/dev_proxy.py :8090  ──SSH──►  GPU: server/live_server.py :8765
  扫描,显示结果                  生成训练场景,                              用 gsplat 训练,
                                转发 /api                                  推流高斯
```

头显上的一切都在一个网页里,没有原生应用。

**扫描。** 网页在 AR 会话中用 `getUserMedia` 读取左右两个透视相机(1728×1728,每秒 30 帧),每个相机大约每半秒拍一张;实际总共约每秒 1.4 张,受限于压缩和上传。同时每个 AR 帧记录一次头部位姿,约每秒 90 次。头转得太快或追踪暂停时不拍照。

**从照片到训练场景,不用 COLMAP。** `scripts/webxr_scan.py` 为每张照片算出相机位姿:把头部位姿插值到照片的时间戳,再结合该相机的镜头位置和内参。这些标定数据是头显自己报告的(`scripts/quest_cameras.json`)。结果写成 COLMAP 工作区并上传。整个过程不运行任何运动恢复结构(SfM)。

**原位出现,不需要对齐。** 位姿来自之后用于显示的同一个 AR 会话,所以训练出的高斯本来就在观看端的坐标系里。它们出现在真实物体所在的位置,大小也是真实的。

**训练。** `server/live_server.py` 是一个精简的 [gsplat](https://github.com/nerfstudio-project/gsplat) 训练循环。使用 MCMC 致密化策略把高斯数量控制在上限以内(默认 30 万),这是一体机头显能渲染的规模。

**推流。** 每 20 步,服务器在 GPU 上把全部高斯打包成 [Spark](https://sparkjs.dev/) 渲染器内部使用的格式(每个高斯 16 字节),观看端收到后不用解析就能直接上传到显卡。实际传输的是与上一帧的差值,按列排列后压缩,体积约为原始帧的三分之一。

**链路慢也不滞后。** 观看端每画完一帧就向服务器确认。服务器只允许大约一秒的帧在途,并根据链路状况调整这个窗口。所以连接慢的时候,看到的是更少但更新的帧,画面不会越来越落后于训练。

## 头显里的操作

| 按键 | 作用 |
|---|---|
| 扳机 | 开始扫描;再扣一次结束并开始训练 |
| B / Y | 显示或隐藏重建结果背后的真实房间 |
| A / X | 在房间尺度和桌面全息之间切换 |
| 按下摇杆 | 重置:停止训练,清空一切,准备新的扫描 |
| 握把 | 平移场景(房间尺度)或移动全息模型 |
| 双手握把 | 旋转场景(房间尺度)或缩放全息模型 |

扫描时,视野前方的信息板显示已拍的照片数,每张照片的拍摄位置会留下一个小标记。重建结果的第一帧到达时,真实房间被隐藏。隐藏的办法是在观看者周围放一个黑色球壳,因为 three.js 在 AR 会话里总是把背景清成透明。

不是在当前会话里扫描的场景(手机拍的、OpenQuestCapture 录的)以桌面全息显示,或者按房间尺度显示后手动对齐。

## 实测数据

RTX 3080 Ti。GPU 机器在另一个大洲,链路约 1.2 MB/s。

| | |
|---|---|
| 扫描 | 走动 75 秒得到约 105 张照片,其中 2 张没有对应的头部位姿 |
| 扫描结束到出现第一批高斯 | 大约半分钟,主要花在上传上(没有精确计时) |
| 训练速度 | 每秒 50 到 100 步,取决于高斯数量 |
| 走动扫描一个房间(约 4.7 × 2.5 米)的画质 | 10000 步后留出视角 PSNR 23.6 dB;1500 步时为 18.9 dB |
| 传输的帧大小 | 原始帧的 30–43% |
| 实时刷新,前 1000 步 | 每秒 5 次(每 20 步一帧) |
| 实时刷新,之后 | 每秒 1.5–2 次,受链路限制 |

照片时间戳与对应头部位姿之间的延迟用 `scripts/calibrate_scan.py` 测过:偏移在 0 到 30 毫秒之间时留出视角 PSNR 持平,再大就下降(0 毫秒时 18.9 dB,60 毫秒时 17.8 dB,120 毫秒时 16.9 dB)。

## 运行方法

需要一台带 NVIDIA 显卡的机器,装好 PyTorch、gsplat 1.5 和 `fused-ssim`,并有 gsplat 的 `examples/` 目录(用它的 COLMAP 解析器)。

在 GPU 机器上:

```bash
python server/live_server.py --port 8765 \
  --scenes-dir /path/to/scenes \
  --gsplat-examples /path/to/gsplat/examples
```

扫描会上传到 scenes 目录,服务运行期间随时被识别。其他 COLMAP 工作区可以用 `--scene NAME=DIR` 添加。

在 Mac 上开两个终端:

```bash
scripts/tunnel.sh                # Mac :8765 -> GPU :8765
```

```bash
python3 scripts/dev_proxy.py     # 观看页面在 http://localhost:8090
```

`scripts/deploy.sh` 把 `server/` 拷到我们的 GPU 机器并启动,里面的路径是我们自己环境的。`scripts/tunnel.sh` 复用一条已经打开的 SSH 连接,因为那台机器是密码登录。代理通过同一条连接用 `rsync` 上传扫描。

头显必须从安全来源打开页面。可以用 USB 连接(开启开发者模式):

```bash
adb reverse tcp:8090 tcp:8090
```

然后在 Quest 浏览器里打开 `http://localhost:8090`;也可以通过公网 HTTPS 隧道无线使用:

```bash
cloudflared tunnel --url http://localhost:8090
```

然后打开它输出的地址。在页面上点 **Enable scanning** 并允许使用相机,再点 **Enter AR**。

请把头显的边界设为房间边界(Roomscale),并画得大一些。用小的原地边界时,走出去会暂停 AR 会话,暂停期间拍的照片没有头部位姿。

读取另一台头显的镜头标定:

```bash
adb shell dumpsys media.camera
```

把 50 号和 51 号相机的 `android.lens.intrinsicCalibration`、`poseTranslation` 和 `poseRotation` 填进 `scripts/quest_cameras.json`。

### 获取场景的其他方式

任何去畸变后的 COLMAP 工作区(包含 `images/` 和 `sparse/`)都可以训练和观看。

也可以在 Quest 3 或 3S 上用 [OpenQuestCapture](https://github.com/samuelm2/OpenQuestCapture) 录制,然后在装有 Python、`numpy` 和 `opencv-python` 的电脑上转换:

```bash
scripts/pull_capture.sh                                   # 最新的一段录制 -> captures/
```

```bash
python scripts/quest_to_colmap.py captures/20261114_153012 --check
```

转换脚本输出 `captures/20261114_153012_colmap/`:位姿来自头显的追踪,初始点来自头显的深度图,不运行 COLMAP。结束时它会比较左右相机在同一批 3D 点上看到的颜色;加 `--check` 还会把点画到几张照片上,存到 `check/`。常用选项:`--every 2`、`--min-sharpness 30`、`--cameras left`。把结果拷进服务器的 scenes 目录即可。

### 录制与回放

在页面上勾选 "Record each run and replay it when it finishes"。这次训练的所有帧会录在 GPU 机器上,后台拷到 Mac 的 `recordings/`,再从本地以每秒 15 帧回放。回放不需要 GPU 机器,可以当离线保底。一段 3000 步的录像有 70 到 120 MB。

## 目录结构

```text
server/live_server.py      训练循环、帧打包、推流、录制、HTTP API
viewer/index.html          控制面板
viewer/main.js             扫描、读流、原地更新网格、WebXR 交互
viewer/camtest.html        检查头显浏览器提供什么:AR 里的相机、深度
viewer/vendor/             three.js r180 和 Spark 2.4.0,打包在仓库里,头显不依赖 CDN
scripts/dev_proxy.py       提供观看页面、转发 /api、接收扫描、提供录像
scripts/webxr_scan.py      照片和头部位姿 -> COLMAP 工作区 -> 上传 -> 启动训练
scripts/quest_cameras.json 我们这台 Quest 3S 的镜头标定
scripts/calibrate_scan.py  在一次录下的扫描上测量相机与位姿的时间差
scripts/tunnel.sh          到 GPU 机器的 SSH 端口转发
scripts/deploy.sh          把服务端拷到我们的 GPU 机器并重启
scripts/pull_capture.sh    通过 adb 把 OpenQuestCapture 的录制从头显拷出
scripts/quest_to_colmap.py OpenQuestCapture 录制 -> COLMAP 工作区
```

## API

训练服务器:

| 路由 | 作用 |
|---|---|
| `GET /api/status` | 状态、步数、高斯数量、loss,以及每一帧的历史记录 |
| `POST /api/start` | `{"scene": "...", "max_steps": 7000, "cap": 300000, "pace": false, "record": false}` |
| `POST /api/stop` | 停止当前训练 |
| `POST /api/reset` | 停止并丢弃这次训练,观看端不再有内容可显示 |
| `GET /api/stream?id=V` | 观看端 `V` 的帧流 |
| `POST /api/ack?id=V&run=R&seq=N` | 观看端 `V` 已画完第 `R` 次训练的第 `N` 帧 |
| `GET /api/recordings`、`GET /api/recording/NAME` | 已完成的录像;下载其中一段(支持 `Range`) |

Mac 代理额外提供:

| 路由 | 作用 |
|---|---|
| `POST /scan/frame?id=&side=&n=&t=&w=&h=` | 一次扫描中的一张照片(请求体是 JPEG) |
| `POST /scan/finish?id=` | 头部位姿记录;随后开始转换、上传和训练 |
| `GET /scan/status`、`POST /scan/reset` | 这次扫描进行到哪一步;放弃它 |
| `GET /recordings/NAME.lspl` | 本地录像,用于回放 |

帧流和录像文件都是一串消息:四个字节 `LSPL`,小端 u32 的头部长度,JSON 头部,然后是 `bytes` 字节的数据。数据解压后是 16 列、每列 `gaussians` 个字节;把它们逐字节加到上一帧对应的行上(模 256),就得到新的一帧。`"pace": true` 会让训练等待观看端,保证每 20 步都实时显示,代价是训练时间大约变成三倍。

## 当前状态

已经可用(Quest 3S):

- 在 AR 会话内扫描,USB 和公网隧道两种方式都可以
- 不经过 COLMAP 的转换、上传和训练
- 重建结果按真实大小原位出现,实时更新,并隐藏真实房间
- 重置、桌面全息模式、录制与回放

还没做完:

- **深度。** 扫描时读取头显的深度传感器,用来显示实时点云并作为训练的初始点,这部分代码已经写好但默认关闭(地址后加 `?depth` 可开启):请求深度传感会让 AR 会话无法启动。目前训练从沿视线随机撒的点开始。
- **画质。** 走动扫描的房间能到约 23 dB:能认出来,细节偏软。头部位姿直接使用追踪结果,没有做优化。
- **离 GPU 太远。** 我们的 GPU 在另一个大洲。照片要将近一分钟才能传过去,训练后期每秒只能回传一两帧。
- 一次扫描只在它所在的 AR 会话内有效。退出再进入 AR 后,旧的扫描不会自动对齐。

## 致谢

- [gsplat](https://github.com/nerfstudio-project/gsplat):可微分的高斯光栅化和致密化
- [Spark](https://github.com/sparkjsdev/spark):在 three.js 和 WebXR 中渲染高斯
- [three.js](https://threejs.org/)(MIT)
- [OpenQuestCapture](https://github.com/samuelm2/OpenQuestCapture):第一条跑通的流程就建立在它的录制数据和镜头位姿约定之上

本仓库中的全部代码都是在黑客松期间编写的。GPU 机器上的 gsplat 环境,以及三个用手机拍摄的测试场景,在比赛之前就已存在,两者都不在本仓库中。

## 许可证

[MIT](LICENSE)。`viewer/vendor/` 中打包的 three.js 和 Spark 保留各自的 MIT 许可证。
