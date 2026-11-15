#!/usr/bin/env bash
# Render an orbit video of the scene the training server is showing right now.
#   scripts/render_orbit.sh NAME [options for server/render_orbit.py]
#   scripts/render_orbit.sh orbit --distance 12 --elevation 9 --azimuth 10 --seconds 40
#   scripts/render_orbit.sh pushin --distance 12 --elevation 9 --azimuth 10 --seconds 40 --push-in 8
#
# Saves the server's current frame as renders/<scene>/final_frame.bin and .json (once: an
# existing copy is kept and reused, so a scene can still be rendered after the server has
# moved on), renders on the GPU machine and writes videos/<scene>-NAME.mp4.
# Needs scripts/tunnel.sh running.
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
host="${HOST:-autodl_3080TI}"
remote=/root/autodl-tmp/immersive_bay
gsplat_project=/root/autodl-tmp/Gaussian_Story

scene="$(python3 - <<'PY'
import http.client, json, os, urllib.request, zlib
status = json.load(urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=20))
scene = os.environ.get("SCENE") or status.get("scene", {}).get("name")
if not scene:
    raise SystemExit("the server has no scene loaded; set SCENE=<name> to reuse a saved frame")
folder = f"renders/{scene}"
if not os.path.exists(f"{folder}/final_frame.bin"):
    if status.get("scene", {}).get("name") != scene or status["state"] not in ("done", "stopped"):
        raise SystemExit(f"{scene} is not the finished scene on the server, and no frame of it is saved")
    connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=60)
    connection.request("GET", "/api/stream?id=render-orbit")
    response = connection.getresponse()
    def read(count):
        data = b""
        while len(data) < count:
            data += response.read(count - len(data))
        return data
    while True:  # the first message with a payload is the whole frame
        prefix = read(8)
        header = json.loads(read(int.from_bytes(prefix[4:], "little")))
        if header["bytes"]:
            payload = read(header["bytes"])
            break
    connection.close()
    planes = zlib.decompress(payload)
    count = header["gaussians"]
    rows = bytearray(count * 16)  # stored column by column: turn back into rows
    for column in range(16):
        rows[column::16] = planes[column * count:(column + 1) * count]
    os.makedirs(folder, exist_ok=True)
    open(f"{folder}/final_frame.bin", "wb").write(rows)
    json.dump({"scene": status["scene"], "step": header["step"], "gaussians": count,
               "psnr": header["psnr"], "config": status["config"]},
              open(f"{folder}/final_frame.json", "w"), indent=1)
print(scene)
PY
)"
echo "scene: $scene"
ssh -o BatchMode=yes "$host" "mkdir -p $remote/renders/$scene/orbit"
rsync -az server/render_orbit.py "$host:$remote/server/render_orbit.py"
rsync -az "renders/$scene/final_frame.bin" "renders/$scene/final_frame.json" "$host:$remote/renders/$scene/"
ssh -o BatchMode=yes "$host" "bash -s" <<REMOTE
set -euo pipefail
source $gsplat_project/env/nzd/autodl/activate.sh >/dev/null 2>&1
cd $remote
python server/render_orbit.py --frame renders/$scene/final_frame.bin --info renders/$scene/final_frame.json \
  --out renders/$scene/orbit/$name.mp4 $* 2>&1 | grep -v -i "warn" | tail -3
REMOTE
mkdir -p videos
rsync -az --partial "$host:$remote/renders/$scene/orbit/$name.mp4" "videos/$scene-$name.mp4"
ls -la "videos/$scene-$name.mp4"
