#!/usr/bin/env python3
"""Find the camera-to-pose time offset (and check left/right) for scans made in the viewer.

    python3 scripts/calibrate_scan.py scan-134759

A photo's timestamp and the head pose history are on slightly different clocks, and the
camera adds its own delay. This rebuilds one recorded scan with several candidate offsets,
with the two lens calibrations as given and swapped, trains each briefly on the GPU
machine, and reports the held-out PSNR. The best offset goes into Scans(latency_ms=...).
Needs scripts/tunnel.sh running.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webxr_scan import CAPTURES, build_workspace  # noqa: E402

import os

HOST, REMOTE = "autodl_3080TI", "/root/autodl-tmp/immersive_bay/scenes"
# Point this at a second server instance, so calibration runs do not take the training
# server away from whoever is using the viewer.
API = os.environ.get("CAL_API", "http://127.0.0.1:8766")


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, method="POST" if data else "GET")
    return json.load(urllib.request.urlopen(request, timeout=60))


def main():
    scan_id = sys.argv[1]
    offsets = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [-20, 10, 40, 70, 100, 130]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    scan_dir = CAPTURES / scan_id
    calibration = json.loads(Path(__file__).with_name("quest_cameras.json").read_text())
    swapped = {**calibration, "left": calibration["right"], "right": calibration["left"]}
    images = f"{REMOTE}/cal-images-{scan_id}"
    subprocess.run(["ssh", HOST, f"mkdir -p {images}"], check=True)
    subprocess.run(["rsync", "-az", f"{scan_dir}/frames/", f"{HOST}:{images}/"], check=True)
    results = []
    for label, lenses in (("as given", calibration), ("swapped", swapped)):
        for offset in offsets:
            tag = ("s" if label == "swapped" else "g") + f"{int(offset):+d}".replace("+", "p").replace("-", "m")
            name = f"cal-{scan_id}-{tag}"
            out = CAPTURES / "_cal" / name
            summary = build_workspace(scan_dir, out, lenses, offset, copy_images=False)
            remote = f"{REMOTE}/{name}"
            subprocess.run(["ssh", HOST, f"mkdir -p {remote} && ln -sfn {images} {remote}/images"], check=True)
            subprocess.run(["rsync", "-az", f"{out}/sparse", f"{out}/quest_capture.json", f"{HOST}:{remote}/"], check=True)
            call("/api/start", {"scene": name, "max_steps": steps, "record": False})
            while True:
                time.sleep(2)
                status = call("/api/status")
                if status["state"] in ("done", "error", "stopped"):
                    break
            psnr = status["history"][-1]["psnr"] if status["state"] == "done" else float("nan")
            results.append((label, offset, summary["frames"], psnr))
            print(f"{label:9s} offset {offset:+5.0f} ms  frames {summary['frames']:3d}  PSNR {psnr:5.2f}", flush=True)
    best = max(results, key=lambda r: r[3])
    print(f"best: lenses {best[0]}, offset {best[1]:+.0f} ms, PSNR {best[3]:.2f}")
    names = " ".join(f"{REMOTE}/cal-{scan_id}-*" for _ in [0])
    subprocess.run(["ssh", HOST, f"rm -rf {names} {images}"], check=False)


if __name__ == "__main__":
    main()
