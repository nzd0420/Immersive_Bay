#!/usr/bin/env bash
# Render first-person videos of a scan made in the viewer, on the GPU machine.
#   scripts/render_walk.sh scan-164255 [steps]
# Writes renders/<scan>/learning.mp4, final.mp4 and photos.mp4 (see server/render_walk.py).
set -euo pipefail
cd "$(dirname "$0")/.."
scan="$1"; steps="${2:-15000}"
host="${HOST:-autodl_3080TI}"
remote=/root/autodl-tmp/immersive_bay
gsplat_project=/root/autodl-tmp/Gaussian_Story

mkdir -p "renders/$scan"
python3 scripts/walk_path.py "$scan" > "renders/$scan/path.json"
ssh -o BatchMode=yes "$host" "mkdir -p $remote/renders/$scan"
rsync -az server/ "$host:$remote/server/"
rsync -az "renders/$scan/path.json" "$host:$remote/renders/$scan/path.json"
ssh -o BatchMode=yes "$host" "bash -s" <<REMOTE
set -euo pipefail
source $gsplat_project/env/nzd/autodl/activate.sh
cd $remote
python server/render_walk.py --scene-dir scenes/$scan --path renders/$scan/path.json \
  --gsplat-examples $gsplat_project/baselines/gsplat/upstream/examples \
  --out renders/$scan --steps $steps
REMOTE
rsync -az --partial --include="*.mp4" --exclude="*" "$host:$remote/renders/$scan/" "renders/$scan/"
ls -la "renders/$scan"
