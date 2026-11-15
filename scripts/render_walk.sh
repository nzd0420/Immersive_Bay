#!/usr/bin/env bash
# Render first-person videos of a scan made in the viewer, on the GPU machine.
#   scripts/render_walk.sh scan-164255 [steps]
# Writes renders/<scan>/learning.mp4, final.mp4 and photos.mp4 (see server/render_walk.py).
#
# One camera move out of the walk, into renders/<scan>/<name>/; what follows the name goes
# to render_walk.py:
#   scripts/render_walk.sh scan-164255 15000 arch --frames 755:1005 --first-step 1000 --ease 1
set -euo pipefail
cd "$(dirname "$0")/.."
scan="$1"; steps="${2:-15000}"; name="${3:-}"
shift $(( $# < 3 ? $# : 3 ))
out="renders/$scan${name:+/$name}"
host="${HOST:-autodl_pro6000}"
remote=/root/autodl-tmp/immersive_bay
gsplat_project=/root/autodl-tmp/Gaussian_Story

mkdir -p "$out"
python3 scripts/walk_path.py "$scan" > "renders/$scan/path.json"
ssh -o BatchMode=yes "$host" "mkdir -p $remote/$out"
rsync -az server/ "$host:$remote/server/"
rsync -az "renders/$scan/path.json" "$host:$remote/renders/$scan/path.json"
ssh -o BatchMode=yes "$host" "bash -s" <<REMOTE
set -euo pipefail
if [ -f $remote/activate.sh ]; then
  source $remote/activate.sh
  examples=$remote/gsplat/examples
else
  source $gsplat_project/env/nzd/autodl/activate.sh
  examples=$gsplat_project/baselines/gsplat/upstream/examples
fi
cd $remote
python server/render_walk.py --scene-dir scenes/$scan --path renders/$scan/path.json \
  --gsplat-examples \$examples \
  --out $out --steps $steps $*
REMOTE
rsync -az --partial --include="*.mp4" --exclude="*" "$host:$remote/$out/" "$out/"
ls -la "$out"
