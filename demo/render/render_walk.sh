#!/usr/bin/env bash
# Render first-person videos of a scan made in the viewer, on the GPU machine.
#   demo/render/render_walk.sh scan-164255 [steps]
# Writes demo/renders/<scan>/learning.mp4, final.mp4 and photos.mp4 (see render_walk.py).
#
# One camera move out of the walk, into demo/renders/<scan>/<name>/; what follows the name
# goes to render_walk.py:
#   demo/render/render_walk.sh scan-164255 15000 arch --frames 755:1005 --first-step 1000 --ease 1
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source scripts/config.sh
need_gpu_host
scan="$1"; steps="${2:-15000}"; name="${3:-}"
shift $(( $# < 3 ? $# : 3 ))
out="renders/$scan${name:+/$name}"   # on the GPU machine; kept here under demo/

mkdir -p "demo/$out"
python3 demo/render/walk_path.py "$scan" > "demo/renders/$scan/path.json"
ssh -o BatchMode=yes "$GPU_HOST" "mkdir -p $GPU_DIR/$out"
# The renderer imports the training code, so it runs next to it on the GPU machine.
rsync -az server/ "$GPU_HOST:$GPU_DIR/server/"
rsync -az demo/render/render_walk.py "$GPU_HOST:$GPU_DIR/server/render_walk.py"
rsync -az "demo/renders/$scan/path.json" "$GPU_HOST:$GPU_DIR/renders/$scan/path.json"
ssh -o BatchMode=yes "$GPU_HOST" "bash -s" <<REMOTE
set -euo pipefail
cd $GPU_DIR
source $GPU_ACTIVATE
python server/render_walk.py --scene-dir scenes/$scan --path renders/$scan/path.json \
  --gsplat-examples $GSPLAT_EXAMPLES \
  --out $out --steps $steps $*
REMOTE
rsync -az --partial --include="*.mp4" --exclude="*" "$GPU_HOST:$GPU_DIR/$out/" "demo/$out/"
ls -la "demo/$out"
