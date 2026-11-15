#!/usr/bin/env bash
# Copy server/ to the GPU machine and (re)start the live training server there.
# Usage: scripts/deploy.sh [host]     (an SSH alias; defaults to $HOST, then autodl_pro6000)
set -euo pipefail
cd "$(dirname "$0")/.."
host="${1:-${HOST:-autodl_pro6000}}"
remote=/root/autodl-tmp/immersive_bay
# The Python environment is $remote/activate.sh with gsplat's examples in $remote/gsplat.
# On a machine without those, an existing gsplat project is reused read-only: its venv
# and its example COLMAP parser. Nothing is written inside it.
gsplat_project=/root/autodl-tmp/Gaussian_Story
port=8765

ssh -o BatchMode=yes "$host" "mkdir -p $remote/server $remote/logs"
rsync -az --delete server/ "$host:$remote/server/"
ssh -o BatchMode=yes "$host" "bash -s" <<EOF
set -euo pipefail
pkill -f "live_server.py --port $port" 2>/dev/null || true
sleep 1
if [ -f $remote/activate.sh ]; then
  source $remote/activate.sh
  examples=$remote/gsplat/examples
else
  source $gsplat_project/env/nzd/autodl/activate.sh
  examples=$gsplat_project/baselines/gsplat/upstream/examples
fi
cd $remote
scenes=""
for name in succulent laptop lobby; do
  dir=$gsplat_project/local/data/\$name/reconstruction/processed
  if [ -d "\$dir/images" ]; then scenes="\$scenes --scene \$name=\$dir"; fi
done
# Quest scans uploaded to scenes/<name>/ are picked up while the server runs.
mkdir -p $remote/scenes
nohup setsid python server/live_server.py --port $port \$scenes --scenes-dir $remote/scenes \
  --gsplat-examples \$examples \
  > logs/server.log 2>&1 < /dev/null &
sleep 4
tail -5 logs/server.log
EOF
