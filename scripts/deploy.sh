#!/usr/bin/env bash
# Copy server/ to the GPU machine and (re)start the live training server there.
# Usage: scripts/deploy.sh            (host defaults to autodl_3080TI)
set -euo pipefail
cd "$(dirname "$0")/.."
host="${1:-autodl_3080TI}"
remote=/root/autodl-tmp/immersive_bay
# An existing gsplat install on that machine is reused read-only: its venv and its
# example COLMAP parser. Nothing is written inside it.
gsplat_project=/root/autodl-tmp/Gaussian_Story
port=8765

ssh -o BatchMode=yes "$host" "mkdir -p $remote/server $remote/logs"
rsync -az --delete server/ "$host:$remote/server/"
ssh -o BatchMode=yes "$host" "bash -s" <<EOF
set -euo pipefail
pkill -f "live_server.py --port $port" 2>/dev/null || true
sleep 1
source $gsplat_project/env/nzd/autodl/activate.sh
cd $remote
scenes=""
for name in succulent laptop lobby; do
  dir=$gsplat_project/local/data/\$name/reconstruction/processed
  if [ -d "\$dir/images" ]; then scenes="\$scenes --scene \$name=\$dir"; fi
done
nohup setsid python server/live_server.py --port $port \$scenes \
  --gsplat-examples $gsplat_project/baselines/gsplat/upstream/examples \
  > logs/server.log 2>&1 < /dev/null &
sleep 4
tail -5 logs/server.log
EOF
