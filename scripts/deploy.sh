#!/usr/bin/env bash
# Copy server/ to the GPU machine and (re)start the live training server there.
# Usage: scripts/deploy.sh [host]     (host: an SSH alias; default GPU_HOST from local.env)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
GPU_HOST="${1:-$GPU_HOST}"
need_gpu_host
port=8765

ssh -o BatchMode=yes "$GPU_HOST" "mkdir -p $GPU_DIR/server $GPU_DIR/logs $GPU_DIR/scenes"
rsync -az --delete server/ "$GPU_HOST:$GPU_DIR/server/"
ssh -o BatchMode=yes "$GPU_HOST" "bash -s" <<EOF2
set -euo pipefail
pkill -f "live_server.py --port $port" 2>/dev/null || true
sleep 1
cd $GPU_DIR
source $GPU_ACTIVATE
scenes=""
for scene in $EXTRA_SCENES; do scenes="\$scenes --scene \$scene"; done
# Scans uploaded to scenes/<name>/ are picked up while the server runs.
nohup setsid python server/live_server.py --port $port \$scenes --scenes-dir scenes \
  --gsplat-examples $GSPLAT_EXAMPLES \
  > logs/server.log 2>&1 < /dev/null &
sleep 4
tail -5 logs/server.log
EOF2
