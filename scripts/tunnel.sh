#!/usr/bin/env bash
# Forward this Mac's port 8765 to the live training server on the GPU machine.
#
# Usage: scripts/tunnel.sh [host]     (an SSH alias; defaults to $HOST, then autodl_pro6000)
#
# This uses the shared SSH connection (ControlMaster) to the machine and then holds it open
# for as long as this script runs. Ctrl-C to stop. With key login the connection is opened
# here; a machine with password login needs an interactive `ssh <host>` first.
host="${1:-${HOST:-autodl_pro6000}}"
if ! ssh -O check "$host" 2>/dev/null && ! ssh -o BatchMode=yes "$host" true 2>/dev/null; then
  echo "Cannot reach $host without a password. Run 'ssh $host' once in a terminal" >&2
  echo "(it asks for the password), leave it open, then run this script again." >&2
  exit 1
fi
ssh -O cancel -L 8765:127.0.0.1:8765 "$host" 2>/dev/null
ssh -O forward -L 8765:127.0.0.1:8765 "$host" || exit 1
echo "Forwarding localhost:8765 -> $host:8765"
exec ssh "$host" 'while true; do sleep 30; done'
