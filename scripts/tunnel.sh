#!/usr/bin/env bash
# Forward this Mac's port 8765 to the live training server on the GPU machine.
#
# The GPU machine uses password login, so this reuses the shared SSH connection
# (ControlMaster) that an interactive `ssh autodl_3080TI` opened, and then holds that
# connection open for as long as this script runs. Ctrl-C to stop.
host="${1:-autodl_3080TI}"
if ! ssh -O check "$host" 2>/dev/null; then
  echo "No shared SSH connection to $host. Run 'ssh $host' once in a terminal" >&2
  echo "(it asks for the password), leave it open, then run this script again." >&2
  exit 1
fi
ssh -O cancel -L 8765:127.0.0.1:8765 "$host" 2>/dev/null
ssh -O forward -L 8765:127.0.0.1:8765 "$host" || exit 1
echo "Forwarding localhost:8765 -> $host:8765"
exec ssh "$host" 'while true; do sleep 30; done'
