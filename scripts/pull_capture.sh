#!/usr/bin/env bash
# Copy an OpenQuestCapture recording from a USB-connected Quest into captures/.
#
#   scripts/pull_capture.sh            # list recordings on the headset, pull the newest
#   scripts/pull_capture.sh --list     # only list them
#   scripts/pull_capture.sh 20261114_153012
#
# Recordings are folders named YYYYMMDD_hhmmss under the app's files directory.
# Developer mode must be on and the headset must have accepted this computer's
# USB debugging prompt.
set -euo pipefail
# Stop Git Bash on Windows from rewriting /sdcard paths into Windows paths.
export MSYS_NO_PATHCONV=1

remote=/sdcard/Android/data/com.samusynth.OpenQuestCapture/files
dest="$(cd "$(dirname "$0")/.." && pwd)/captures"

adb="${ADB:-}"
if [[ -z "$adb" ]]; then
  if command -v adb >/dev/null; then adb=adb
  elif [[ -n "${LOCALAPPDATA:-}" && -x "$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe" ]]; then
    adb="$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe"
  elif [[ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]]; then
    adb="$HOME/Library/Android/sdk/platform-tools/adb"
  else
    echo "adb not found; install Android platform-tools or set ADB=/path/to/adb" >&2; exit 1
  fi
fi

if ! "$adb" get-state >/dev/null 2>&1; then
  echo "No headset found. Plug in the Quest, put it on, and allow USB debugging." >&2
  "$adb" devices >&2; exit 1
fi

sessions=$("$adb" shell "ls -1 $remote 2>/dev/null" | tr -d '\r' | grep -E '^[0-9]{8}_[0-9]{6}$' | sort || true)
if [[ -z "$sessions" ]]; then
  echo "No recordings in $remote. Record one in OpenQuestCapture first." >&2; exit 1
fi

echo "Recordings on the headset:"
for s in $sessions; do
  echo "  $s  $("$adb" shell "du -sh $remote/$s" | tr -d '\r' | cut -f1)"
done
[[ "${1:-}" == "--list" ]] && exit 0

name="${1:-$(echo "$sessions" | tail -1)}"
if ! grep -qx "$name" <<<"$sessions"; then
  echo "No recording named $name on the headset." >&2; exit 1
fi

mkdir -p "$dest"
echo "Pulling $name -> captures/$name"

"$adb" pull "$remote/$name" "$dest/"

d="$dest/$name"
count() { find "$d/$1" -type f 2>/dev/null | wc -l | tr -d ' '; }
echo
echo "Done: $d"
echo "  left camera frames:  $(count left_camera_raw)"
echo "  right camera frames: $(count right_camera_raw)"
echo "  depth frames:        $(( $(count left_depth) + $(count right_depth) ))"
echo "  head poses:          $(( $(wc -l < "$d/hmd_poses.csv" 2>/dev/null || echo 1) - 1 ))"
