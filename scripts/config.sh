# Sourced by the shell scripts: where the GPU machine is and where the project lives on it.
#
# Your own values go in scripts/local.env, which is not tracked. Copy local.env.example.
# Anything already set in the environment wins over the file.
#
#   GPU_HOST         SSH alias of the GPU machine (required)
#   GPU_DIR          the project folder there; relative to the home directory, or absolute
#   GPU_ACTIVATE     a script that activates the Python environment, relative to GPU_DIR
#   GSPLAT_EXAMPLES  gsplat's examples/ folder, relative to GPU_DIR
#   EXTRA_SCENES     more COLMAP workspaces for the server: "name=/path name2=/path2"
config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$config_dir/local.env" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ''|\#*) continue ;; esac
    value="${value%\"}"; value="${value#\"}"
    if [ -z "${!key:-}" ]; then export "$key=$value"; fi
  done < "$config_dir/local.env"
fi
GPU_HOST="${GPU_HOST:-${HOST:-}}"
GPU_DIR="${GPU_DIR:-live-splat}"
GPU_ACTIVATE="${GPU_ACTIVATE:-activate.sh}"
GSPLAT_EXAMPLES="${GSPLAT_EXAMPLES:-gsplat/examples}"
EXTRA_SCENES="${EXTRA_SCENES:-}"

need_gpu_host() {
  if [ -z "$GPU_HOST" ]; then
    echo "No GPU machine set. Put GPU_HOST=<ssh alias> in scripts/local.env" >&2
    echo "(see scripts/local.env.example)." >&2
    exit 1
  fi
}
