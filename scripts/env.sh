# Shared settings for run.sh, start_jupyter.sh and install.sh (sourced, not executed).
#   REPO          this repository's folder
#   ISAACLAB_DIR  the Isaac Lab checkout: $ISAACLAB_DIR if set, else ./IsaacLab, else ../IsaacLab
#   ISAACLAB_COMMIT  the Isaac Lab commit this tutorial was tested with

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_COMMIT="3e7e6137089d1db8823f3011941ab1ee73cc9954"  # Isaac Lab main, 2026-11-16 (v3.0.0)

if [ -z "${ISAACLAB_DIR:-}" ]; then
  if [ -d "$REPO/IsaacLab" ]; then
    ISAACLAB_DIR="$REPO/IsaacLab"
  elif [ -d "$REPO/../IsaacLab" ]; then
    ISAACLAB_DIR="$(cd "$REPO/../IsaacLab" && pwd)"
  else
    ISAACLAB_DIR="$REPO/IsaacLab"  # where install.sh will put it
  fi
fi
export ISAACLAB_DIR

# uv installs itself into ~/.local/bin
if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

# WSL only: draw OpenGL viewer windows on the NVIDIA GPU (through Direct3D 12) instead of the CPU
# software renderer "llvmpipe", which is ~6x too slow for real time. Native Linux uses the NVIDIA driver directly.
if grep -qi microsoft /proc/version 2>/dev/null; then
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi
