#!/bin/bash
# Bootstrap and run the Linux/NVIDIA Vulkan bring-up smoke on a Colab-style VM.
# Usage: bash tests/vulkan/google_colab_nvidia_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MLX_SRC="$ROOT_DIR/mlx-src"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "═══════════════════════════════════════"
echo "  Google Colab NVIDIA Vulkan Smoke"
echo "═══════════════════════════════════════"

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
fi

echo ""
echo "→ Runtime detection"
if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "NVIDIA runtime detected."
    nvidia-smi
else
    echo "No NVIDIA GPU runtime detected."
    echo "Colab note: switch to a GPU runtime if you want the full Vulkan/NVIDIA smoke."
    echo "Continuing with compile/import-only validation on this machine."
fi

echo ""
echo "→ Installing Vulkan/build prerequisites"
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    git \
    glslc \
    libvulkan-dev \
    ninja-build \
    python3-dev \
    python3-pip \
    vulkan-tools

echo ""
echo "→ Python build deps"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel numpy

echo ""
echo "→ Vulkan summary"
if [ "$HAS_NVIDIA" -eq 1 ] && command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary || true
else
    echo "Skipping vulkaninfo summary without an attached NVIDIA runtime"
fi

echo ""
echo "→ Building mlx.core with Vulkan enabled"
cd "$MLX_SRC"
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_CPU=ON" \
    "$PYTHON_BIN" setup.py build_ext --inplace

echo ""
if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "→ Running Stage 25 with NVIDIA vendor gate"
    cd "$ROOT_DIR"
    PYTHONPATH="$MLX_SRC/python" \
    MLX_VULKAN_REQUIRE_VENDOR=nvidia \
    "$PYTHON_BIN" "$ROOT_DIR/tests/vulkan/test_stage25_linux_vulkan_bringup.py"
else
    echo "→ Running compile/import-only probe"
    cd "$ROOT_DIR"
    PYTHONPATH="$MLX_SRC/python" "$PYTHON_BIN" - <<'PY'
import mlx.core as mx

print("mlx.core import OK")
print("mx.is_available(mx.gpu) =", mx.is_available(mx.gpu))
print("mx.default_device() =", mx.default_device())
PY
    echo ""
    echo "Compile/import-only probe passed. Re-run this notebook on a GPU runtime for the full NVIDIA smoke."
fi
