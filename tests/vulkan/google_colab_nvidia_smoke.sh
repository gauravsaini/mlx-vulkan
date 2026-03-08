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

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi not found. Start a GPU runtime first."
    exit 1
fi

echo ""
echo "→ GPU"
nvidia-smi

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
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary || true
else
    echo "vulkaninfo not found after install"
fi

echo ""
echo "→ Building mlx.core with Vulkan enabled"
cd "$MLX_SRC"
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_CPU=ON" \
    "$PYTHON_BIN" setup.py build_ext --inplace

echo ""
echo "→ Running Stage 25 with NVIDIA vendor gate"
cd "$ROOT_DIR"
PYTHONPATH="$MLX_SRC/python" \
MLX_VULKAN_REQUIRE_VENDOR=nvidia \
"$PYTHON_BIN" "$ROOT_DIR/tests/vulkan/test_stage25_linux_vulkan_bringup.py"
