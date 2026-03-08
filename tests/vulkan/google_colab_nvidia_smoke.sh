#!/bin/bash
# Bootstrap and run the Linux/NVIDIA Vulkan bring-up smoke on a Colab-style VM.
# Usage: bash tests/vulkan/google_colab_nvidia_smoke.sh
# Optional:
#   MLX_COLAB_TRACE=1  Enable shell xtrace for deeper debugging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MLX_SRC="$ROOT_DIR/mlx-src"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MLX_COLAB_TRACE="${MLX_COLAB_TRACE:-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
VULKAN_HAS_LLVMPIPE=0

if [ "$MLX_COLAB_TRACE" = "1" ]; then
    set -x
fi

section() {
    echo ""
    echo "→ $1"
}

run_cmd() {
    echo "+ $*"
    "$@"
}

echo "═══════════════════════════════════════"
echo "  Google Colab NVIDIA Vulkan Smoke"
echo "═══════════════════════════════════════"

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
fi

section "Runtime detection"
if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "NVIDIA runtime detected."
    run_cmd nvidia-smi
else
    echo "No NVIDIA GPU runtime detected."
    echo "Colab note: switch to a GPU runtime if you want the full Vulkan/NVIDIA smoke."
    echo "Continuing with compile/import-only validation on this machine."
fi

section "Environment summary"
echo "ROOT_DIR=$ROOT_DIR"
echo "MLX_SRC=$MLX_SRC"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
run_cmd uname -a
run_cmd "$PYTHON_BIN" --version

section "Installing Vulkan/build prerequisites"
run_cmd apt-get update
run_cmd apt-get install -y \
    build-essential \
    cmake \
    git \
    glslang-tools \
    liblapack-dev \
    liblapacke-dev \
    libvulkan-dev \
    ninja-build \
    spirv-tools \
    python3-dev \
    python3-pip \
    vulkan-tools

if ! command -v glslc >/dev/null 2>&1; then
    if command -v glslangValidator >/dev/null 2>&1; then
        echo ""
        echo "→ Creating glslc compatibility wrapper from glslangValidator"
        cat >/usr/local/bin/glslc <<'EOF'
#!/bin/bash
set -euo pipefail

args=()
shader_stage=""
target_env=""
input=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target-env=*)
            target_env="${1#*=}"
            shift
            ;;
        -fshader-stage=compute)
            shader_stage="comp"
            shift
            ;;
        -o)
            args+=("$1")
            shift
            args+=("$1")
            shift
            ;;
        -I*|-D*)
            args+=("$1")
            shift
            ;;
        -*)
            args+=("$1")
            shift
            ;;
        *)
            input="$1"
            shift
            ;;
    esac
done

if [ -z "$input" ]; then
    echo "glslc wrapper: missing input shader" >&2
    exit 1
fi

cmd=(glslangValidator -V)
if [ -n "$target_env" ]; then
    cmd+=(--target-env "$target_env")
fi
if [ -n "$shader_stage" ]; then
    cmd+=(-S "$shader_stage")
fi
cmd+=("${args[@]}" "$input")
"${cmd[@]}"
EOF
        chmod +x /usr/local/bin/glslc
    else
        echo "FAIL: neither glslc nor glslangValidator is available after package install"
        exit 1
    fi
fi

section "Toolchain summary"
run_cmd which "$PYTHON_BIN"
run_cmd cmake --version
run_cmd gcc --version
run_cmd g++ --version
run_cmd git --version
run_cmd glslc --version
if command -v glslangValidator >/dev/null 2>&1; then
    run_cmd glslangValidator --version
fi
run_cmd "$PYTHON_BIN" -m pip --version
echo "+ Vulkan ICD files"
find /usr/share/vulkan /etc/vulkan -maxdepth 3 -type f \( -name '*.json' -o -name '*.icd' \) 2>/dev/null | sort || true

section "Python build deps"
run_cmd "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel numpy

section "Vulkan summary"
if [ "$HAS_NVIDIA" -eq 1 ] && command -v vulkaninfo >/dev/null 2>&1; then
    VULKAN_SUMMARY="$(vulkaninfo --summary 2>&1 || true)"
    printf '%s\n' "$VULKAN_SUMMARY"
    if printf '%s\n' "$VULKAN_SUMMARY" | grep -qi "llvmpipe"; then
        VULKAN_HAS_LLVMPIPE=1
        echo ""
        echo "WARNING: Vulkan is currently exposing llvmpipe, not the NVIDIA GPU."
        echo "The Colab runtime has an NVIDIA CUDA device, but not a usable NVIDIA Vulkan ICD."
        echo "The script will stop at compile/import validation instead of running the Stage 25 NVIDIA gate."
    fi
else
    echo "Skipping vulkaninfo summary without an attached NVIDIA runtime"
fi

section "Building mlx.core with Vulkan enabled"
cd "$MLX_SRC"
echo "+ CMAKE_ARGS=-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_CPU=ON $PYTHON_BIN setup.py build_ext --inplace"
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_CPU=ON" \
    "$PYTHON_BIN" setup.py build_ext --inplace
echo "+ Built python artifacts"
find "$MLX_SRC/build" "$MLX_SRC/python" -maxdepth 4 -type f \( -name 'core*.so' -o -name 'core*.pyd' -o -name 'core*.dylib' \) 2>/dev/null | sort || true

if [ "$HAS_NVIDIA" -eq 1 ] && [ "$VULKAN_HAS_LLVMPIPE" -eq 0 ]; then
    section "Running Stage 25 with NVIDIA vendor gate"
    cd "$ROOT_DIR"
    echo "+ PYTHONPATH=$MLX_SRC/python MLX_VULKAN_REQUIRE_VENDOR=nvidia $PYTHON_BIN $ROOT_DIR/tests/vulkan/test_stage25_linux_vulkan_bringup.py"
    PYTHONPATH="$MLX_SRC/python" \
    MLX_VULKAN_REQUIRE_VENDOR=nvidia \
    "$PYTHON_BIN" "$ROOT_DIR/tests/vulkan/test_stage25_linux_vulkan_bringup.py"
else
    section "Running compile/import-only probe"
    cd "$ROOT_DIR"
    PYTHONPATH="$MLX_SRC/python" "$PYTHON_BIN" - <<'PY'
import mlx.core as mx

print("mlx.core import OK")
print("mx.is_available(mx.gpu) =", mx.is_available(mx.gpu))
print("mx.default_device() =", mx.default_device())
PY
    echo ""
    if [ "$HAS_NVIDIA" -eq 1 ] && [ "$VULKAN_HAS_LLVMPIPE" -eq 1 ]; then
        echo "Compile/import-only probe passed, but Vulkan is still routed to llvmpipe."
        echo "This Colab image is not currently usable for the NVIDIA Vulkan smoke."
    else
        echo "Compile/import-only probe passed. Re-run this notebook on a GPU runtime for the full NVIDIA smoke."
    fi
fi
