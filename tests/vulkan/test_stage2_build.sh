#!/bin/bash
# Stage 2: CMake Build
# Tests: CMake configure + compile succeeds
# Must run on Linux with Vulkan SDK installed

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_SRC="$SCRIPT_DIR/../../mlx-src"
BUILD_DIR="$SCRIPT_DIR/../../build-vulkan"

echo "═══════════════════════════════════════"
echo "  STAGE 2: CMake Build"
echo "═══════════════════════════════════════"

# Check prerequisites
for cmd in cmake ninja glslc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ Missing: $cmd"
        exit 1
    fi
done
echo "✅ Build tools found"

# Configure
echo ""
echo "→ cmake configure..."
cmake -B "$BUILD_DIR" \
      -G Ninja \
      -DMLX_BUILD_VULKAN=ON \
      -DMLX_BUILD_METAL=OFF \
      -DMLX_BUILD_CUDA=OFF \
      -DMLX_BUILD_CPU=ON \
      -DMLX_BUILD_TESTS=OFF \
      -DMLX_BUILD_EXAMPLES=OFF \
      -DMLX_BUILD_PYTHON_BINDINGS=OFF \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      "$MLX_SRC" 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "❌ STAGE 2 FAIL: CMake configure failed"
    exit 1
fi
echo "✅ CMake configure OK"

# Build
echo ""
echo "→ building..."
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) \
    2>&1 | grep -E "error:|warning:|Linking|Building|Compiling" | tail -50

if [ $? -ne 0 ]; then
    echo "❌ STAGE 2 FAIL: Build failed"
    exit 1
fi

# Check SPIR-V files were generated
SPV_COUNT=$(find "$BUILD_DIR" -name "*.spv" 2>/dev/null | wc -l)
echo ""
echo "✅ SPIR-V shaders compiled: $SPV_COUNT"

if [ "$SPV_COUNT" -eq 0 ]; then
    echo "❌ STAGE 2 FAIL: No .spv files generated"
    exit 1
fi

echo ""
echo "✅ STAGE 2 PASS: Build successful"
echo "   Build dir: $BUILD_DIR"
