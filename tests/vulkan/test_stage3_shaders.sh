#!/bin/bash
# Stage 3: SPIR-V Shader Compilation
# Tests: all .comp shaders compile to .spv and pass spirv-val
# Requires: glslc, spirv-val

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADER_DIR="$SCRIPT_DIR/../../mlx-src/mlx/backend/vulkan/kernels"
OUTPUT_DIR="/tmp/mlx_vulkan_spv"

echo "═══════════════════════════════════════"
echo "  STAGE 3: SPIR-V Shader Compilation"
echo "═══════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

PASS=0
FAIL=0
FAILED_SHADERS=()

for SHADER in "$SHADER_DIR"/*.comp; do
    NAME=$(basename "$SHADER" .comp)
    OUTPUT="$OUTPUT_DIR/${NAME}.spv"

    printf "  Compiling %-20s ... " "$NAME"

    # Compile with glslc
    if glslc --target-env=vulkan1.2 \
             -fshader-stage=compute \
             -I"$SHADER_DIR" \
             -o "$OUTPUT" \
             "$SHADER" 2>/tmp/glslc_err.txt; then

        # Validate with spirv-val if available
        if command -v spirv-val &>/dev/null; then
            if spirv-val "$OUTPUT" 2>/tmp/spirv_err.txt; then
                echo "✅ (${NAME}.spv)"
                ((PASS++))
            else
                echo "❌ INVALID SPIR-V"
                cat /tmp/spirv_err.txt
                FAILED_SHADERS+=("$NAME (invalid)")
                ((FAIL++))
            fi
        else
            echo "✅ (no spirv-val, compiled OK)"
            ((PASS++))
        fi
    else
        echo "❌ COMPILE ERROR"
        cat /tmp/glslc_err.txt
        FAILED_SHADERS+=("$NAME (compile error)")
        ((FAIL++))
    fi
done

echo ""
echo "═══════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"

if [ ${#FAILED_SHADERS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed shaders:"
    for f in "${FAILED_SHADERS[@]}"; do
        echo "    - $f"
    done
    echo ""
    echo "❌ STAGE 3 FAIL"
    exit 1
else
    echo ""
    echo "✅ STAGE 3 PASS: All shaders compiled to SPIR-V"
    echo "   Output: $OUTPUT_DIR"
fi
