#!/bin/bash
# MLX Vulkan Backend - Master Test Runner
# Runs all stages in order and stops at first failure
# Usage: ./tests/vulkan/run_all_stages.sh [--skip-build] [--stage N]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_BUILD=0
START_STAGE=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=1 ;;
        --stage) START_STAGE="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

PASS=0
FAIL=0
SKIP=0

stage_run() {
    local stage_num="$1"
    local stage_name="$2"
    local cmd="$3"

    if [ "$stage_num" -lt "$START_STAGE" ]; then
        echo "⏭️  Stage $stage_num skipped (--stage $START_STAGE)"
        ((SKIP++))
        return
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Stage $stage_num: $stage_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$cmd"; then
        ((PASS++))
        echo "✅ Stage $stage_num PASSED"
    else
        ((FAIL++))
        echo "❌ Stage $stage_num FAILED"
        echo ""
        echo "Fix Stage $stage_num before proceeding."
        echo "Resume from here with: $0 --stage $stage_num"
        exit 1
    fi
}

echo "╔═══════════════════════════════════════╗"
echo "║  MLX Vulkan Backend Test Runner       ║"
echo "║  Starting from Stage $START_STAGE                  ║"
echo "╚═══════════════════════════════════════╝"

# Infrastructure stages (can run without Linux GPU on macOS dev)
stage_run 1 "Vulkan Device Detection" \
    "bash '$SCRIPT_DIR/test_stage1_device.sh'"

if [ "$SKIP_BUILD" -eq 0 ]; then
    stage_run 2 "CMake Build" \
        "bash '$SCRIPT_DIR/test_stage2_build.sh'"
fi

stage_run 3 "SPIR-V Shader Compilation" \
    "bash '$SCRIPT_DIR/test_stage3_shaders.sh'"

# Python stages (require mlx installed)
stage_run 6 "MLX CPU Stream Sanity" \
    "python '$SCRIPT_DIR/test_stage6_mlx_cpu.py'"

stage_run 7 "GPU Stream Copy" \
    "python '$SCRIPT_DIR/test_stage7_gpu_copy.py'"

stage_run 8 "Unary GPU Ops" \
    "python '$SCRIPT_DIR/test_stage8_unary.py'"

stage_run 9 "Binary GPU Ops" \
    "python '$SCRIPT_DIR/test_stage9_binary.py'"

stage_run 10 "Reduction GPU Ops" \
    "python '$SCRIPT_DIR/test_stage10_reduce.py'"

stage_run 11 "Matmul GPU Op" \
    "python '$SCRIPT_DIR/test_stage11_matmul.py'"

stage_run 12 "Neural Net GPU Ops" \
    "python '$SCRIPT_DIR/test_stage12_nn_ops.py'"

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  Final Results                        ║"
printf  "║  ✅ Passed: %-27s║\n" "$PASS"
printf  "║  ❌ Failed: %-27s║\n" "$FAIL"
printf  "║  ⏭️  Skipped: %-26s║\n" "$SKIP"
echo "╚═══════════════════════════════════════╝"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo ""
echo "🎉 ALL STAGES PASSED"
