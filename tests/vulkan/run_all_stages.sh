#!/bin/bash
# MLX Vulkan Backend - Master Test Runner
# Runs all stages in order and stops at first failure
# Usage: ./tests/vulkan/run_all_stages.sh [--skip-build] [--stage N]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_BUILD=0
START_STAGE=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
        SKIP=$((SKIP + 1))
        return
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Stage $stage_num: $stage_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$cmd"; then
        PASS=$((PASS + 1))
        echo "✅ Stage $stage_num PASSED"
    else
        FAIL=$((FAIL + 1))
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
echo "Using Python: $PYTHON_BIN"

# Infrastructure stages (can run without Linux GPU on macOS dev)
stage_run 1 "Vulkan Device Detection" \
    "bash '$SCRIPT_DIR/test_stage1_device.sh'"

if [ "$SKIP_BUILD" -eq 0 ]; then
    stage_run 2 "CMake Build" \
        "bash '$SCRIPT_DIR/test_stage2_build.sh'"
fi

stage_run 3 "SPIR-V Shader Compilation" \
    "bash '$SCRIPT_DIR/test_stage3_shaders.sh'"

stage_run 3 "Allocator Validation" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage3a_allocator.py'"

stage_run 4 "Unified Memory / Host Transfers" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage4_unified_mem.py'"

stage_run 5 "Workgroup Tuning / Subgroup Info" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage5_workgroup_tune.py'"

# Python stages (require mlx installed)
stage_run 6 "MLX CPU Stream Sanity" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage6_mlx_cpu.py'"

stage_run 7 "GPU Stream Copy" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage7_gpu_copy.py'"

stage_run 8 "Unary GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage8_unary.py'"

stage_run 9 "Binary GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage9_binary.py'"

stage_run 10 "Reduction GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage10_reduce.py'"

stage_run 11 "Matmul GPU Op" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage11_matmul.py'"

stage_run 12 "Neural Net GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage12_nn_ops.py'"
stage_run 13 "Indexing GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage13_indexing.py'"

stage_run 14 "Sorting GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage14_sort.py'"

stage_run 15 "Scan Prefix GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage15_scan.py'"

stage_run 16 "NN Extended GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage16_nn_extended.py'"

stage_run 17 "FFT GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage17_fft.py'"

stage_run 17 "AddMM/Conv/Random GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage17_addmm_conv_rbits.py'"

stage_run 18 "Concatenate GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage18_concat.py'"

stage_run 19 "Quantized GPU Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage19_quantized.py'"

stage_run 20 "Linalg CPU Fallbacks on GPU Stream" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage20_linalg.py'"

stage_run 21 "Advanced Matmul Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage21_advanced_mm.py'"

stage_run 22 "Event / Fence Synchronization" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage22_sync.py'"

stage_run 23 "Shape / Misc Ops" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage23_shape_misc.py'"

stage_run 24 "QQMatmul / Quantize / GatherQMM" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage24_qqmatmul.py'"

stage_run 24 "Subgroup / Workgroup Validation" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage24_subgroup.py'"

stage_run 25 "Linux Vulkan Bring-up Smoke" \
    "'$PYTHON_BIN' '$SCRIPT_DIR/test_stage25_linux_vulkan_bringup.py'"
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
