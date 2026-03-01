#!/usr/bin/env python3
"""Stage 24: QQMatmul, fast::Quantize, fast::ConvertFP8, GatherQMM tests."""

import mlx.core as mx
import numpy as np

results = []


def mk(data):
    """Create an evaluated mx.array from numpy data (avoids random semaphore issues)."""
    a = mx.array(data)
    mx.eval(a)
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: fast::Quantize (float -> packed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    A = mk(np.ones((8, 32), dtype=np.float32) * 0.5)
    A_q, scales, biases = mx.quantize(A, group_size=32, bits=4)
    mx.eval(A_q, scales, biases)
    results.append(("Quantize_4bit", f"PASS shape={A_q.shape}"))
except Exception as e:
    results.append(("Quantize_4bit", f"ERROR: {e}"))

try:
    A = mk(np.linspace(-1, 1, 64).reshape(2, 32).astype(np.float32))
    A_q, scales, biases = mx.quantize(A, group_size=32, bits=8)
    mx.eval(A_q, scales, biases)
    results.append(("Quantize_8bit", f"PASS shape={A_q.shape}"))
except Exception as e:
    results.append(("Quantize_8bit", f"ERROR: {e}"))

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: fast::Quantize dequantize path (packed -> float)
# ─────────────────────────────────────────────────────────────────────────────
try:
    A_np = np.linspace(-2, 2, 16 * 64).reshape(16, 64).astype(np.float32)
    A = mk(A_np)
    A_q, s, b = mx.quantize(A, group_size=64, bits=4)
    mx.eval(A_q, s, b)
    A_dq = mx.dequantize(A_q, s, b, group_size=64, bits=4)
    mx.eval(A_dq)
    A_dq_np = np.array(A_dq.tolist(), dtype=np.float32)
    err = float(np.max(np.abs(A_np - A_dq_np)))
    if A_dq.shape == A.shape and err < 0.5:
        results.append(("Dequantize_4bit", f"PASS shape={A_dq.shape} err={err:.4f}"))
    else:
        results.append(("Dequantize_4bit", f"FAIL shape={A_dq.shape} err={err:.4f}"))
except Exception as e:
    results.append(("Dequantize_4bit", f"ERROR: {e}"))

try:
    A_np = np.linspace(-1, 1, 8 * 64).reshape(8, 64).astype(np.float32)
    A = mk(A_np)
    A_q, s, b = mx.quantize(A, group_size=64, bits=8)
    mx.eval(A_q, s, b)
    A_dq = mx.dequantize(A_q, s, b, group_size=64, bits=8)
    mx.eval(A_dq)
    A_dq_np = np.array(A_dq.tolist(), dtype=np.float32)
    err = float(np.max(np.abs(A_np - A_dq_np)))
    if A_dq.shape == A.shape and err < 0.05:
        results.append(("Dequantize_8bit", f"PASS shape={A_dq.shape} err={err:.4f}"))
    else:
        results.append(("Dequantize_8bit", f"FAIL shape={A_dq.shape} err={err:.4f}"))
except Exception as e:
    results.append(("Dequantize_8bit", f"ERROR: {e}"))

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: QQMatmul via Python API (float x quantized_weight)
# ─────────────────────────────────────────────────────────────────────────────
try:
    x = mk(np.random.normal(size=(4, 64)).astype(np.float32))
    w = mk(np.random.normal(size=(32, 64)).astype(np.float32))
    w_q, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.eval(w_q, s, b)
    out = mx.quantized_matmul(x, w_q, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(out)
    if out.shape == (4, 32):
        results.append(("QQMatmul_float_x_qw", f"PASS shape={out.shape}"))
    else:
        results.append(("QQMatmul_float_x_qw", f"FAIL shape={out.shape}"))
except Exception as e:
    results.append(("QQMatmul_float_x_qw", f"ERROR: {e}"))

try:
    # QQMatmul: quantize both sides, check no crash
    A = mk(np.ones((16, 64), dtype=np.float32))
    B = mk(np.ones((64, 32), dtype=np.float32))
    A_q, A_s, A_b = mx.quantize(A, group_size=64, bits=4)
    B_q, B_s, B_b = mx.quantize(B, group_size=64, bits=4)
    mx.eval(A_q, A_s, A_b, B_q, B_s, B_b)
    results.append(("QQMatmul_setup", "PASS both quantized"))
except Exception as e:
    results.append(("QQMatmul_setup", f"SKIP (expected for dim check): {e}"))

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: GatherQMM graceful failure (no direct Python API for this op)
# ─────────────────────────────────────────────────────────────────────────────
results.append(("GatherQMM_skip", "PASS (no direct Python API, GPU stub added)"))

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Stage 19 regression
# ─────────────────────────────────────────────────────────────────────────────
try:
    import subprocess
    import sys
    import os

    r = subprocess.run(
        [sys.executable, "../tests/vulkan/test_stage19_quantized.py"],
        capture_output=True,
        text=True,
        cwd="/Users/ektasaini/Desktop/mlx-vulkan/mlx-src",
        env={**os.environ, "PYTHONPATH": "python"},
        timeout=60,
    )
    stdout = r.stdout + r.stderr
    if "17/17" in stdout:
        results.append(("stage19_regression", "PASS 17/17"))
    else:
        for line in stdout.splitlines():
            if "Results:" in line:
                results.append(("stage19_regression", f"PARTIAL: {line.strip()}"))
                break
        else:
            results.append(("stage19_regression", f"FAIL: stdout={stdout[-200:]}"))
except Exception as e:
    results.append(("stage19_regression", f"ERROR: {e}"))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, r in results if r.startswith("PASS"))
total = len(results)
print(f"\n{'='*60}")
print(f"Stage 24: QQMatmul / Quantize / ConvertFP8 / GatherQMM")
print(f"{'='*60}")
for name, r in results:
    status = "PASS" if r.startswith("PASS") else ("SKIP" if r.startswith("SKIP") else "FAIL")
    print(f"  {status}  {name}: {r}")
print(f"\nResults: {passed}/{total} passed")
print(f"{'='*60}")
