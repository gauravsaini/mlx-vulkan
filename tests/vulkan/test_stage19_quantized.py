#!/usr/bin/env python3
"""
Stage 19: QuantizedMatmul GPU dispatch test

MLX Vulkan backend limitations:
  - fast::Quantize (used by mx.quantize/mx.dequantize) is NO_GPU_MULTI
    -> must run mx.quantize and mx.dequantize on CPU stream
  - mx.quantized_matmul dispatches QuantizedMatmul::eval_gpu (our new impl)
    -> runs on GPU stream

Quantize conventions:
  w is [N_out, K_in], groups along K_in (last dim, must be divisible by group_size)
  w_q: [N_out, K_in * bits / 32]    packed uint32
  scales/biases: [N_out, K_in / group_size]
  mx.quantized_matmul(x, w_q, scales, biases, transpose=True) -> x @ w^T

Our GPU implementation:
  transpose_=True: DEQUANT_AFFINE_TRANS (op=2) -> dq[K,N] -> Matmul x[M,K] @ dq[K,N]
  transpose_=False: DEQUANT_AFFINE (op=1) -> dq[K,N] -> Matmul x[M,K] @ dq[K,N]
"""

import sys
import traceback
import mlx.core as mx

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  {msg}")
        FAIL += 1

def run_test(name, fn):
    global FAIL
    try:
        fn()
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        traceback.print_exc()
        FAIL += 1

def cpu_quantize(w, group_size, bits):
    """Quantize on CPU stream (fast::Quantize has no Vulkan GPU path)."""
    with mx.stream(mx.cpu):
        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(w_q, scales, biases)
    return w_q, scales, biases

def cpu_dequantize(w_q, scales, biases, group_size, bits):
    """Dequantize on CPU stream."""
    with mx.stream(mx.cpu):
        w_dq = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=bits)
        mx.eval(w_dq)
    return w_dq

def cpu_matmul(a, b):
    """Matrix multiply on CPU stream."""
    with mx.stream(mx.cpu):
        out = mx.matmul(a, b)
        mx.eval(out)
    return out


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_quantize_shapes_4bit():
    """Verify mx.quantize produces expected packed/scale shapes."""
    N, K = 64, 128
    group_size, bits = 64, 4
    elems_per_word = 32 // bits  # = 8

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)

    check("shapes_4bit/w_q", list(w_q.shape) == [N, K // elems_per_word],
          f"expected [{N},{K//elems_per_word}] got {w_q.shape}")
    check("shapes_4bit/scales", list(scales.shape) == [N, K // group_size],
          f"expected [{N},{K//group_size}] got {scales.shape}")
    check("shapes_4bit/dtype", w_q.dtype == mx.uint32,
          f"expected uint32 got {w_q.dtype}")


def test_quantize_shapes_8bit():
    """Verify mx.quantize shapes for 8-bit."""
    N, K = 32, 128
    group_size, bits = 64, 8
    elems_per_word = 32 // bits  # = 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)

    check("shapes_8bit/w_q", list(w_q.shape) == [N, K // elems_per_word],
          f"expected [{N},{K//elems_per_word}] got {w_q.shape}")


def test_dequantize_roundtrip_4bit():
    """Quantize then dequantize should approximate original."""
    N, K = 64, 128
    group_size, bits = 64, 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)

    check("dequant_4bit/shape", list(w_dq.shape) == [N, K],
          f"expected [{N},{K}] got {w_dq.shape}")
    diff = mx.max(mx.abs(w_dq - w)).item()
    check("dequant_4bit/accuracy", diff < 0.5, f"max_diff={diff:.4f}")


def test_dequantize_roundtrip_8bit():
    """Quantize + dequantize for 8-bit."""
    N, K = 32, 128
    group_size, bits = 64, 8

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)

    check("dequant_8bit/shape", list(w_dq.shape) == [N, K],
          f"expected [{N},{K}] got {w_dq.shape}")
    diff = mx.max(mx.abs(w_dq - w)).item()
    check("dequant_8bit/accuracy", diff < 0.05, f"max_diff={diff:.4f}")


def test_quantized_matmul_4bit():
    """GPU quantized_matmul (transpose=True, 4-bit) vs CPU float reference."""
    mx.set_default_device(mx.gpu)
    N, K, M = 64, 128, 4
    group_size, bits = 64, 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    x = mx.random.normal(shape=[M, K])
    mx.eval(x)

    # GPU path: QuantizedMatmul::eval_gpu with DEQUANT_AFFINE_TRANS
    gpu_out = mx.quantized_matmul(x, w_q, scales, biases,
                                   transpose=True,
                                   group_size=group_size,
                                   bits=bits)
    mx.eval(gpu_out)

    # CPU reference: dequantize w_q -> [N, K], then x[M,K] @ w[N,K]^T -> [M,N]
    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)
    ref = cpu_matmul(x, w_dq.T)

    check("qmm_4bit/shape", list(gpu_out.shape) == [M, N],
          f"expected [{M},{N}] got {gpu_out.shape}")
    diff = mx.max(mx.abs(gpu_out - ref)).item()
    check("qmm_4bit/numerics", diff < 1e-2, f"max_diff={diff:.6f}")


def test_quantized_matmul_8bit():
    """GPU quantized_matmul (transpose=True, 8-bit)."""
    mx.set_default_device(mx.gpu)
    N, K, M = 32, 128, 2
    group_size, bits = 64, 8

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    x = mx.random.normal(shape=[M, K])
    mx.eval(x)

    gpu_out = mx.quantized_matmul(x, w_q, scales, biases,
                                   transpose=True,
                                   group_size=group_size,
                                   bits=bits)
    mx.eval(gpu_out)

    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)
    ref = cpu_matmul(x, w_dq.T)

    check("qmm_8bit/shape", list(gpu_out.shape) == [M, N],
          f"expected [{M},{N}] got {gpu_out.shape}")
    diff = mx.max(mx.abs(gpu_out - ref)).item()
    check("qmm_8bit/numerics", diff < 1e-2, f"max_diff={diff:.6f}")


def test_quantized_matmul_larger():
    """GPU quantized_matmul with larger dimensions."""
    mx.set_default_device(mx.gpu)
    N, K, M = 128, 256, 16
    group_size, bits = 64, 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    x = mx.random.normal(shape=[M, K])
    mx.eval(x)

    gpu_out = mx.quantized_matmul(x, w_q, scales, biases,
                                   transpose=True,
                                   group_size=group_size,
                                   bits=bits)
    mx.eval(gpu_out)

    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)
    ref = cpu_matmul(x, w_dq.T)

    check("qmm_larger/shape", list(gpu_out.shape) == [M, N],
          f"expected [{M},{N}] got {gpu_out.shape}")
    diff = mx.max(mx.abs(gpu_out - ref)).item()
    check("qmm_larger/numerics", diff < 1e-2, f"max_diff={diff:.6f}")


def test_qmm_vs_float_matmul():
    """GPU quantized_matmul vs float matmul — error bounded by quantization."""
    mx.set_default_device(mx.gpu)
    N, K, M = 64, 128, 8
    group_size, bits = 64, 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    x = mx.random.normal(shape=[M, K])
    mx.eval(x)

    gpu_out = mx.quantized_matmul(x, w_q, scales, biases,
                                   transpose=True,
                                   group_size=group_size,
                                   bits=bits)
    mx.eval(gpu_out)

    # Float reference (original unquantized weights)
    ref = cpu_matmul(x, w.T)

    # Quantization error for 4-bit affine on unit-normal data:
    # typical max diff < 2.0 for M=8, N=64, K=128
    diff = mx.max(mx.abs(gpu_out - ref)).item()
    check("qmm_vs_float/diff_bounded", diff < 5.0,
          f"max_diff={diff:.4f}")


def test_group_size_32():
    """Test with group_size=32 (smaller group, more scale params)."""
    mx.set_default_device(mx.gpu)
    N, K, M = 32, 128, 4
    group_size, bits = 32, 4

    w = mx.random.normal(shape=[N, K])
    mx.eval(w)
    w_q, scales, biases = cpu_quantize(w, group_size, bits)
    x = mx.random.normal(shape=[M, K])
    mx.eval(x)

    gpu_out = mx.quantized_matmul(x, w_q, scales, biases,
                                   transpose=True,
                                   group_size=group_size,
                                   bits=bits)
    mx.eval(gpu_out)

    w_dq = cpu_dequantize(w_q, scales, biases, group_size, bits)
    ref = cpu_matmul(x, w_dq.T)

    check("gs32/shape", list(gpu_out.shape) == [M, N],
          f"expected [{M},{N}] got {gpu_out.shape}")
    diff = mx.max(mx.abs(gpu_out - ref)).item()
    check("gs32/numerics", diff < 1e-2, f"max_diff={diff:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 19: QuantizedMatmul GPU Tests")
    print("=" * 60)

    try:
        mx.set_default_device(mx.gpu)
        t = mx.ones([2, 2])
        mx.eval(t)
        print(f"GPU: OK ({mx.default_device()})")
    except Exception as e:
        print(f"GPU not available: {e}")
        sys.exit(1)

    run_test("quantize_shapes_4bit",        test_quantize_shapes_4bit)
    run_test("quantize_shapes_8bit",        test_quantize_shapes_8bit)
    run_test("dequantize_roundtrip_4bit",   test_dequantize_roundtrip_4bit)
    run_test("dequantize_roundtrip_8bit",   test_dequantize_roundtrip_8bit)
    run_test("quantized_matmul_4bit",       test_quantized_matmul_4bit)
    run_test("quantized_matmul_8bit",       test_quantized_matmul_8bit)
    run_test("quantized_matmul_larger",     test_quantized_matmul_larger)
    run_test("qmm_vs_float_matmul",         test_qmm_vs_float_matmul)
    run_test("group_size_32",               test_group_size_32)

    print()
    print("=" * 60)
    print(f"Results: {PASS}/{PASS + FAIL} passed")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
