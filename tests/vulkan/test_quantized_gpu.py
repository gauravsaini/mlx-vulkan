#!/usr/bin/env python3
"""
Strict Vulkan regression for affine quantized matmul.

This covers the real Qwen-style `transpose=True` path with 5-bit weights and
keeps CPU fallback forbidden so the hidden synchronized CPU detour does not
silently pass as "GPU support".
"""

import importlib.util
import os
import sys

import numpy as np


def load_mx():
    core_so = os.environ.get("MLX_CORE_SO")
    if not core_so:
        import mlx.core as mx

        return mx

    import mlx

    spec = importlib.util.spec_from_file_location("mlx.core", core_so)
    mx = importlib.util.module_from_spec(spec)
    sys.modules["mlx.core"] = mx
    spec.loader.exec_module(mx)
    mlx.core = mx
    return mx


def to_numpy(x):
    return np.array(x.tolist(), dtype=np.float32)


def run_case(
    mx,
    name,
    x_np,
    w_np,
    transpose,
    group_size,
    bits,
    atol,
    rtol,
    *,
    x_dtype=None,
    w_dtype=None,
):
    with mx.stream(mx.cpu):
        w_arr = mx.array(w_np, dtype=w_dtype) if w_dtype is not None else mx.array(w_np)
        w_q, scales, biases = mx.quantize(
            w_arr,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        w_dq = mx.dequantize(
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        mx.eval(w_q, scales, biases, w_dq)

    x = mx.array(x_np, dtype=x_dtype) if x_dtype is not None else mx.array(x_np)
    y = mx.quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    mx.eval(y)

    ref = x @ (mx.swapaxes(w_dq, -1, -2) if transpose else w_dq)
    mx.eval(ref)

    got_np = to_numpy(y)
    ref_np = to_numpy(ref)
    if not np.allclose(got_np, ref_np, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch\nexpected:\n{ref_np}\nactual:\n{got_np}"
        )
    print(f"  PASS {name}: shape={got_np.shape}")


def main():
    mx = load_mx()
    if not mx.is_available(mx.gpu):
        print("SKIP: mx.gpu not available")
        return 0

    prev_fail = os.environ.get("MLX_VULKAN_FAIL_ON_CPU_FALLBACK")
    os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = "1"

    try:
        mx.set_default_device(mx.gpu)

        run_case(
            mx,
            "qmm_affine_transpose_5bit_float16",
            (np.arange(1, 1 + 3 * 64, dtype=np.float32).reshape(3, 64) / 32.0).astype(np.float16),
            (np.arange(1, 1 + 48 * 64, dtype=np.float32).reshape(48, 64) / 64.0).astype(np.float16),
            transpose=True,
            group_size=64,
            bits=5,
            atol=8e-2,
            rtol=8e-2,
        )

        run_case(
            mx,
            "qmm_affine_no_transpose_4bit_float32",
            np.arange(1, 1 + 4 * 48, dtype=np.float32).reshape(4, 48) / 24.0,
            np.arange(1, 1 + 48 * 64, dtype=np.float32).reshape(48, 64) / 48.0,
            transpose=False,
            group_size=64,
            bits=4,
            atol=1e-3,
            rtol=1e-3,
        )

        run_case(
            mx,
            "qmm_affine_transpose_5bit_decoder_proj_float16",
            (
                np.arange(1, 1 + 2048, dtype=np.float32).reshape(1, 2048) / 64.0
            ).astype(np.float16),
            (
                np.arange(1, 1 + 512 * 2048, dtype=np.float32).reshape(512, 2048)
                / 256.0
            ).astype(np.float16),
            transpose=True,
            group_size=64,
            bits=5,
            atol=1.2e-1,
            rtol=1.2e-1,
        )

        run_case(
            mx,
            "qmm_affine_transpose_5bit_decoder_mlp_up_float16",
            (
                np.arange(1, 1 + 2048, dtype=np.float32).reshape(1, 2048) / 64.0
            ).astype(np.float16),
            (
                np.arange(1, 1 + 6144 * 2048, dtype=np.float32).reshape(6144, 2048)
                / 256.0
            ).astype(np.float16),
            transpose=True,
            group_size=64,
            bits=5,
            atol=1.5e-1,
            rtol=1.5e-1,
        )

        run_case(
            mx,
            "qmm_affine_transpose_5bit_decoder_mlp_down_float16",
            (
                np.arange(1, 1 + 6144, dtype=np.float32).reshape(1, 6144) / 64.0
            ).astype(np.float16),
            (
                np.arange(1, 1 + 2048 * 6144, dtype=np.float32).reshape(2048, 6144)
                / 256.0
            ).astype(np.float16),
            transpose=True,
            group_size=64,
            bits=5,
            atol=1.5e-1,
            rtol=1.5e-1,
        )

        run_case(
            mx,
            "qmm_affine_transpose_5bit_large_vocab_bfloat16",
            (np.arange(1, 1 + 256, dtype=np.float32).reshape(1, 256) / 64.0),
            (np.arange(1, 1 + 8192 * 256, dtype=np.float32).reshape(8192, 256) / 256.0),
            transpose=True,
            group_size=64,
            bits=5,
            atol=1.5e-1,
            rtol=1.5e-1,
            x_dtype=mx.bfloat16,
            w_dtype=mx.bfloat16,
        )
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan affine quantized matmul smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
