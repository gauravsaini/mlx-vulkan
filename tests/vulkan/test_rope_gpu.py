#!/usr/bin/env python3
"""Regression tests for Vulkan RoPE materialization."""

import importlib.util
import os
import sys

import numpy as np


def rope_reference_np(x_np, dims, offset, base=10000.0, scale=1.0):
    half_dims = dims // 2
    positions = (offset + np.arange(x_np.shape[-2], dtype=np.float32)) * scale
    inv_freqs = np.exp(
        -np.arange(half_dims, dtype=np.float32) * (np.log(base) / half_dims)
    )
    theta = positions[:, None] * inv_freqs[None, :]
    costheta = np.cos(theta)[None, None, :, :]
    sintheta = np.sin(theta)[None, None, :, :]

    x = x_np.astype(np.float32)
    x1 = x[..., :half_dims]
    x2 = x[..., half_dims:dims]
    out = np.empty_like(x, dtype=np.float32)
    out[..., :half_dims] = x1 * costheta - x2 * sintheta
    out[..., half_dims:dims] = x1 * sintheta + x2 * costheta
    if dims < x.shape[-1]:
        out[..., dims:] = x[..., dims:]
    return out


def main():
    core_so = os.environ.get("MLX_CORE_SO")
    if core_so:
        import mlx

        spec = importlib.util.spec_from_file_location("mlx.core", core_so)
        mx = importlib.util.module_from_spec(spec)
        sys.modules["mlx.core"] = mx
        spec.loader.exec_module(mx)
        mlx.core = mx
    else:
        import mlx.core as mx

    if not mx.is_available(mx.gpu):
        print("SKIP: mx.gpu not available")
        return 0

    prev_fail = os.environ.get("MLX_VULKAN_FAIL_ON_CPU_FALLBACK")
    os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = "1"

    try:
        cases = [
            (
                "transpose_prefill",
                (np.arange(1, 1 + 1 * 5 * 4 * 16, dtype=np.float32).reshape(1, 5, 4, 16) / 128.0).astype(np.float16),
                16,
                7,
            ),
            (
                "decode_scalar_offset",
                (0.25 * np.arange(1, 1 + 1 * 1 * 8 * 256, dtype=np.float32).reshape(1, 1, 8, 256) / 256.0).astype(np.float16),
                256,
                11,
            ),
        ]

        for name, x_np, dims, offset in cases:
            expected = rope_reference_np(
                np.transpose(x_np.astype(np.float32), (0, 2, 1, 3)),
                dims,
                offset,
            )

            mx.set_default_device(mx.gpu)
            x_gpu = mx.array(x_np)
            keys_gpu = x_gpu.transpose(0, 2, 1, 3)
            gpu_out = mx.fast.rope(
                keys_gpu,
                dims,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=offset,
            )
            mx.eval(gpu_out)
            got = np.array(gpu_out.tolist(), dtype=np.float32)

            if not np.allclose(got, expected, atol=2e-2, rtol=2e-2):
                raise AssertionError(
                    f"{name} mismatch\nexpected:\n{expected}\nactual:\n{got}"
                )
            print(f"  PASS {name}: shape={got.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan RoPE smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
