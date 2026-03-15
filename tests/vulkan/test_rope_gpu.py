#!/usr/bin/env python3
"""
Regression test for Vulkan RoPE materialization.

This matches the key-cache path where keys are normalized, transposed and then
materialized for cache update after rotary embedding.
"""

import importlib.util
import os
import sys

import numpy as np


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
        mx.set_default_device(mx.gpu)

        x = mx.array(
            (np.arange(1, 1 + 1 * 5 * 4 * 16, dtype=np.float32).reshape(1, 5, 4, 16) / 128.0).astype(np.float16)
        )
        keys = x.transpose(0, 2, 1, 3)
        out = mx.fast.rope(
            keys,
            16,
            traditional=False,
            base=10000.0,
            scale=1.0,
            offset=7,
        )
        mx.eval(out)
        out_np = np.array(out.tolist(), dtype=np.float32)
        if out_np.shape != (1, 4, 5, 16):
            raise AssertionError(f"unexpected RoPE shape: {out_np.shape}")
        print(f"  PASS rope_eval: shape={out_np.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan RoPE smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
