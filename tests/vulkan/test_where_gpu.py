#!/usr/bin/env python3
"""
Regression test for broadcasted bool where on Vulkan.

SDPA's causal masking broadcasts a smaller bool mask across grouped-head score
tensors, so this must stay on GPU under strict fallback mode.
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

        mask_np = np.tril(np.ones((5, 5), dtype=np.bool_))
        true_np = np.arange(1, 1 + 1 * 2 * 4 * 5 * 5, dtype=np.float32).reshape(
            1, 2, 4, 5, 5
        )
        false_np = np.full((1, 2, 4, 5, 5), -1e9, dtype=np.float32)

        mask = mx.array(mask_np)
        true_v = mx.array(true_np.astype(np.float16))
        false_v = mx.array(false_np.astype(np.float16))
        out = mx.where(mask, true_v, false_v)
        mx.eval(out)

        got = np.array(out.tolist(), dtype=np.float32)
        expected = np.where(
            mask_np,
            true_np.astype(np.float16).astype(np.float32),
            false_np.astype(np.float16).astype(np.float32),
        )
        if not np.allclose(got, expected, atol=1e-2, rtol=1e-2):
            raise AssertionError(
                f"broadcasted where mismatch\nexpected:\n{expected}\nactual:\n{got}"
            )
        print(f"  PASS broadcast_bool_where: shape={got.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan where smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
