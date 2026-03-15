#!/usr/bin/env python3
"""
Regression test for eager GPU slice updates.

This matches the KV-cache write pattern used by mlx-lm:
`cache[..., prev:offset, :] = update`.
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

        cache = mx.zeros((1, 8, 256, 16), dtype=mx.float16)
        update_np = (
            np.arange(1, 1 + 1 * 8 * 1 * 16, dtype=np.float32).reshape(1, 8, 1, 16)
            / 32.0
        ).astype(np.float16)
        update = mx.array(update_np)

        cache[..., 0:1, :] = update
        mx.eval(cache)

        got = np.array(cache[..., 0:1, :].tolist(), dtype=np.float32)
        expected = update_np.astype(np.float32)
        if not np.allclose(got, expected, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                f"slice update mismatch\nexpected:\n{expected}\nactual:\n{got}"
            )

        untouched = np.array(cache[..., 1:2, :].tolist(), dtype=np.float32)
        if not np.allclose(untouched, 0.0, atol=1e-5, rtol=1e-5):
            raise AssertionError(f"slice update clobbered neighbors:\n{untouched}")

        print(f"  PASS kv_cache_slice_update: shape={got.shape}")

        noncontig_base = mx.array(
            (
                np.arange(1, 1 + 1 * 1 * 8 * 16, dtype=np.float32).reshape(1, 1, 8, 16)
                / 64.0
            ).astype(np.float16)
        )
        noncontig_update = noncontig_base.transpose(0, 2, 1, 3)
        cache_nc = mx.zeros((1, 8, 256, 16), dtype=mx.float16)
        cache_nc[..., 1:2, :] = noncontig_update
        mx.eval(cache_nc)

        got_nc = np.array(cache_nc[..., 1:2, :].tolist(), dtype=np.float32)
        expected_nc = np.array(noncontig_update.tolist(), dtype=np.float32)
        if not np.allclose(got_nc, expected_nc, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                f"non-contiguous slice update mismatch\nexpected:\n{expected_nc}\nactual:\n{got_nc}"
            )
        print(f"  PASS kv_cache_slice_update_noncontig: shape={got_nc.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan slice update smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
