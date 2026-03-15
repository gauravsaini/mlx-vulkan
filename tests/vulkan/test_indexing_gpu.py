#!/usr/bin/env python3
"""
Regression test for eager GPU indexing staying on-device.

Simple slice and integer indexing must not force eager arrays through the
Python indexing CPU materialization path.
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

        x_np = np.arange(1, 1 + 2 * 6 * 4, dtype=np.float16).reshape(2, 6, 4)
        x = mx.array(x_np)
        tail = x[:, -3:]
        mx.eval(tail)
        tail_np = np.array(tail.tolist(), dtype=np.float32)
        expected_tail = x_np[:, -3:].astype(np.float32)
        if not np.allclose(tail_np, expected_tail, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                f"tuple slice mismatch\nexpected:\n{expected_tail}\nactual:\n{tail_np}"
            )
        print(f"  PASS tuple_slice_tail: shape={tail_np.shape}")

        prefix = mx.array(np.full((1, 3, 4), -1.0, dtype=np.float16))
        suffix = mx.array(np.arange(1, 1 + 1 * 2 * 4, dtype=np.float16).reshape(1, 2, 4))
        combined = mx.concatenate([prefix, suffix], axis=1)
        qwen_tail = combined[:, -(4 - 1) :]
        mx.eval(qwen_tail)
        qwen_tail_np = np.array(qwen_tail.tolist(), dtype=np.float32)
        expected_qwen_tail = np.concatenate(
            [
                np.full((1, 3, 4), -1.0, dtype=np.float16),
                np.arange(1, 1 + 1 * 2 * 4, dtype=np.float16).reshape(1, 2, 4),
            ],
            axis=1,
        )[:, -3:].astype(np.float32)
        if not np.allclose(qwen_tail_np, expected_qwen_tail, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                "concatenate tail slice mismatch\n"
                f"expected:\n{expected_qwen_tail}\nactual:\n{qwen_tail_np}"
            )
        print(f"  PASS concatenate_tail_slice: shape={qwen_tail_np.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan indexing smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
