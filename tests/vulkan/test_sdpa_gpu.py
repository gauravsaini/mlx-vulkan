#!/usr/bin/env python3
"""
Regression test for decode-time grouped-query SDPA on Vulkan.

This matches the warmed Qwen decode shape: q_len=1, grouped-query heads, and
causal masking without an explicit array mask.
"""

import importlib.util
import os
import sys

import numpy as np


def load_mx():
    core_so = os.environ.get("MLX_CORE_SO")
    if core_so:
        import mlx

        spec = importlib.util.spec_from_file_location("mlx.core", core_so)
        mx = importlib.util.module_from_spec(spec)
        sys.modules["mlx.core"] = mx
        spec.loader.exec_module(mx)
        mlx.core = mx
        return mx

    import mlx.core as mx

    return mx


def main():
    mx = load_mx()
    if not mx.is_available(mx.gpu):
        print("SKIP: mx.gpu not available")
        return 0

    prev_fail = os.environ.get("MLX_VULKAN_FAIL_ON_CPU_FALLBACK")
    os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = "1"

    try:
        rng = np.random.default_rng(7)
        q_np = (0.25 * rng.standard_normal((1, 8, 1, 256))).astype(np.float16)
        k_np = (0.25 * rng.standard_normal((1, 2, 6, 256))).astype(np.float16)
        v_np = (0.25 * rng.standard_normal((1, 2, 6, 256))).astype(np.float16)
        scale = float(1.0 / np.sqrt(256.0))

        mx.set_default_device(mx.cpu)
        cpu_out = mx.fast.scaled_dot_product_attention(
            mx.array(q_np),
            mx.array(k_np),
            mx.array(v_np),
            scale=scale,
            mask="causal",
        )
        mx.eval(cpu_out)
        expected = np.array(cpu_out.tolist(), dtype=np.float32)

        mx.set_default_device(mx.gpu)
        gpu_out = mx.fast.scaled_dot_product_attention(
            mx.array(q_np),
            mx.array(k_np),
            mx.array(v_np),
            scale=scale,
            mask="causal",
        )
        mx.eval(gpu_out)
        got = np.array(gpu_out.tolist(), dtype=np.float32)

        if not np.allclose(got, expected, atol=2e-2, rtol=2e-2):
            raise AssertionError(
                f"grouped decode SDPA mismatch\nexpected:\n{expected}\nactual:\n{got}"
            )
        print(f"  PASS grouped_decode_sdpa: shape={got.shape}")
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan SDPA grouped decode smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
