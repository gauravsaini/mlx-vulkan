#!/usr/bin/env python3
"""
Regression test for Vulkan conv1d execution.

This exercises both dense and depthwise 1D convolutions with
MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1 so unsupported GPU paths fail loudly.
"""

import importlib.util
import os
import sys

import numpy as np


def manual_conv1d(x, w, stride=1, padding=0, groups=1):
    batch, width, channels = x.shape
    out_channels, kernel, channels_per_group = w.shape
    out_width = ((width + 2 * padding - kernel) // stride) + 1
    out = np.zeros((batch, out_width, out_channels), dtype=np.float32)
    out_channels_per_group = out_channels // groups

    for n in range(batch):
        for ow in range(out_width):
            for oc in range(out_channels):
                group = oc // out_channels_per_group
                ic_base = group * channels_per_group
                acc = 0.0
                for kw in range(kernel):
                    iw = ow * stride + kw - padding
                    if 0 <= iw < width:
                        for ic in range(channels_per_group):
                            acc += (
                                float(x[n, iw, ic_base + ic]) *
                                float(w[oc, kw, ic])
                            )
                out[n, ow, oc] = acc
    return out


def run_case(mx, name, x_np, w_np, groups, out_dtype, atol, rtol):
    x = mx.array(x_np)
    w = mx.array(w_np)
    y = mx.conv1d(x, w, stride=1, padding=0, dilation=1, groups=groups)
    y = y.astype(out_dtype)
    mx.eval(y)
    got = np.array(y.tolist(), dtype=np.float32)
    expected = manual_conv1d(
        np.array(x_np, dtype=np.float32),
        np.array(w_np, dtype=np.float32),
        stride=1,
        padding=0,
        groups=groups,
    )
    if not np.allclose(got, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch\nexpected:\n{expected}\nactual:\n{got}"
        )
    print(f"  PASS {name}: shape={got.shape}")


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

        dense_x = np.arange(1, 1 + 1 * 8 * 4, dtype=np.float32).reshape(1, 8, 4)
        dense_w = np.arange(1, 1 + 5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
        run_case(
            mx,
            "dense_conv1d_float32",
            dense_x,
            dense_w,
            groups=1,
            out_dtype=mx.float32,
            atol=1e-5,
            rtol=1e-5,
        )

        depthwise_x = (
            np.arange(1, 1 + 1 * 8 * 4, dtype=np.float32).reshape(1, 8, 4) / 8.0
        ).astype(np.float16)
        depthwise_w = (
            np.arange(1, 1 + 4 * 4 * 1, dtype=np.float32).reshape(4, 4, 1) / 16.0
        ).astype(np.float16)
        run_case(
            mx,
            "depthwise_conv1d_float16",
            depthwise_x,
            depthwise_w,
            groups=4,
            out_dtype=mx.float16,
            atol=2e-2,
            rtol=2e-2,
        )
    finally:
        if prev_fail is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail

    print("PASS: Vulkan conv1d smoke succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
