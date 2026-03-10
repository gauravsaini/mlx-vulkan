#!/usr/bin/env python3
"""
Transformer-critical op audit for Vulkan.

This script is meant to stay narrower than the tiny-transformer training smoke:
it isolates the forward/backward building blocks so we can identify the first
missing training-path op on real Linux GPU hardware.
"""

import importlib.util
import os
import platform
import sys

import numpy as np


def check(name, fn, errors):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as exc:
        errors.append(f"{name}: {exc}")


def main():
    try:
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
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:
        print(f"FAIL: Cannot import MLX modules: {exc}")
        return 1

    print("========================================")
    print("  Transformer Ops Vulkan Audit")
    print("========================================")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")

    errors = []
    prev_fail_on_cpu_fallback = os.environ.get("MLX_VULKAN_FAIL_ON_CPU_FALLBACK")
    os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = "1"

    def check_gpu():
        if not mx.is_available(mx.gpu):
            raise RuntimeError("mx.is_available(mx.gpu) returned false")
        info = mx.device_info(mx.gpu)
        if isinstance(info, list):
            info = info[0]
        print(f"  GPU info: {info}")

    def check_embeddings_and_attention():
        emb = nn.Embedding(16, 32)
        x = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=mx.int32)
        h = emb(x)
        q = h.reshape(2, 4, 4, 8).transpose(0, 2, 1, 3)
        k = h.reshape(2, 4, 4, 8).transpose(0, 2, 1, 3)
        v = h.reshape(2, 4, 4, 8).transpose(0, 2, 1, 3)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(4, mx.float32)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (8 ** -0.5)
        scores = scores + mask[None, None, :, :]
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        mx.eval(out)
        out_np = np.asarray(out)
        if out_np.shape != (2, 4, 4, 8):
            raise RuntimeError(f"unexpected attention shape: {out_np.shape}")
        if not np.isfinite(out_np).all():
            raise RuntimeError("attention output contains non-finite values")

    def check_layer_norm_forward():
        layer = nn.LayerNorm(32)
        x = mx.random.normal(shape=(2, 4, 32))
        y = layer(x)
        mx.eval(y)
        y_np = np.asarray(y)
        if y_np.shape != (2, 4, 32):
            raise RuntimeError(f"unexpected layer norm shape: {y_np.shape}")
        if not np.isfinite(y_np).all():
            raise RuntimeError("layer norm forward contains non-finite values")

    def check_layer_norm_backward():
        layer = nn.LayerNorm(32)
        x = mx.random.normal(shape=(2, 4, 32))

        def loss_fn(inp):
            return mx.mean(layer(inp) * layer(inp))

        loss, grad = mx.value_and_grad(loss_fn)(x)
        mx.eval(loss, grad)
        grad_np = np.asarray(grad)
        if not np.isfinite(grad_np).all():
            raise RuntimeError("layer norm backward returned non-finite gradients")

    def check_optimizer_update():
        params = {"w": mx.ones((4, 4), dtype=mx.float32)}
        grads = {"w": mx.full((4, 4), 0.125, dtype=mx.float32)}
        optimizer = optim.Adam(learning_rate=1e-2)
        optimizer.init(params)
        updated = optimizer.apply_gradients(grads, params)
        delta = np.asarray(updated["w"]) - np.asarray(params["w"])
        if np.allclose(delta, 0.0, atol=1e-8):
            raise RuntimeError("optimizer update produced no change")

    try:
        check("gpu detect", check_gpu, errors)
        check("embeddings + causal attention forward", check_embeddings_and_attention, errors)
        check("layer norm forward", check_layer_norm_forward, errors)
        check("layer norm backward", check_layer_norm_backward, errors)
        check("optimizer update", check_optimizer_update, errors)
    finally:
        if prev_fail_on_cpu_fallback is None:
            os.environ.pop("MLX_VULKAN_FAIL_ON_CPU_FALLBACK", None)
        else:
            os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = prev_fail_on_cpu_fallback

    print()
    if errors:
        for err in errors:
            print(f"  FAIL {err}")
        print(f"\nFAIL: {len(errors)} checks failed")
        return 1

    print("PASS: transformer-critical ops audit succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
