#!/usr/bin/env python3
"""
Tiny transformer training smoke for the Vulkan backend.

This gate intentionally uses manual causal attention instead of
mx.fast.scaled_dot_product_attention so the training path stays on the core
ops we are trying to validate for Vulkan compatibility.

It trains on real token embeddings with a next-token cross-entropy objective.
"""

import importlib.util
import os
import platform
import sys

import numpy as np


def info_to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def vendor_matches(info, required_vendor):
    text = " ".join(info_to_str(v).lower() for v in info.values())
    if required_vendor == "amd":
        return "amd" in text or "radeon" in text
    if required_vendor == "nvidia":
        return "nvidia" in text or "tesla" in text or "geforce" in text
    if required_vendor == "intel":
        return "intel" in text or "arc" in text
    raise RuntimeError(f"Unsupported MLX_VULKAN_REQUIRE_VENDOR value: {required_vendor}")


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
    print("  Tiny Transformer Vulkan Smoke")
    print("========================================")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")

    errors = []
    required_vendor = os.environ.get("MLX_VULKAN_REQUIRE_VENDOR", "").strip().lower()
    prev_fail_on_cpu_fallback = os.environ.get("MLX_VULKAN_FAIL_ON_CPU_FALLBACK")
    os.environ["MLX_VULKAN_FAIL_ON_CPU_FALLBACK"] = "1"

    def check_gpu():
        if not mx.is_available(mx.gpu):
            raise RuntimeError("mx.is_available(mx.gpu) returned false")
        info = mx.device_info(mx.gpu)
        if isinstance(info, list):
            info = info[0]
        print(f"  GPU info: {info}")
        if required_vendor and not vendor_matches(info, required_vendor):
            raise RuntimeError(f"{required_vendor} GPU required, got: {info}")

    class TinyBlock(nn.Module):
        def __init__(self, dims, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dims // heads
            self.ln1 = nn.LayerNorm(dims)
            self.ln2 = nn.LayerNorm(dims)
            self.q_proj = nn.Linear(dims, dims, bias=False)
            self.k_proj = nn.Linear(dims, dims, bias=False)
            self.v_proj = nn.Linear(dims, dims, bias=False)
            self.o_proj = nn.Linear(dims, dims, bias=False)
            self.ff1 = nn.Linear(dims, dims * 4, bias=False)
            self.ff2 = nn.Linear(dims * 4, dims, bias=False)

        def __call__(self, x, mask):
            h = self.ln1(x)
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)

            bsz, seq, dims = q.shape
            q = q.reshape(bsz, seq, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
            scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            attn_out = attn @ v
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq, dims)
            x = x + self.o_proj(attn_out)

            h = self.ln2(x)
            # Avoid nn.relu here: it is an mx.compile wrapper, and Vulkan
            # compiled graphs still route through CPU fallback.
            h = self.ff2(mx.maximum(self.ff1(h), 0))
            return x + h

    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size=16, dims=32, heads=4, layers=1, seq_len=8):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, dims)
            self.pos_emb = nn.Embedding(seq_len, dims)
            self.blocks = [TinyBlock(dims, heads) for _ in range(layers)]
            self.ln = nn.LayerNorm(dims)
            self.out_proj = nn.Linear(dims, vocab_size, bias=False)

        def __call__(self, tokens):
            seq = tokens.shape[1]
            positions = mx.arange(seq, dtype=mx.int32)[None, :]
            x = self.token_emb(tokens) + self.pos_emb(positions)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq, mx.float32)
            mask = mask[None, None, :, :]
            for block in self.blocks:
                x = block(x, mask)
            x = self.ln(x)
            return self.out_proj(x)

    def check_training():
        mx.random.seed(11)
        vocab_size = 16
        seq_len = 8
        batch = 8

        base = np.arange(seq_len + 1, dtype=np.int32)
        tokens_np = np.stack(
            [(base + i) % vocab_size for i in range(batch)],
            axis=0,
        )
        x = mx.array(tokens_np[:, :-1], dtype=mx.int32)
        y = mx.array(tokens_np[:, 1:], dtype=mx.int32)

        model = TinyTransformer(vocab_size=vocab_size, seq_len=seq_len)
        optimizer = optim.SGD(learning_rate=1e-2)
        optimizer.init(model.parameters())

        def loss_fn(params):
            model.update(params)
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = mx.value_and_grad(loss_fn)
        params = model.parameters()
        losses = []
        for _ in range(30):
            loss, grads = loss_and_grad(params)
            params = optimizer.apply_gradients(grads, params)
            mx.eval(loss)
            losses.append(float(loss.tolist()))

        if not np.isfinite(losses).all():
            raise RuntimeError(f"non-finite losses: {losses}")
        if not losses[-1] < losses[0]:
            raise RuntimeError(f"loss did not decrease: {losses[0]} -> {losses[-1]}")
        if min(losses[-5:]) > losses[0]:
            raise RuntimeError(f"training did not converge enough: {losses}")
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    try:
        check("gpu detect", check_gpu, errors)
        check("tiny transformer training", check_training, errors)
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

    print("PASS: tiny transformer Vulkan smoke succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
