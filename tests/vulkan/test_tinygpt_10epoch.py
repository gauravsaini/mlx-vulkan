#!/usr/bin/env python3
"""
TinyGPT 10-epoch training smoke for the Vulkan backend.

This extends the shorter tiny-transformer smoke into a longer convergence gate
using a tiny causal language model trained on a small repeated token corpus.
The path intentionally stays on eager, Vulkan-safe core ops and forbids CPU
fallback during training.
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
    print("  TinyGPT 10-Epoch Vulkan Smoke")
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
            h = self.ff2(mx.maximum(self.ff1(h), 0))
            return x + h

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size=24, dims=64, heads=4, layers=2, seq_len=16):
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
            return self.out_proj(self.ln(x))

    def build_dataset(vocab_size, seq_len, batch_size):
        base_pattern = np.array(
            [0, 5, 9, 13, 17, 21, 4, 8, 12, 16, 20, 3, 7, 11, 15, 19, 23],
            dtype=np.int32,
        )
        rows = []
        for i in range(batch_size):
            offset = (3 * i) % vocab_size
            rows.append((base_pattern[: seq_len + 1] + offset) % vocab_size)
        tokens = np.stack(rows, axis=0)
        return (
            mx.array(tokens[:, :-1], dtype=mx.int32),
            mx.array(tokens[:, 1:], dtype=mx.int32),
        )

    def check_training():
        mx.random.seed(17)
        vocab_size = 24
        seq_len = 16
        batch_size = 12
        epochs = 10

        x, y = build_dataset(vocab_size, seq_len, batch_size)
        model = TinyGPT(vocab_size=vocab_size, seq_len=seq_len)
        optimizer = optim.Adam(learning_rate=3e-3)
        optimizer.init(model.parameters())

        def loss_fn(params):
            model.update(params)
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = mx.value_and_grad(loss_fn)
        params = model.parameters()
        epoch_losses = []

        for _ in range(epochs):
            loss, grads = loss_and_grad(params)
            params = optimizer.apply_gradients(grads, params)
            mx.eval(loss)
            epoch_losses.append(float(loss.tolist()))

        if not np.isfinite(epoch_losses).all():
            raise RuntimeError(f"non-finite losses: {epoch_losses}")
        if not epoch_losses[-1] < epoch_losses[0]:
            raise RuntimeError(
                f"loss did not decrease over {epochs} epochs: "
                f"{epoch_losses[0]} -> {epoch_losses[-1]}"
            )
        if min(epoch_losses[-3:]) >= epoch_losses[0]:
            raise RuntimeError(f"late training did not improve enough: {epoch_losses}")
        print(
            "  Epoch losses:",
            ", ".join(f"{loss:.4f}" for loss in epoch_losses),
        )

    try:
        check("gpu detect", check_gpu, errors)
        check("tinygpt 10-epoch training", check_training, errors)
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

    print("PASS: TinyGPT 10-epoch Vulkan smoke succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
