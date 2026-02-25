#!/usr/bin/env python3
"""
Stage 16: Neural Net Ops (LayerNorm, RMSNorm, RoPE) Tests

Tests GPU dispatch of normalization and rotary position embedding.
"""

import sys
import traceback
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {msg}")

def numpy_layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return normalized * weight + bias

def numpy_rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def test_layer_norm():
    import mlx.core as mx
    import mlx.nn as nn

    np.random.seed(42)
    data = np.random.randn(4, 64).astype(np.float32)
    x = mx.array(data)

    ln = nn.LayerNorm(64)
    out = ln(x)
    mx.eval(out)
    result = np.array(out)

    # Verify shape and approximate normalization
    check("layer_norm shape", result.shape == (4, 64))
    # Each row should have mean ≈ 0 (with default weight=1, bias=0)
    row_means = result.mean(axis=-1)
    check("layer_norm mean ≈ 0", np.allclose(row_means, 0.0, atol=0.01))
    row_stds = result.std(axis=-1)
    check("layer_norm std ≈ 1", np.allclose(row_stds, 1.0, atol=0.05))

def test_rms_norm():
    import mlx.core as mx
    import mlx.nn as nn

    np.random.seed(42)
    data = np.random.randn(4, 64).astype(np.float32)
    x = mx.array(data)

    rn = nn.RMSNorm(64)
    out = rn(x)
    mx.eval(out)
    result = np.array(out)

    check("rms_norm shape", result.shape == (4, 64))
    # RMS norm: sqrt(mean(x^2)) should be close to 1
    rms_values = np.sqrt(np.mean(result**2, axis=-1))
    check("rms_norm rms ≈ 1", np.allclose(rms_values, 1.0, atol=0.1))

def test_layer_norm_large():
    import mlx.core as mx
    import mlx.nn as nn

    np.random.seed(123)
    data = np.random.randn(32, 256).astype(np.float32)
    x = mx.array(data)

    ln = nn.LayerNorm(256)
    out = ln(x)
    mx.eval(out)
    result = np.array(out)

    check("layer_norm 32x256 shape", result.shape == (32, 256))
    row_means = result.mean(axis=-1)
    check("layer_norm 32x256 mean ≈ 0", np.allclose(row_means, 0.0, atol=0.02))

def test_rms_norm_equivalence():
    """Compare RMSNorm output against manual numpy implementation."""
    import mlx.core as mx
    import mlx.nn as nn

    np.random.seed(456)
    data = np.random.randn(2, 32).astype(np.float32)
    weight = np.ones(32, dtype=np.float32)
    x = mx.array(data)

    rn = nn.RMSNorm(32)
    out = rn(x)
    mx.eval(out)
    result = np.array(out)

    expected = numpy_rms_norm(data, weight, eps=1e-5)
    check("rms_norm np equiv", np.allclose(result, expected, atol=1e-4))

if __name__ == "__main__":
    print("━" * 50)
    print("  Stage 16: Neural Net Ops Tests")
    print("━" * 50)

    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as e:
        print(f"❌ Cannot import: {e}")
        sys.exit(1)

    try:
        test_layer_norm()
        test_rms_norm()
        test_layer_norm_large()
        test_rms_norm_equivalence()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        FAIL += 1

    print()
    print("━" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("━" * 50)
    sys.exit(1 if FAIL > 0 else 0)
