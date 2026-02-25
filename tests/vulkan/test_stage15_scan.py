#!/usr/bin/env python3
"""
Stage 15: Scan (Prefix Sum/Product) Tests

Tests mx.cumsum() and mx.cumprod() via scan.comp shader.
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

def test_cumsum_1d():
    import mlx.core as mx
    a = mx.array([1, 2, 3, 4], dtype=mx.float32)
    out = mx.cumsum(a)
    mx.eval(out)
    result = np.array(out)
    expected = np.cumsum([1, 2, 3, 4]).astype(np.float32)
    check("cumsum 1D", np.allclose(result, expected))

def test_cumsum_2d():
    import mlx.core as mx
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = mx.array(data)
    out = mx.cumsum(a, axis=-1)
    mx.eval(out)
    result = np.array(out)
    expected = np.cumsum(data, axis=-1)
    check("cumsum 2D axis=-1", np.allclose(result, expected))

def test_cumprod_1d():
    import mlx.core as mx
    a = mx.array([1, 2, 3, 4], dtype=mx.float32)
    out = mx.cumprod(a)
    mx.eval(out)
    result = np.array(out)
    expected = np.cumprod([1, 2, 3, 4]).astype(np.float32)
    check("cumprod 1D", np.allclose(result, expected))

def test_cumsum_larger():
    import mlx.core as mx
    np.random.seed(42)
    data = np.random.randn(10, 128).astype(np.float32)
    a = mx.array(data)
    out = mx.cumsum(a, axis=-1)
    mx.eval(out)
    result = np.array(out)
    expected = np.cumsum(data, axis=-1)
    check("cumsum 10x128 axis=-1", np.allclose(result, expected, atol=1e-4))

def test_cumsum_large_fallback():
    """Scan > 512 falls back to CPU."""
    import mlx.core as mx
    np.random.seed(99)
    data = np.random.randn(1024).astype(np.float32)
    a = mx.array(data)
    out = mx.cumsum(a)
    mx.eval(out)
    result = np.array(out)
    expected = np.cumsum(data)
    check("cumsum 1024 (CPU fallback)", np.allclose(result, expected, atol=1e-3))

if __name__ == "__main__":
    print("━" * 50)
    print("  Stage 15: Scan (Prefix Sum/Product) Tests")
    print("━" * 50)

    try:
        import mlx.core as mx
    except ImportError:
        print("❌ Cannot import mlx.core")
        sys.exit(1)

    try:
        test_cumsum_1d()
        test_cumsum_2d()
        test_cumprod_1d()
        test_cumsum_larger()
        test_cumsum_large_fallback()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        FAIL += 1

    print()
    print("━" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("━" * 50)
    sys.exit(1 if FAIL > 0 else 0)
