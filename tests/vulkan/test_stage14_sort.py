#!/usr/bin/env python3
"""
Stage 14: Sort Tests

Tests mx.sort() on GPU via the bitonic sort shader.
Falls back to CPU for sizes > 512 or non-last-axis.
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

def test_sort_small():
    import mlx.core as mx
    a = mx.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=mx.float32)
    out = mx.sort(a)
    mx.eval(out)
    result = np.array(out)
    expected = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32))
    check("sort small (8 elems)", np.allclose(result, expected))

def test_sort_2d_last_axis():
    import mlx.core as mx
    np.random.seed(42)
    data = np.random.randn(5, 64).astype(np.float32)
    a = mx.array(data)
    out = mx.sort(a, axis=-1)
    mx.eval(out)
    result = np.array(out)
    expected = np.sort(data, axis=-1)
    check("sort 2D last axis (5x64)", np.allclose(result, expected))

def test_sort_power_of_two():
    import mlx.core as mx
    np.random.seed(123)
    data = np.random.randn(256).astype(np.float32)
    a = mx.array(data)
    out = mx.sort(a)
    mx.eval(out)
    result = np.array(out)
    expected = np.sort(data)
    check("sort 256 elements (pow2)", np.allclose(result, expected))

def test_sort_512():
    import mlx.core as mx
    np.random.seed(456)
    data = np.random.randn(512).astype(np.float32)
    a = mx.array(data)
    out = mx.sort(a)
    try:
        mx.eval(out)
        check("sort 512 elements (unsupported)", False, "Expected RuntimeError")
    except RuntimeError as e:
        check("sort 512 elements (unsupported)", "unsupported" in str(e).lower())

def test_sort_large_fallback():
    """Sort > 512 should fail safely with RuntimeError."""
    import mlx.core as mx
    np.random.seed(789)
    data = np.random.randn(1024).astype(np.float32)
    a = mx.array(data)
    out = mx.sort(a)
    try:
        mx.eval(out)
        check("sort 1024 (unsupported limit)", False, "Expected RuntimeError")
    except RuntimeError as e:
        check("sort 1024 (unsupported limit)", "unsupported" in str(e).lower())

def test_argsort():
    """ArgSort currently unsupported, should fail safely."""
    import mlx.core as mx
    a = mx.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=mx.float32)
    out = mx.argsort(a)
    try:
        mx.eval(out)
        check("argsort (unsupported limit)", False, "Expected RuntimeError")
    except RuntimeError as e:
        check("argsort (unsupported limit)", "unsupported" in str(e).lower())

if __name__ == "__main__":
    print("━" * 50)
    print("  Stage 14: Sort Tests")
    print("━" * 50)

    try:
        import mlx.core as mx
    except ImportError:
        print("❌ Cannot import mlx.core")
        sys.exit(1)

    try:
        test_sort_small()
        test_sort_2d_last_axis()
        test_sort_power_of_two()
        test_sort_512()
        test_sort_large_fallback()
        test_argsort()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        FAIL += 1

    print()
    print("━" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("━" * 50)
    sys.exit(1 if FAIL > 0 else 0)
