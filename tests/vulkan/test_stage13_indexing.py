#!/usr/bin/env python3
"""
Stage 13: Indexing (Gather/Scatter) Tests

Tests mx.take, mx.put, and array indexing with integer arrays.
These verify GatherAxis and ScatterAxis GPU dispatch.
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

def test_take_1d():
    import mlx.core as mx
    a = mx.array([10, 20, 30, 40, 50], dtype=mx.float32)
    idx = mx.array([0, 2, 4], dtype=mx.int32)
    out = mx.take(a, idx)
    mx.eval(out)
    check("take 1D", np.allclose(np.array(out), [10, 20+10, 30+20]))  # mx.take semantics

    # Actually test basic indexing
    result = np.array(out)
    expected = np.array([10, 30, 50], dtype=np.float32)
    check("take 1D values", np.allclose(result, expected))

def test_take_axis():
    import mlx.core as mx
    a = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)
    idx = mx.array([2, 0], dtype=mx.int32)
    out = mx.take(a, idx, axis=1)
    mx.eval(out)
    result = np.array(out)
    expected = np.array([[3, 1], [6, 4]], dtype=np.float32)
    check("take axis=1 shape", result.shape == (2, 2))
    check("take axis=1 values", np.allclose(result, expected))

def test_negative_indices():
    import mlx.core as mx
    a = mx.array([10, 20, 30, 40, 50], dtype=mx.float32)
    idx = mx.array([-1, -2], dtype=mx.int32)
    out = mx.take(a, idx)
    mx.eval(out)
    result = np.array(out)
    expected = np.array([50, 40], dtype=np.float32)
    check("negative indices", np.allclose(result, expected))

def test_scatter_basic():
    import mlx.core as mx
    # Test basic scatter: put values at indices
    src = mx.zeros(5, dtype=mx.float32)
    idx = mx.array([1, 3], dtype=mx.int32)
    updates = mx.array([100, 200], dtype=mx.float32)
    # Use scatter_along_axis for ScatterAxis dispatch
    try:
        out = mx.zeros(5, dtype=mx.float32)
        out = mx.put_along_axis(out, idx, updates, axis=0)
        mx.eval(out)
        check("scatter basic", True)
    except Exception as e:
        check("scatter basic", False, str(e))

def test_large_gather():
    import mlx.core as mx
    n = 10000
    a = mx.arange(n, dtype=mx.float32)
    idx = mx.array(np.random.randint(0, n, size=1000).astype(np.int32))
    out = mx.take(a, idx)
    mx.eval(out)
    result = np.array(out)
    expected = np.arange(n, dtype=np.float32)[np.array(idx)]
    check("large gather (10K source, 1K indices)", np.allclose(result, expected))

if __name__ == "__main__":
    print("━" * 50)
    print("  Stage 13: Indexing (Gather/Scatter) Tests")
    print("━" * 50)

    try:
        import mlx.core as mx
    except ImportError:
        print("❌ Cannot import mlx.core")
        sys.exit(1)

    try:
        test_take_1d()
        test_take_axis()
        test_negative_indices()
        test_scatter_basic()
        test_large_gather()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        FAIL += 1

    print()
    print("━" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("━" * 50)
    sys.exit(1 if FAIL > 0 else 0)
