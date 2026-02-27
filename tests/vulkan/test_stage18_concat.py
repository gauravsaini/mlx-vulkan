#!/usr/bin/env python3
import sys
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

def test_concat_1d():
    import mlx.core as mx
    a = mx.array([1, 2], dtype=mx.float32)
    b = mx.array([3, 4], dtype=mx.float32)
    out = mx.concatenate([a, b], axis=0)
    mx.eval(out)
    check("concat 1d", np.allclose(np.array(out), [1, 2, 3, 4]))

def test_concat_2d():
    import mlx.core as mx
    a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
    
    out_ax0 = mx.concatenate([a, b], axis=0)
    mx.eval(out_ax0)
    expected_0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    check("concat 2d axis 0", np.allclose(np.array(out_ax0), expected_0))
    
    out_ax1 = mx.concatenate([a, b], axis=1)
    mx.eval(out_ax1)
    expected_1 = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    check("concat 2d axis 1", np.allclose(np.array(out_ax1), expected_1))

if __name__ == "__main__":
    print("-" * 50)
    print("  Stage 18: Array Concatenation Tests")
    print("-" * 50)
    try:
        test_concat_1d()
        test_concat_2d()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("-" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("-" * 50)
    if FAIL > 0:
        sys.exit(1)
