"""
Stage 7: First GPU operation - array copy to GPU stream
Run: python tests/vulkan/test_stage7_gpu_copy.py
Pass: Arrays copied to GPU without crash, values correct
Requires: MLX built with MLX_BUILD_VULKAN=ON
"""
import sys
import numpy as np

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 7: GPU Stream Copy")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ FAIL: {e}")
        return False

    gpu = mx.gpu
    errors = []

    # Test 1: Simple scalar copy
    try:
        a = mx.array(42.0)
        b = mx.array(a, stream=gpu)
        mx.eval(b)
        assert abs(b.item() - 42.0) < 1e-6, f"Got {b.item()}"
        print("  ✅ scalar copy to GPU")
    except Exception as e:
        errors.append(f"scalar copy: {e}")

    # Test 2: 1D array copy
    try:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = mx.array(data)
        b = mx.array(a, stream=gpu)
        mx.eval(b)
        assert b.tolist() == data, f"Got {b.tolist()}"
        print("  ✅ 1D array copy")
    except Exception as e:
        errors.append(f"1D copy: {e}")

    # Test 3: 2D array copy
    try:
        data = [[1.0, 2.0], [3.0, 4.0]]
        a = mx.array(data)
        b = mx.array(a, stream=gpu)
        mx.eval(b)
        assert b.shape == (2, 2)
        flat = [x for row in b.tolist() for x in row]
        assert flat == [1.0, 2.0, 3.0, 4.0], f"Got {flat}"
        print("  ✅ 2D array copy")
    except Exception as e:
        errors.append(f"2D copy: {e}")

    # Test 4: Float16 copy
    try:
        a = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
        b = mx.array(a, stream=gpu)
        mx.eval(b)
        assert b.dtype == mx.float16
        print("  ✅ float16 copy")
    except Exception as e:
        errors.append(f"float16 copy: {e}")

    # Test 5: Zeros/ones
    try:
        z = mx.zeros((10,), stream=gpu)
        mx.eval(z)
        assert z.tolist() == [0.0] * 10
        print("  ✅ zeros on GPU")
    except Exception as e:
        errors.append(f"zeros: {e}")

    print()
    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        print(f"\n❌ STAGE 7 FAIL: {len(errors)} errors")
        return False

    print("✅ STAGE 7 PASS: GPU copy working")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
