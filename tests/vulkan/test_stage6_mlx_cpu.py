"""
Stage 6: MLX loads with Vulkan backend, CPU stream still works
Run: python tests/vulkan/test_stage6_mlx_cpu.py
Pass: All assertions pass, no crash
"""
import sys
import traceback

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 6: MLX CPU Stream Sanity")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
        print(f"✅ mlx imported: {mx.__version__}")
    except ImportError as e:
        print(f"❌ FAIL: Cannot import mlx: {e}")
        print("   Build with: pip install -e mlx-src/")
        return False

    errors = []

    # Basic array creation
    try:
        a = mx.array([1.0, 2.0, 3.0])
        mx.eval(a)
        assert a.tolist() == [1.0, 2.0, 3.0]
        print("  ✅ array creation + eval")
    except Exception as e:
        errors.append(f"array creation: {e}")

    # Add
    try:
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = mx.add(a, b)
        mx.eval(c)
        assert c.tolist() == [5.0, 7.0, 9.0], f"Got {c.tolist()}"
        print("  ✅ add")
    except Exception as e:
        errors.append(f"add: {e}")

    # Multiply
    try:
        a = mx.array([2.0, 3.0, 4.0])
        b = mx.array([2.0, 2.0, 2.0])
        c = mx.multiply(a, b)
        mx.eval(c)
        assert c.tolist() == [4.0, 6.0, 8.0], f"Got {c.tolist()}"
        print("  ✅ multiply")
    except Exception as e:
        errors.append(f"multiply: {e}")

    # Reshape
    try:
        a = mx.ones((4, 4))
        b = mx.reshape(a, (2, 8))
        mx.eval(b)
        assert b.shape == (2, 8)
        print("  ✅ reshape")
    except Exception as e:
        errors.append(f"reshape: {e}")

    # Sum reduction
    try:
        a = mx.array([1.0, 2.0, 3.0, 4.0])
        s = mx.sum(a)
        mx.eval(s)
        assert abs(s.item() - 10.0) < 1e-5, f"Got {s.item()}"
        print("  ✅ sum")
    except Exception as e:
        errors.append(f"sum: {e}")

    print()
    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        print(f"\n❌ STAGE 6 FAIL: {len(errors)} errors")
        return False
    else:
        print("✅ STAGE 6 PASS: MLX CPU stream working")
        return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
