"""
Stage 10: Reduction ops on GPU (sum, max, min, mean)
Run: python tests/vulkan/test_stage10_reduce.py
Pass: All reductions match numpy within atol=1e-3
"""
import sys
import numpy as np

ATOL = 1e-3

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 10: Reduction GPU Ops")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ {e}"); return False

    gpu = mx.gpu
    errors = []
    rng = np.random.RandomState(42)

    # 1D reductions
    data_1d = rng.uniform(-3, 3, size=256).astype(np.float32)
    a_1d = mx.array(data_1d)

    tests_1d = [
        ("sum (1D)",  mx.sum,  np.sum,  data_1d),
        ("max (1D)",  mx.max,  np.max,  data_1d),
        ("min (1D)",  mx.min,  np.min,  data_1d),
        ("mean (1D)", mx.mean, np.mean, data_1d),
    ]

    for name, mx_fn, np_fn, data in tests_1d:
        try:
            result = mx_fn(mx.array(data), stream=gpu)
            mx.eval(result)
            got = result.item()
            expected = float(np_fn(data))
            if abs(got - expected) > ATOL:
                errors.append(f"{name}: got={got:.4f} expected={expected:.4f}")
                print(f"  ❌ {name} (got={got:.4f}, expected={expected:.4f})")
            else:
                print(f"  ✅ {name}")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"  ❌ {name}: {e}")

    # 2D axis reductions
    data_2d = rng.uniform(-3, 3, size=(8, 32)).astype(np.float32)
    a_2d = mx.array(data_2d)

    for axis in [0, 1]:
        for name_short, mx_fn, np_fn in [
            ("sum",  mx.sum,  np.sum),
            ("max",  mx.max,  np.max),
            ("min",  mx.min,  np.min),
        ]:
            name = f"{name_short} axis={axis}"
            try:
                result = mx_fn(a_2d, axis=axis, stream=gpu)
                mx.eval(result)
                got = np.array(result.tolist(), dtype=np.float32)
                expected = np_fn(data_2d, axis=axis).astype(np.float32)

                if not np.allclose(got, expected, atol=ATOL):
                    max_err = np.max(np.abs(got - expected))
                    errors.append(f"{name}: max_err={max_err:.6f}")
                    print(f"  ❌ {name} (max_err={max_err:.6f})")
                else:
                    print(f"  ✅ {name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"  ❌ {name}: {e}")

    # Prod (small data to avoid overflow)
    try:
        small = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
        result = mx.prod(mx.array(small), stream=gpu)
        mx.eval(result)
        got = result.item()
        expected = float(np.prod(small))
        assert abs(got - expected) < ATOL, f"got={got} expected={expected}"
        print("  ✅ prod (1D)")
    except Exception as e:
        errors.append(f"prod: {e}")
        print(f"  ❌ prod: {e}")

    print()
    if errors:
        print(f"❌ STAGE 10 FAIL: {len(errors)} errors")
        return False

    print("✅ STAGE 10 PASS: All reductions correct")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
