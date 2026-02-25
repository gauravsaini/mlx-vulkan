"""
Stage 9: Elementwise binary ops on GPU
Run: python tests/vulkan/test_stage9_binary.py
Pass: All binary ops match numpy reference within atol=1e-4
"""
import sys
import numpy as np

ATOL = 1e-4

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 9: Binary GPU Ops")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ {e}"); return False

    gpu = mx.gpu
    errors = []
    rng = np.random.RandomState(42)

    a_data = rng.uniform(-3.0, 3.0, size=128).astype(np.float32)
    b_data = rng.uniform(0.5, 3.0, size=128).astype(np.float32)  # positive for div

    a_mx = mx.array(a_data)
    b_mx = mx.array(b_data)

    test_cases = [
        ("add",      mx.add,      lambda a,b: a + b),
        ("subtract", mx.subtract, lambda a,b: a - b),
        ("multiply", mx.multiply, lambda a,b: a * b),
        ("divide",   mx.divide,   lambda a,b: a / b),
        ("maximum",  mx.maximum,  np.maximum),
        ("minimum",  mx.minimum,  np.minimum),
        ("power",    mx.power,    lambda a,b: np.abs(a) ** b),  # use abs to avoid complex
        ("equal",    mx.equal,    lambda a,b: (a == b).astype(np.float32)),
        ("not_equal",mx.not_equal,lambda a,b: (a != b).astype(np.float32)),
        ("less",     mx.less,     lambda a,b: (a < b).astype(np.float32)),
        ("greater",  mx.greater,  lambda a,b: (a > b).astype(np.float32)),
    ]

    for name, mx_fn, np_fn in test_cases:
        try:
            inp_a = a_mx
            inp_b = b_mx

            # power: use abs of a to avoid complex numbers
            if name == "power":
                inp_a = mx.abs(a_mx)

            result = mx_fn(inp_a, inp_b, stream=gpu)
            mx.eval(result)
            got = np.array(result.tolist(), dtype=np.float32)

            if name == "power":
                expected = np_fn(np.abs(a_data), b_data)
            else:
                expected = np_fn(a_data, b_data).astype(np.float32)

            mask = np.isfinite(expected) & np.isfinite(got)
            if not np.allclose(got[mask], expected[mask], atol=ATOL):
                max_err = np.max(np.abs(got[mask] - expected[mask]))
                errors.append(f"{name}: max_err={max_err:.6f}")
                print(f"  ❌ {name} (max_err={max_err:.6f})")
            else:
                print(f"  ✅ {name}")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"  ❌ {name}: {e}")

    # Test scalar broadcast
    try:
        a = mx.array([1., 2., 3., 4.])
        scalar = mx.array(2.0)
        result = mx.multiply(a, scalar, stream=gpu)
        mx.eval(result)
        assert result.tolist() == [2., 4., 6., 8.], f"Got {result.tolist()}"
        print("  ✅ scalar broadcast (mul)")
    except Exception as e:
        errors.append(f"scalar broadcast: {e}")
        print(f"  ❌ scalar broadcast: {e}")

    print()
    if errors:
        print(f"❌ STAGE 9 FAIL: {len(errors)} failed")
        return False

    print(f"✅ STAGE 9 PASS: All binary ops correct")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
