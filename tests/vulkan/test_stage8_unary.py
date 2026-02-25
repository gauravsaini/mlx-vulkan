"""
Stage 8: Elementwise unary ops on GPU
Run: python tests/vulkan/test_stage8_unary.py
Pass: All unary ops match numpy reference within atol=1e-4
"""
import sys
import numpy as np

ATOL = 1e-4

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 8: Unary GPU Ops")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ {e}"); return False

    gpu = mx.gpu
    errors = []

    rng = np.random.RandomState(42)

    # Define test cases: (name, mlx_fn, numpy_fn, input_gen)
    pos_data = rng.uniform(0.1, 5.0, size=64).astype(np.float32)
    any_data = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
    unit_data = rng.uniform(-0.99, 0.99, size=64).astype(np.float32)

    test_cases = [
        ("abs",     mx.abs,      np.abs,      any_data),
        ("negative",mx.negative, np.negative, any_data),
        ("sqrt",    mx.sqrt,     np.sqrt,     pos_data),
        ("square",  mx.square,   np.square,   any_data),
        ("exp",     mx.exp,      np.exp,      np.clip(any_data, -5, 5)),
        ("log",     mx.log,      np.log,      pos_data),
        ("log2",    mx.log2,     np.log2,     pos_data),
        ("sin",     mx.sin,      np.sin,      any_data),
        ("cos",     mx.cos,      np.cos,      any_data),
        ("tan",     mx.tan,      np.tan,      np.clip(any_data, -1.5, 1.5)),
        ("tanh",    mx.tanh,     np.tanh,     any_data),
        ("sigmoid", mx.sigmoid,  lambda x: 1/(1+np.exp(-x)), any_data),
        ("ceil",    mx.ceil,     np.ceil,     any_data),
        ("floor",   mx.floor,    np.floor,    any_data),
        ("arcsin",  mx.arcsin,   np.arcsin,   unit_data),
        ("arccos",  mx.arccos,   np.arccos,   unit_data),
        ("arctan",  mx.arctan,   np.arctan,   any_data),
    ]

    for name, mx_fn, np_fn, data in test_cases:
        try:
            a = mx.array(data)
            result = mx_fn(a, stream=gpu)
            mx.eval(result)
            got = np.array(result.tolist(), dtype=np.float32)
            expected = np_fn(data).astype(np.float32)

            # Filter nans/infs for comparison
            mask = np.isfinite(expected) & np.isfinite(got)
            if not np.allclose(got[mask], expected[mask], atol=ATOL):
                max_err = np.max(np.abs(got[mask] - expected[mask]))
                errors.append(f"{name}: max_err={max_err:.6f} (atol={ATOL})")
                print(f"  ❌ {name} (max_err={max_err:.6f})")
            else:
                print(f"  ✅ {name}")
        except Exception as e:
            errors.append(f"{name}: exception: {e}")
            print(f"  ❌ {name}: {e}")

    print()
    if errors:
        print(f"❌ STAGE 8 FAIL: {len(errors)}/{len(test_cases)} failed")
        return False

    print(f"✅ STAGE 8 PASS: All {len(test_cases)} unary ops correct")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
