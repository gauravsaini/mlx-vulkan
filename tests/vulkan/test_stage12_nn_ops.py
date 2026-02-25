"""
Stage 12: Neural net ops - softmax, layernorm, argmax
Run: python tests/vulkan/test_stage12_nn_ops.py
Pass: All ops match CPU reference within atol=1e-4
"""
import sys
import numpy as np

ATOL = 1e-4

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 12: Neural Net GPU Ops")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ {e}"); return False

    gpu = mx.gpu
    errors = []
    rng = np.random.RandomState(42)

    # Softmax
    try:
        data = rng.randn(32, 128).astype(np.float32)
        x = mx.array(data)
        result = mx.softmax(x, axis=-1, stream=gpu)
        mx.eval(result)
        got = np.array(result.tolist(), dtype=np.float32)

        # Rows must sum to 1
        row_sums = got.sum(axis=-1)
        max_sum_err = np.max(np.abs(row_sums - 1.0))
        if max_sum_err > 1e-5:
            errors.append(f"softmax row sum err: {max_sum_err:.6f}")
            print(f"  ❌ softmax (row sum err={max_sum_err:.6f})")
        else:
            # Compare vs numpy
            exp_x = np.exp(data - data.max(axis=-1, keepdims=True))
            expected = (exp_x / exp_x.sum(axis=-1, keepdims=True)).astype(np.float32)
            if np.allclose(got, expected, atol=ATOL):
                print(f"  ✅ softmax (32x128)")
            else:
                max_err = np.max(np.abs(got - expected))
                errors.append(f"softmax value: max_err={max_err:.6f}")
                print(f"  ❌ softmax value mismatch (max_err={max_err:.6f})")
    except Exception as e:
        errors.append(f"softmax: {e}")
        print(f"  ❌ softmax: {e}")

    # Argmax / Argmin
    for fn_name, mx_fn, np_fn in [
        ("argmax", mx.argmax, np.argmax),
        ("argmin", mx.argmin, np.argmin),
    ]:
        try:
            data = rng.randn(16, 64).astype(np.float32)
            x = mx.array(data)
            result = mx_fn(x, axis=-1, stream=gpu)
            mx.eval(result)
            got = np.array(result.tolist(), dtype=np.int32)
            expected = np_fn(data, axis=-1).astype(np.int32)
            if not np.all(got == expected):
                mismatch = np.sum(got != expected)
                errors.append(f"{fn_name}: {mismatch} mismatches")
                print(f"  ❌ {fn_name} ({mismatch} mismatches)")
            else:
                print(f"  ✅ {fn_name}")
        except Exception as e:
            errors.append(f"{fn_name}: {e}")
            print(f"  ❌ {fn_name}: {e}")

    # Sort
    try:
        data = rng.randn(8, 32).astype(np.float32)
        x = mx.array(data)
        result = mx.sort(x, axis=-1, stream=gpu)
        mx.eval(result)
        got = np.array(result.tolist(), dtype=np.float32)
        expected = np.sort(data, axis=-1)
        if not np.allclose(got, expected, atol=ATOL):
            errors.append(f"sort: mismatch")
            print(f"  ❌ sort")
        else:
            print(f"  ✅ sort")
    except Exception as e:
        errors.append(f"sort: {e}")
        print(f"  ❌ sort: {e}")

    # LogSumExp
    try:
        data = rng.randn(16, 64).astype(np.float32)
        x = mx.array(data)
        result = mx.logsumexp(x, axis=-1, stream=gpu)
        mx.eval(result)
        got = np.array(result.tolist(), dtype=np.float32)
        # numpy reference
        expected = np.log(np.sum(np.exp(data - data.max(axis=-1, keepdims=True)), axis=-1)) + data.max(axis=-1)
        if not np.allclose(got, expected.astype(np.float32), atol=ATOL):
            max_err = np.max(np.abs(got - expected))
            errors.append(f"logsumexp: max_err={max_err:.6f}")
            print(f"  ❌ logsumexp (max_err={max_err:.6f})")
        else:
            print(f"  ✅ logsumexp")
    except Exception as e:
        errors.append(f"logsumexp: {e}")
        print(f"  ❌ logsumexp: {e}")

    print()
    if errors:
        print(f"❌ STAGE 12 FAIL: {len(errors)} errors")
        return False

    print("✅ STAGE 12 PASS: Neural net ops correct")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
