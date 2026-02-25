"""
Stage 11: Matmul on GPU
Run: python tests/vulkan/test_stage11_matmul.py
Pass: Matmul matches CPU reference within atol=1e-3
"""
import sys
import numpy as np

ATOL = 1e-3

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 11: Matmul GPU Op")
    print("═══════════════════════════════════════")

    try:
        import mlx.core as mx
    except ImportError as e:
        print(f"❌ {e}"); return False

    gpu = mx.gpu
    errors = []
    rng = np.random.RandomState(42)

    # Exact test: ones @ ones = N * ones
    for N in [4, 16, 32]:
        try:
            a = mx.ones((N, N))
            b = mx.ones((N, N))
            c = mx.matmul(a, b, stream=gpu)
            mx.eval(c)
            expected = np.full((N, N), float(N), dtype=np.float32)
            got = np.array(c.tolist(), dtype=np.float32)
            if not np.allclose(got, expected, atol=ATOL):
                errors.append(f"ones {N}x{N}: max_err={np.max(np.abs(got-expected)):.4f}")
                print(f"  ❌ ones {N}x{N} matmul")
            else:
                print(f"  ✅ ones {N}x{N} @ {N}x{N}")
        except Exception as e:
            errors.append(f"ones {N}x{N}: {e}")
            print(f"  ❌ ones {N}x{N}: {e}")

    # Numerical equivalence vs CPU
    sizes = [(32, 64, 48), (64, 128, 64), (128, 256, 128)]
    for (M, K, N) in sizes:
        try:
            A_np = rng.randn(M, K).astype(np.float32)
            B_np = rng.randn(K, N).astype(np.float32)
            A = mx.array(A_np)
            B = mx.array(B_np)

            cpu_res = mx.matmul(A, B, stream=mx.cpu)
            gpu_res = mx.matmul(A, B, stream=gpu)
            mx.eval(cpu_res, gpu_res)

            cpu_np = np.array(cpu_res.tolist(), dtype=np.float32)
            gpu_np = np.array(gpu_res.tolist(), dtype=np.float32)

            if not np.allclose(cpu_np, gpu_np, atol=ATOL, rtol=1e-3):
                max_err = np.max(np.abs(cpu_np - gpu_np))
                errors.append(f"({M},{K})@({K},{N}): max_err={max_err:.4f}")
                print(f"  ❌ ({M},{K})@({K},{N}) - max_err={max_err:.4f}")
            else:
                print(f"  ✅ ({M},{K})@({K},{N}) - CPU vs GPU match")
        except Exception as e:
            errors.append(f"({M},{K},{N}): {e}")
            print(f"  ❌ ({M},{K},{N}): {e}")

    # Batch matmul
    try:
        A = mx.array(rng.randn(4, 16, 32).astype(np.float32))
        B = mx.array(rng.randn(4, 32, 16).astype(np.float32))
        C_gpu = mx.matmul(A, B, stream=gpu)
        C_cpu = mx.matmul(A, B, stream=mx.cpu)
        mx.eval(C_gpu, C_cpu)
        gpu_np = np.array(C_gpu.tolist(), dtype=np.float32)
        cpu_np = np.array(C_cpu.tolist(), dtype=np.float32)
        if not np.allclose(gpu_np, cpu_np, atol=ATOL, rtol=1e-3):
            errors.append(f"batch matmul: max_err={np.max(np.abs(gpu_np-cpu_np)):.4f}")
            print(f"  ❌ batch matmul (4,16,32)@(4,32,16)")
        else:
            print(f"  ✅ batch matmul (4,16,32)@(4,32,16)")
    except Exception as e:
        errors.append(f"batch: {e}")
        print(f"  ❌ batch matmul: {e}")

    print()
    if errors:
        print(f"❌ STAGE 11 FAIL: {len(errors)} errors")
        return False

    print("✅ STAGE 11 PASS: Matmul correct")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
