import mlx.core as mx
import numpy as np

for n in [128, 2048, 4096, 8192]:
    print(f"\nTesting Hadamard (n={n})...")
    
    # Run on GPU
    a_gpu = mx.random.uniform(-1, 1, shape=(3, n), dtype=mx.float32, stream=mx.gpu)
    scale = 1.0 / np.sqrt(n)
    
    out_gpu = mx.hadamard_transform(a_gpu, scale=scale, stream=mx.gpu)
    mx.eval(out_gpu)
    actual = np.array(out_gpu)
    
    # Run on CPU for ground truth
    a_cpu = mx.array(np.array(a_gpu))
    expected = np.array(mx.hadamard_transform(a_cpu, scale=scale, stream=mx.cpu))
    
    if np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        print(f"PASSED for n={n}!")
    else:
        print(f"FAILED for n={n}!")
        print("Expected:", expected[0, :3])
        print("Actual:", actual[0, :3])
