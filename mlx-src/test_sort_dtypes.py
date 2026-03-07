import mlx.core as mx
import numpy as np

# ensure we dispatch arrays >= 128 to hit the radix branch
size = (4, 300)

for dtype in [mx.float16, mx.bfloat16, mx.int8, mx.uint16, mx.int16, mx.float32]:
    print(f"Testing {dtype}...")
    if dtype in [mx.float16, mx.bfloat16, mx.float32]:
        a = mx.random.uniform(-100, 100, size, dtype=dtype)
    else:
        a = mx.random.randint(-100, 100, size, dtype=dtype)
        
    out = mx.sort(a, axis=-1)
    
    # Check correctness against numpy
    a_f32 = a.astype(mx.float32)
    a_f32_np = np.array(a_f32)
    expected = np.sort(a_f32_np, axis=-1)
    
    # evaluate output
    actual = np.array(out.astype(mx.float32))
    
    match = np.allclose(actual, expected, equal_nan=True) if dtype in [mx.float16, mx.bfloat16, mx.float32] else np.array_equal(actual, expected)
    if not match:
        print(f"FAILED for {dtype}!")
        print("Expected:", expected[0, :10])
        print("Actual:", actual[0, :10])
    else:
        print(f"PASSED for {dtype}!")
