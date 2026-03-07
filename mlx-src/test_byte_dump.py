import mlx.core as mx
import numpy as np

# Let's inspect the raw bytes of the boolean array.
x_f16 = mx.array([0.0, 1.0, float("inf"), float("-inf"), float("nan")]).astype(mx.float16)
out_mx = mx.isinf(x_f16)

# Force evaluation and get raw bytes
out_np = np.array(out_mx)

print("out_np shape:", out_np.shape)
print("out_np dtype:", out_np.dtype)
print("Python list:", out_mx.tolist())
try:
    print("Raw bytes:", out_np.view(np.uint8).tolist())
except Exception as e:
    print("Could not view as uint8:", e)

# Test with 20 elements
x_f16_20 = mx.array([float("inf")] * 20).astype(mx.float16)
out_mx_20 = mx.isinf(x_f16_20)
print("\n20 elements:")
print("Python list:", out_mx_20.tolist())

# Test with 20 zeroes
z_f16_20 = mx.array([0.0] * 20).astype(mx.float16)
z_out = mx.isinf(z_f16_20)
print("\n20 zeroes:")
print("Python list:", z_out.tolist())
