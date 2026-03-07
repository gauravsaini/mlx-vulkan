import mlx.core as mx
import numpy as np

x = mx.array([0.0, float("inf"), float("-inf"), float("nan")]).astype(mx.float16)

# Print raw bytes
print("CPU representation of float16 values:")
x_np = np.array(x).view(np.uint16)
for i, v in enumerate([0.0, float("inf"), float("-inf"), float("nan")]):
    print(f"{v}: 0x{x_np[i]:04x}")
