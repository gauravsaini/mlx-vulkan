import mlx.core as mx
import numpy as np

# Let's use UNARY_ISINF (34) but inject controlled values
# float32 values
# 0.0, 1.0, inf, -inf
print("Float32")
x_f32 = mx.array([0.0, 1.0, float("inf"), float("-inf")]).astype(mx.float32)
print("isinf:", mx.isinf(x_f32).tolist())

print("Float16")
x_f16 = mx.array([0.0, 1.0, float("inf"), float("-inf")]).astype(mx.float16)
print("isinf:", mx.isinf(x_f16).tolist())
