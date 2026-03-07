import mlx.core as mx

x16 = mx.array([0.0, float("-inf"), 0.0, float("inf"), float("nan"), 0.0], dtype=mx.float16)
# Just a copy to see float16 unpacking
copy16 = x16.astype(mx.float32)
print("copy16:", copy16)

