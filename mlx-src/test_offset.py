import mlx.core as mx
x = mx.array([0.0, 1.0, float("inf"), float("-inf"), float("nan")]).astype(mx.float16)

try:
    print("x offset:", x.offset)
except Exception as e:
    print("Error getting offset:", e)

try:
    print("x itemsize:", x.itemsize)
except Exception as e:
    print("Error itemsize:", e)
