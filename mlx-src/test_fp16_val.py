import mlx.core as mx

x = mx.array([0.0, 1.0, float("inf"), float("-inf"), float("nan")]).astype(mx.float16)

# This will run float16 -> float16 through unary.comp
y = mx.abs(x)

print(y.tolist())
