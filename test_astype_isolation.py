import mlx.core as mx
mx.set_default_device(mx.gpu)
x = mx.array([1, 2], mx.int64)
# Cast int64 to float32
y = x.astype(mx.float32)
mx.eval(y)
print("Cast output:", y.tolist())
