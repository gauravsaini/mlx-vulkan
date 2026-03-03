import mlx.core as mx

mx.set_default_device(mx.gpu)
a = mx.ones((2, 2), dtype=mx.bool_)
print("a:", a)
b = mx.all(a)
print("all(a):", b)
