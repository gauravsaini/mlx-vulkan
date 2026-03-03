import mlx.core as mx

mx.set_default_device(mx.gpu)
a = mx.zeros((10, 0), dtype=mx.int32)
b = mx.zeros((0, 10), dtype=mx.int32)
out = a @ b
z = mx.zeros((10, 10), dtype=mx.int32)
print("array_equal:", mx.array_equal(out, z))
