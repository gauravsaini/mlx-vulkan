import mlx.core as mx

mx.set_default_device(mx.gpu)
a = mx.zeros((10, 0))
b = mx.zeros((0, 10))
out = a @ b
z = mx.zeros((10, 10))

print("out evaluated?", out)
print("array_equal:", mx.array_equal(out, z))
