import mlx.core as mx
mx.set_default_device(mx.gpu)
a = mx.ones((2, 2))
b = mx.ones((2, 2))
c = mx.matmul(a, b)
mx.eval(c)
print(c)
