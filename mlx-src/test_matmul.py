import mlx.core as mx

mx.set_default_device(mx.gpu)

a = mx.ones((16, 16))
b = mx.ones((16, 16))
c = mx.matmul(a, b)
mx.eval(c)
print(c)

