import mlx.core as mx
mx.set_default_device(mx.gpu)
a = mx.array([0.0, 1.0])
print("GPU RESULT:", mx.logical_not(a))
