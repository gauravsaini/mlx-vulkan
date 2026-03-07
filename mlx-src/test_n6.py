import mlx.core as mx
mx.set_default_device(mx.gpu)
a = mx.array([-1.0, 1.0, 0.0, 1.0, -2.0, 3.0])
print("GPU RESULT:", mx.logical_not(a))
mx.set_default_device(mx.cpu)
print("CPU RESULT:", mx.logical_not(a))
