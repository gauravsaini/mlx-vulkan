import mlx.core as mx
mx.set_default_device(mx.gpu)
x32 = mx.array([0.0, float('-inf')], dtype=mx.float32)
y32 = mx.isinf(x32)
mx.eval(y32)
print("float32:", y32)

x16 = mx.array([0.0, float('-inf'), 0.0, float('inf'), float('nan'), 0.0], dtype=mx.float16)
y16 = mx.isinf(x16)
mx.eval(y16)
print("float16:", y16)
