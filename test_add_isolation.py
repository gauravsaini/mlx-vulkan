import mlx.core as mx
mx.set_default_device(mx.gpu)

# vector + vector
x1 = mx.array([1, 2], mx.int64)
y1 = mx.array([3.0, 4.0], mx.float32)
z1 = x1 + y1
mx.eval(z1)
print("vector + vector:", z1.tolist())

# vector + scalar
x2 = mx.array([1, 2], mx.int64)
z2 = x2 + 3.0
mx.eval(z2)
print("vector + scalar:", z2.tolist())

# scalar + scalar
x3 = mx.array(1, mx.int64)
z3 = x3 + 3.0
mx.eval(z3)
print("scalar + scalar:", z3.tolist())

# float vector + int vector
x4 = mx.array([1.0, 2.0], mx.float32)
y4 = mx.array([3, 4], mx.int64)
z4 = x4 + y4
mx.eval(z4)
print("float vector + int vector:", z4.tolist())

