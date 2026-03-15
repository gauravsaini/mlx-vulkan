import mlx.core as mx

@mx.compile
def f(x, y):
    return mx.sin(x) * mx.exp(y) + mx.maximum(x, y) - mx.abs(x)

a = mx.array([-1.0, 0.0, 1.0], dtype=mx.float32)
b = mx.array([2.0, -2.0, 0.5], dtype=mx.float32)

print("Inputs:")
print("a =", a)
print("b =", b)

out = f(a, b)
print("Output:", out)
