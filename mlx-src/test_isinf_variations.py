import mlx.core as mx

x = mx.array([float("inf"), 1.0, 0.0, float("-inf"), float("nan")]).astype(mx.float16)

print("x:", x)
print("isinf expected:   ", [True if float(v) in [float("inf"), float("-inf")] else False for v in x])
print("isinf mlx:       ", mx.isinf(x).tolist())
print("isneginf:        ", mx.isneginf(x).tolist())

x2 = mx.array([0.0, 0.0, 0.0, 0.0, 0.0]).astype(mx.float16)
print("\nx2:", x2)
print("isinf x2:        ", mx.isinf(x2).tolist())
print("isneginf x2:     ", mx.isneginf(x2).tolist())

x3 = mx.array([float("-inf")] * 5).astype(mx.float16)
print("\nx3:", x3)
print("isinf x3:        ", mx.isinf(x3).tolist())
print("isneginf x3:     ", mx.isneginf(x3).tolist())

