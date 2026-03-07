import mlx.core as mx

x = mx.array([0.0, 1.0, float("inf"), float("-inf"), float("nan")]).astype(mx.float16)

# Test boolean array generation for float16
print("x:", x)
print("isinf expected:   ", [True if float(v) in [float("inf"), float("-inf")] else False for v in x])
print("isinf mlx:       ", mx.isinf(x).tolist())
print("isnan expected:   ", [True if str(v) == "nan" else False for v in x])
print("isnan mlx:       ", mx.isnan(x).tolist())
print("isneginf expected:", [True if float(v) == float("-inf") else False for v in x])
print("isneginf mlx:    ", mx.isneginf(x).tolist())
