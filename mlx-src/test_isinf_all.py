import mlx.core as mx

for dtype_str, dtype in [("float32", mx.float32), ("float16", mx.float16), ("bfloat16", mx.bfloat16)]:
    x = mx.array([0.0, 1.0, float("inf"), float("-inf"), float("nan")]).astype(dtype)
    print(f"--- {dtype_str} ---")
    print("isinf:   ", mx.isinf(x).tolist())
    print("isnan:   ", mx.isnan(x).tolist())
    print("isneginf:", mx.isneginf(x).tolist())
    print("logical_not:", mx.logical_not(mx.array([True, False, True, False, False])).tolist())

