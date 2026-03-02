import mlx.core as mx

gpu = mx.gpu
a = mx.array([5.0] * 256)

# Run sum
res = mx.sum(a, stream=gpu)
mx.eval(res)

print(f"[TEST] GPU res: {res}")
