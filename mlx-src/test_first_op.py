import mlx.core as mx

gpu = mx.gpu

res = mx.array([5.0], stream=gpu) * 2.0
mx.eval(res)
print(f"first op: {res}")
