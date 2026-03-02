import mlx.core as mx
try:
    a = mx.array([5.0] * 256)
    res = mx.sum(a, stream=mx.gpu)
    mx.eval(res)
    print(f"first res: {res}")
except Exception as e:
    print(f"Error: {e}")
