import mlx.core as mx
import time

gpu = mx.gpu
a = mx.array([5.0] * 256)

res = mx.sum(a, stream=gpu)
time.sleep(1)
mx.eval(res)

print(f"[TEST] GPU res: {res}")
