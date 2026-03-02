import mlx.core as mx
import numpy as np

gpu = mx.gpu
data = np.ones(256, dtype=np.float32) * 2.0
a = mx.array(data)

print(f"[TEST] input sum: {np.sum(data)}")

# NO WARMUP
# NO TOLIST BEFORE EVALUATING REDUCE

# Run sum
res = mx.sum(a, stream=gpu)
mx.eval(res)

print(f"[TEST] GPU res: {res}")
print(f"[TEST] GPU res type: {res.dtype}")
print(f"[TEST] RAW gpu array bytes: {res.tolist()}")
