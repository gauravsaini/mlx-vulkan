import mlx.core as mx
import numpy as np

data = np.arange(10, dtype=np.float16)
mx_data = mx.array(data)

a = mx.argmax(mx_data, axis=0)
b = np.argmax(data, axis=0)
mx.eval(a)
print("Data:", data)
print("Argmax MLX:", a.item())
print("Argmax NP:", b.item())
