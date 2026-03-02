import mlx.core as mx
import numpy as np

data = np.random.rand(100).astype(np.float16)
mx_data = mx.array(data)

a = mx.argmax(mx_data, axis=0)
b = np.argmax(data, axis=0)
mx.eval(a)
print("Argmax MLX:", a.item())
print("Argmax NP:", b.item())
if a.item() != b.item():
    print("Value MLX:", data[a.item()])
    print("Value NP :", data[b.item()])
