import mlx.core as mx
import numpy as np

shape = (5,)
data = np.ones(shape).astype(np.float16)
mx_data = mx.array(data)

a_mlx = mx.sum(mx_data)
a_np = np.sum(data)

mx.eval(a_mlx)
print("MLX f16 sum:", np.array(a_mlx).item(), "NP:", a_np)
