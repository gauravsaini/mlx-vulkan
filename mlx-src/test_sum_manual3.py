import mlx.core as mx
import numpy as np

shape = (5, 5, 1, 5, 5)
data = np.random.rand(*shape).astype(np.float32)
mx_data = mx.array(data)

# test Transpose + Sum
t = (0, 1, 4, 3, 2)
a = (0,)

mx_data_t = mx.transpose(mx_data, t)
data_t = np.transpose(data, t)

a_mlx = mx.sum(mx_data_t, axis=a)
a_np = np.sum(data_t, axis=a)

mx.eval(a_mlx)
if not np.allclose(a_mlx, a_np):
    print("FAILED")
    print("MLX:\n", np.array(a_mlx).flatten()[:10])
    print("NP:\n", a_np.flatten()[:10])
else:
    print("PASSED")
