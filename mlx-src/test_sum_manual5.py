import mlx.core as mx
import numpy as np

shape = (5, 5, 1, 5, 5)
data = np.random.rand(*shape).astype(np.float32)
mx_data = mx.array(data)
t = (0, 1, 2, 3, 4)
mx_data_t = mx.transpose(mx_data, t)
mx.eval(mx_data_t)
print("mx_data_t flat 0-5:", np.array(mx_data_t).flatten()[:5])

a = (0,)
a_mlx = mx.sum(mx_data_t, axis=a)
mx.eval(a_mlx)
if np.sum(np.array(a_mlx)) == 0.0:
    print("FAILED! sum MLX sum: 0.0")
else:
    print("PASSED! sum:", np.sum(np.array(a_mlx)))
