import mlx.core as mx
import numpy as np

def run_test():
    shape = (5, 5, 1, 5, 5)
    np.random.seed(42)
    x_npy = (np.random.randn(*shape) * 128).astype(np.int32)
    x_mlx = mx.array(x_npy)

    t = (0, 1, 2, 3, 4)
    a = (0,)

    mx_data_t = mx.transpose(x_mlx, t)
    data_t = np.transpose(x_npy, t)

    a_mlx = mx.sum(mx_data_t, axis=a)
    a_np = np.sum(data_t, axis=a)

    mx.eval(a_mlx)
    if not np.all(np.array(a_mlx) == a_np):
        print("FAILED! sum MLX:", np.array(a_mlx).flatten()[:5])
        print("NP:", a_np.flatten()[:5])
    else:
        print("PASSED!")

print("Run 1:")
run_test()
