import mlx.core as mx
import numpy as np

def run_test():
    shape = (5, 5, 1, 5, 5)
    data = np.random.rand(*shape).astype(np.float32)
    mx_data = mx.array(data)

    t = (0, 1, 2, 3, 4)
    a = (0,)

    mx_data_t = mx.transpose(mx_data, t)
    data_t = np.transpose(data, t)

    a_mlx = mx.sum(mx_data_t, axis=a)
    a_np = np.sum(data_t, axis=a)

    mx.eval(a_mlx)
    if not np.allclose(a_mlx, a_np):
        print("FAILED! sum MLX:", np.sum(np.array(a_mlx)), "NP:", np.sum(a_np))
    else:
        print("PASSED! sum:", np.sum(np.array(a_mlx)))

print("Run 1:")
run_test()
print("Run 2:")
run_test()
