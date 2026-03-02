import mlx.core as mx
import numpy as np

def run_test():
    shape = (3, 3, 3)
    np.random.seed(0)
    x = np.random.uniform(0, 2, size=shape).astype(np.float32)

    y = mx.array(x)
    r_npy = np.sum(x, axis=(0, 2))
    r_mlx = mx.sum(y, axis=(0, 2))

    mx.eval(r_mlx)

    if not np.allclose(r_npy, r_mlx, atol=1e-4):
        print("FAILED!")
        print("MLX:", np.array(r_mlx).tolist())
        print("NPY:", r_npy.tolist())
    else:
        print("PASSED!")

print("Run 1:")
run_test()
print("Run 2:")
run_test()
