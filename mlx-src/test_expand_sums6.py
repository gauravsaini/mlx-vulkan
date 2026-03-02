import mlx.core as mx
import numpy as np

z_mlx = mx.array([1.5, -2.5], dtype=mx.float32)
b_mlx = z_mlx / 1000
mx.eval(b_mlx)

print("dtype:", b_mlx.dtype)
print("b_mlx:", np.array(b_mlx))
