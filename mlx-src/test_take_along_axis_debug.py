import mlx.core as mx
import numpy as np

a_np = np.arange(8).reshape(2, 2, 2)
a_mlx = mx.array(a_np)
idx_np = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0])
idx_mlx = mx.array(idx_np)

mx.set_default_device(mx.gpu)

ax = 1
shape = [2] * 3
shape[ax] = 3

out_np = np.take_along_axis(a_np, idx_np.reshape(shape), axis=ax)
out_mlx = mx.take_along_axis(a_mlx, mx.reshape(idx_mlx, shape), axis=ax)

print("a_mlx dtype:", a_mlx.dtype, "itemsize:", a_mlx.itemsize)
print("idx_mlx dtype:", idx_mlx.dtype, "itemsize:", idx_mlx.itemsize)
print("out_mlx dtype:", out_mlx.dtype, "itemsize:", out_mlx.itemsize)

print("Expected output shape:", out_np.shape)
print("Expected:\n", out_np)
print("Vulkan:\n", np.array(out_mlx))
