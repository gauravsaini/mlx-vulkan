import mlx.core as mx
import numpy as np

x_npy = np.random.randn(5, 1, 5, 1, 5, 1).astype(np.float32)
x_mlx = mx.array(x_npy)

shape = [5, 5, 5, 1, 5, 1]
y_mlx = mx.broadcast_to(x_mlx, shape)
z_mlx = mx.sum(y_mlx, axis=(0,))

mx.eval(z_mlx)
print("z_mlx shape:", z_mlx.shape)
print("z_mlx strides:", z_mlx.strides)

b_mlx = z_mlx / 1000.0
mx.eval(b_mlx)
print("b_mlx shape:", b_mlx.shape)
print("b_mlx strides:", b_mlx.strides)

print("z_mlx max:", np.max(np.array(z_mlx)), "min:", np.min(np.array(z_mlx)))
print("b_mlx max:", np.max(np.array(b_mlx)), "min:", np.min(np.array(b_mlx)))
