import mlx.core as mx
import numpy as np

def run_test():
    x_npy = np.random.randn(5, 1, 5, 1, 5, 1).astype(np.float32)
    x_mlx = mx.array(x_npy)
    
    shape = [5, 5, 5, 5, 5, 1]
    y_npy = np.broadcast_to(x_npy, shape)
    y_mlx = mx.broadcast_to(x_mlx, shape)
    
    a = (5,)
    z_npy = np.sum(y_npy, axis=a)
    z_mlx = mx.sum(y_mlx, axis=a)
    mx.eval(z_mlx)
    
    print("NPY Sum Max:", np.max(z_npy), "Min:", np.min(z_npy))
    print("MLX Sum Max:", np.max(np.array(z_mlx)), "Min:", np.min(np.array(z_mlx)))

run_test()
