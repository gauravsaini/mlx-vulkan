import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)

def np_softmax(x, axis):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / np.sum(ex, axis=axis, keepdims=True)

for i in range(10):
    for dtype, atol in [(np.float32, 1e-6), (np.float16, 1e-3)]:
        a_npy = np.random.randn(16, 8, 32).astype(dtype)
        a_mlx = mx.array(a_npy)

        for axes in (None, 0, 1, 2, (0, 1), (1, 2), (0, 2), (0, 1, 2)):
            b_npy = np_softmax(a_npy, axes)
            b_mlx = mx.softmax(a_mlx, axes)
            
            if not np.allclose(b_npy, b_mlx, atol=atol):
                print(f"FAILED iter={i} dtype={dtype} axes={axes}")
                print("DIFF MAX:", np.max(np.abs(b_npy - np.array(b_mlx))))
                exit(1)

print("SUCCESS")
