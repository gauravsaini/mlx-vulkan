import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)

def np_softmax_f16(x, axis):
    # force float16 math exactly
    x = x.astype(np.float16)
    maxx = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - maxx, dtype=np.float16)
    sumx = np.sum(ex, axis=axis, keepdims=True, dtype=np.float16)
    return (ex / sumx).astype(np.float16)

def np_softmax_f32(x, axis):
    # force float32 math
    x = x.astype(np.float32)
    maxx = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - maxx)
    sumx = np.sum(ex, axis=axis, keepdims=True)
    return (ex / sumx).astype(np.float16)

np.random.seed(42)  # This seed made iter=0 fail on axes=1 in the other script
a_npy = np.random.randn(16, 8, 32).astype(np.float16)
a_mlx = mx.array(a_npy)

axes = 1
b_npy_f16 = np_softmax_f16(a_npy, axes)
b_npy_f32 = np_softmax_f32(a_npy, axes)
b_mlx = mx.softmax(a_mlx, axes)
mx.eval(b_mlx)
b_mlx_np = np.array(b_mlx)

print("DIFF vs Numpy F16 math:", np.max(np.abs(b_npy_f16 - b_mlx_np)))
print("DIFF vs Numpy F32 math:", np.max(np.abs(b_npy_f32 - b_mlx_np)))

