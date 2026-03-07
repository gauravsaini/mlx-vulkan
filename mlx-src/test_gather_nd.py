import mlx.core as mx

# 1D gather with 2D indices (mx.take, axis=0)
src = mx.random.normal(shape=(10, 5))
idx = mx.random.randint(shape=(3, 4), low=0, high=10)

out1 = mx.take(src, idx, axis=0)

import numpy as np
out_np = np.take(np.array(src), np.array(idx), axis=0)

if np.allclose(out1, out_np):
    print("Multi-axis Gather MATCH!")
else:
    print("Multi-axis Gather FAILED!")
