import mlx.core as mx
import numpy as np

a_mlx = mx.random.randint(shape=(32, 32, 32), low=-100, high=100)
rev_idx = mx.arange(31, -1, -1)

c1 = mx.cumsum(a_mlx[rev_idx, :, :], axis=0)[rev_idx, :, :]
c2 = mx.cumsum(a_mlx, axis=0, inclusive=True, reverse=True)

if not mx.array_equal(c1, c2):
    print("axis=0 FAILED")
    diff = np.where(np.array(c1) != np.array(c2))
    idx = (diff[0][0], diff[1][0], diff[2][0])
    print(f"At {idx}: c1 (expected) = {c1[idx]}, c2 (got) = {c2[idx]}")
else:
    print("axis=0 MATCH")
