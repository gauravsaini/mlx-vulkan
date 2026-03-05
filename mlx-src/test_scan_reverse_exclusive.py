import mlx.core as mx

# Fails natively on test_ops.py line 2092
a_mlx = mx.random.randint(shape=(32, 32, 32), low=-100, high=100)
rev_idx = mx.arange(31, -1, -1)

# Debug Axis 1
c1 = mx.cumsum(a_mlx[:, rev_idx, :], axis=1)[:, rev_idx, :][:, 1:, :]
c2 = mx.cumsum(a_mlx, axis=1, inclusive=False, reverse=True)[:, :-1, :]

import numpy as np
if not np.array_equal(c1, c2):
   print("Axis=1 Exclusive Reverse FAILED!")
   diff = np.where(np.array(c1) != np.array(c2))
   idx = (diff[0][0], diff[1][0], diff[2][0])
   print(f"At {idx}: c1 (expected) = {c1[idx]}, c2 (got) = {c2[idx]}")
   print(f"c1 row (expected): {c1[idx[0], :5, idx[2]]}")
   print(f"c2 row (got): {c2[idx[0], :5, idx[2]]}")
else:
   print("Axis=1 Exclusive Reverse MATCH!")

