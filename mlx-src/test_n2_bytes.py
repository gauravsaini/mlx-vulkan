import mlx.core as mx
mx.set_default_device(mx.gpu)
a = mx.array([0.0, 1.0])
res = mx.logical_not(a)
mx.eval(res)
import numpy as np
arr = np.array(res)
print("Bytes:", arr.tobytes())
print("Expected:", np.logical_not(np.array([0.0, 1.0])).tobytes())
