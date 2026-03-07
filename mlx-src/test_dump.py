import mlx.core as mx
import numpy as np
mx.set_default_device(mx.gpu)
a = mx.array([-1.0, 1.0, 0.0, 1.0, -2.0, 3.0])
res = mx.logical_not(a)
mx.eval(res)
print("Bytes:", np.array(res).tobytes())
print("Primitive:", res.primitive)
if hasattr(res, "inputs"):
    print("Inputs:", [i.dtype for i in res.inputs])
else:
    print("No inputs attribute exposed to python, but we can check the output type:", res.dtype)
