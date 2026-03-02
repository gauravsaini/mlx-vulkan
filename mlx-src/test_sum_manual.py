import mlx.core as mx
import numpy as np

# Test 1D Sum
data = np.ones(5).astype(np.float32)
mx_data = mx.array(data)
a = mx.sum(mx_data, axis=0)
b = np.sum(data, axis=0)
mx.eval(a)
print("1D Sum Match:", a.item() == b.item(), "MLX:", a.item(), "NP:", b.item())

# Test ND Sum
data2 = np.ones((5, 5, 1, 5, 5)).astype(np.float32)
mx_data2 = mx.array(data2)
a2 = mx.sum(mx_data2, axis=0)
b2 = np.sum(data2, axis=0)
mx.eval(a2)
print("ND Sum Match:", (np.array(a2) == b2).all())
if not (np.array(a2) == b2).all():
    print("ND Sum MLX first val:", np.array(a2).flatten()[0])
    print("ND Sum NP first val:", b2.flatten()[0])
