import mlx.core as mx
import numpy as np

# Test 1D Sum 1
data = np.ones(5).astype(np.float32)
mx_data = mx.array(data)
a = mx.sum(mx_data, axis=0)
mx.eval(a)
print("1D Sum 1:", a.item())

# Test 1D Sum 2
data = np.ones(5).astype(np.float32)
mx_data = mx.array(data)
a = mx.sum(mx_data, axis=0)
mx.eval(a)
print("1D Sum 2:", a.item())

# Test 1D Sum 3
data = np.ones(5).astype(np.float32)
mx_data = mx.array(data)
a = mx.sum(mx_data, axis=0)
mx.eval(a)
print("1D Sum 3:", a.item())
