import mlx.core as mx

# Force CPU device
mx.set_default_device(mx.cpu)

x16 = mx.array([0.0]*20).astype(mx.float16)
print("CPU Float16 0.0 isinf:", mx.isinf(x16).tolist()[:5])

x16_inf = mx.array([float("inf")]*5).astype(mx.float16)
print("CPU Float16 inf isinf:", mx.isinf(x16_inf).tolist())
