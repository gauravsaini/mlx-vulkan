import mlx.core as mx
print("Default device:", mx.default_device())
print("Device info:", mx.gpu.device_info() if mx.default_device() == mx.gpu else "CPU")
