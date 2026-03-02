import mlx.core as mx
import numpy as np

for shape in [(10, 12, 13), (1, 10), (10, 1), (12, 13)]:
    data = np.random.rand(*shape).astype(np.float32)
    mx_data = mx.array(data)
    for axis in range(len(shape)):
        a = mx.argmax(mx_data, axis=axis)
        b = np.argmax(data, axis=axis)
        mx.eval(a)
        res = "PASS" if a.tolist() == b.tolist() else "FAIL"
        print(f"Shape {shape} Axis {axis}: {res}")
