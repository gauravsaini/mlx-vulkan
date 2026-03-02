import mlx.core as mx
import numpy as np

def run_test():
    dtypes = [(mx.float32, np.float32), (mx.float16, np.float16)]
    for mx_dt, np_dt in dtypes:
        for op in ["sum", "prod", "min", "max"]:
            a_npy = np.array([float('nan'), 1.0, 2.0], dtype=np_dt)
            a_mlx = mx.array(a_npy)
            
            ref = getattr(np, op)(a_npy)
            # MLX operations do not always have the same name natively on the mx module
            if op == "sum": mlx_op = mx.sum
            elif op == "prod": mlx_op = mx.prod
            elif op == "min": mlx_op = mx.min
            elif op == "max": mlx_op = mx.max
            
            out = mlx_op(a_mlx)
            
            ref = np.array(ref)
            out = np.array(out)
            if not np.array_equal(out, ref, equal_nan=True):
                print(f"FAILED {mx_dt} {op}: expected {ref}, got {out}")
            else:
                print(f"PASSED {mx_dt} {op}")

run_test()
