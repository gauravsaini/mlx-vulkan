import mlx.core as mx
import numpy as np

def run_test():
    a = mx.array([-1.0, 0.0, 1.0, 2.0], dtype=mx.bfloat16)
    b = mx.exp(a)
    
    # MLX evaluates bfloat16 using the standard float32/float16 hardware paths or CPU fallbacks
    # if our shader works, it will output a mlx array.
    mx.eval(b)
    
    print("a:", a)
    print("exp(a):", b)
    print("dtype:", b.dtype)
    print("numpy equivalent:", np.exp(np.array(a, dtype=np.float32)))

run_test()
