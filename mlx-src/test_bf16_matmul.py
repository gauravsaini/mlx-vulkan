import mlx.core as mx

def run_test():
    a = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)
    b = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.bfloat16)
    
    # test MATMUL
    c = mx.matmul(a, b)
    mx.eval(c)
    print("matmul(a,b) bfloat16:\n", c, "dtype:", c.dtype)

    a2 = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float16)
    b2 = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.float16)
    
    c2 = mx.matmul(a2, b2)
    mx.eval(c2)
    print("matmul(a2,b2) float16:\n", c2, "dtype:", c2.dtype)

run_test()
