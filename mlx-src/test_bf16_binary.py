import mlx.core as mx

def run_test():
    a = mx.array([1.5, 2.5, -3.5, 4.0], dtype=mx.bfloat16)
    b = mx.array([2.0, 0.5, 1.0, 1.0], dtype=mx.bfloat16)
    
    # Test Addition
    c = a + b
    mx.eval(c)
    print("a + b:", c)
run_test()
