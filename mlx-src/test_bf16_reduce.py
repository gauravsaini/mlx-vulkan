import mlx.core as mx

def run_test():
    a = mx.array([1.5, 2.5, -3.5, 4.0], dtype=mx.bfloat16)

    s = mx.sum(a)
    mx.eval(s)
    print("sum(a):", s)

    m = mx.max(a)
    mx.eval(m)
    print("max(a):", m)

    b = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
    
    s_f16 = mx.sum(b)
    mx.eval(s_f16)
    print("sum(b_f16):", s_f16)

run_test()
