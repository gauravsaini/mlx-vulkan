import mlx.core as mx

def run_test():
    a = mx.array([1, 2, 3], dtype=mx.int32)
    p = mx.prod(a)
    print("arr:", a)
    print("prod:", p, "dtype:", p.dtype)

run_test()
