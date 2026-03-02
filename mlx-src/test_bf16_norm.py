import mlx.core as mx

def run_test():
    mx.set_default_device(mx.gpu)
    a = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)
    
    # LayerNorm
    ln_w = mx.array([1.0, 1.0], dtype=mx.bfloat16)
    ln_b = mx.array([0.0, 0.0], dtype=mx.bfloat16)
    
    out_ln = mx.fast.layer_norm(a, ln_w, ln_b, 1e-5)
    mx.eval(out_ln)
    print("LayerNorm bfloat16:\n", out_ln, "dtype:", out_ln.dtype)

    # RMSNorm
    rms_w = mx.array([1.0, 1.0], dtype=mx.bfloat16)
    out_rms = mx.fast.rms_norm(a, rms_w, 1e-5)
    mx.eval(out_rms)
    print("RMSNorm bfloat16:\n", out_rms, "dtype:", out_rms.dtype)

run_test()
