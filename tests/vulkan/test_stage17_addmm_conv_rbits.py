#!/usr/bin/env python3
"""
Stage 17: AddMM, Convolution, and RandomBits Tests

Tests GPU dispatch for:
1. AddMM (matrix multiplication + vector additions)
2. Convolution (2D convolutions)
3. RandomBits (Threefry PRNG sequence generation)
"""

import sys
import traceback
import numpy as np
import mlx.core as mx

PASS = 0
FAIL = 0

def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {msg}")

def test_addmm():
    print("-- Testing AddMM --")
    
    # A * B + C
    a_np = np.random.randn(32, 64).astype(np.float32)
    b_np = np.random.randn(64, 128).astype(np.float32)
    c_np = np.random.randn(32, 128).astype(np.float32)
    
    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)
    c_mx = mx.array(c_np)
    
    # MLX matrix multiplication handles AddMM inherently when expressed as:
    # alpha * (A @ B) + beta * C
    # The framework matches the `addmm` pattern during graph optimizations.
    out_mx = mx.addmm(c_mx, a_mx, b_mx, alpha=1.0, beta=1.0)
    mx.eval(out_mx)
    result = np.array(out_mx)
    
    expected = (a_np @ b_np) + c_np
    check("addmm (alpha=1, beta=1)", np.allclose(result, expected, atol=1e-4), "Output mismatch")

    # Test alpha and beta scaling
    out_mx_scaled = mx.addmm(c_mx, a_mx, b_mx, alpha=2.0, beta=0.5)
    mx.eval(out_mx_scaled)
    result_scaled = np.array(out_mx_scaled)
    
    expected_scaled = 2.0 * (a_np @ b_np) + 0.5 * c_np
    check("addmm (alpha=2, beta=0.5)", np.allclose(result_scaled, expected_scaled, atol=1e-4), "Scaled output mismatch")

def test_random_bits():
    print("-- Testing RandomBits (PRNG) --")
    
    # Seed the RNG to test deterministic output
    mx.random.seed(42)
    
    # Generate random uniform floats, which under the hood calls RandomBits
    a = mx.random.uniform(shape=(10, 10))
    mx.eval(a)
    result_a = np.array(a)
    
    mx.random.seed(42)
    b = mx.random.uniform(shape=(10, 10))
    mx.eval(b)
    result_b = np.array(b)
    
    check("random uniform exact reproducibility", np.allclose(result_a, result_b), "Seed did not produce identical arrays")
    check("random uniform range", np.all((result_a >= 0) & (result_a < 1)), "Output outside [0, 1)")
    
    # Generate normal distribution
    c = mx.random.normal(shape=(10, 10))
    mx.eval(c)
    result_c = np.array(c)
    check("random normal shape", result_c.shape == (10, 10), "Bad normal distribution shape")

def test_convolution():
    print("-- Testing Convolution 2D --")
    import mlx.nn as nn
    
    # N, H, W, C layout for MLX
    # Batch=1, H=8, W=8, C=3
    x_np = np.random.randn(1, 8, 8, 3).astype(np.float32)
    x = mx.array(x_np)
    
    # Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False)
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False)
    
    out = conv(x)
    mx.eval(out)
    result = np.array(out)
    
    check("conv2d shape", result.shape == (1, 8, 8, 16), f"Expected (1, 8, 8, 16) but got {result.shape}")
    
    # Test valid padding and stride
    conv_valid = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, stride=2, bias=False)
    out_valid = conv_valid(x)
    mx.eval(out_valid)
    result_valid = np.array(out_valid)
    
    # Output spatial dim: floor((8 - 3 + 0)/2) + 1 = floor(5/2) + 1 = 3 -> 3x3 
    check("conv2d valid+stride shape", result_valid.shape == (1, 3, 3, 16), f"Expected (1, 3, 3, 16) got {result_valid.shape}")


if __name__ == "__main__":
    print("━" * 50)
    print("  Stage 17: AddMM, Convolution, RandomBits Tests")
    print("━" * 50)

    try:
        test_addmm()
        test_random_bits()
        test_convolution()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        FAIL += 1

    print()
    print("━" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("━" * 50)
    sys.exit(1 if FAIL > 0 else 0)
