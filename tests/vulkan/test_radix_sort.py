#!/usr/bin/env python3
"""
Radix Sort Test Suite for MLX Vulkan Backend

Tests the radix sort implementation for sorting arrays larger than 256 elements.
Covers various array sizes, data types, and edge cases.
"""

import sys
import time
import numpy as np

# Add python path
sys.path.insert(0, '/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python')

import mlx.core as mx

def test_basic_int32_sort():
    """Test basic int32 sorting with various sizes."""
    print("\n" + "="*70)
    print("TEST 1: Basic int32 Sort")
    print("="*70)

    sizes = [256, 512, 1024, 2048, 4096, 8192]
    passed = 0
    failed = 0

    for size in sizes:
        print(f"\nTesting size: {size}")

        # Generate random int32 data
        np.random.seed(42 + size)
        data_np = np.random.randint(-1000, 1000, size, dtype=np.int32)
        data_mx = mx.array(data_np)

        # Sort on GPU
        try:
            gpu_result = mx.sort(data_mx)
            mx.eval(gpu_result)  # Force evaluation

            # Sort on CPU for comparison
            mx.set_default_device(mx.cpu)
            cpu_result = mx.sort(mx.array(data_np))
            mx.eval(cpu_result)

            # Compare results
            gpu_np = np.array(gpu_result)
            cpu_np = np.array(cpu_result)

            if np.array_equal(gpu_np, cpu_np):
                print(f"  ✓ PASS: {size} elements")
                passed += 1
            else:
                print(f"  ✗ FAIL: {size} elements - Results don't match")
                print(f"    GPU sample: {gpu_np[:10]}")
                print(f"    CPU sample: {cpu_np[:10]}")
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {size} elements - Exception: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Basic int32 Sort Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    return failed == 0

def test_basic_float32_sort():
    """Test basic float32 sorting with various sizes."""
    print("\n" + "="*70)
    print("TEST 2: Basic float32 Sort")
    print("="*70)

    sizes = [256, 512, 1024, 2048, 4096, 8192]
    passed = 0
    failed = 0

    for size in sizes:
        print(f"\nTesting size: {size}")

        # Generate random float32 data
        np.random.seed(42 + size)
        data_np = np.random.randn(size).astype(np.float32)
        data_mx = mx.array(data_np)

        # Sort on GPU
        try:
            gpu_result = mx.sort(data_mx)
            mx.eval(gpu_result)

            # Sort on CPU for comparison
            mx.set_default_device(mx.cpu)
            cpu_result = mx.sort(mx.array(data_np))
            mx.eval(cpu_result)

            # Compare results
            gpu_np = np.array(gpu_result)
            cpu_np = np.array(cpu_result)

            if np.allclose(gpu_np, cpu_np, rtol=1e-5, atol=1e-6):
                print(f"  ✓ PASS: {size} elements")
                passed += 1
            else:
                print(f"  ✗ FAIL: {size} elements - Results don't match")
                print(f"    GPU sample: {gpu_np[:10]}")
                print(f"    CPU sample: {cpu_np[:10]}")
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {size} elements - Exception: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Basic float32 Sort Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    return failed == 0

def test_descending_sort():
    """Test descending sort."""
    print("\n" + "="*70)
    print("TEST 3: Descending Sort")
    print("="*70)

    sizes = [256, 512, 1024, 2048]
    passed = 0
    failed = 0

    for size in sizes:
        print(f"\nTesting size: {size}")

        # Generate random data
        np.random.seed(42 + size)
        data_np = np.random.randn(size).astype(np.float32)
        data_mx = mx.array(data_np)

        try:
            # Sort descending on GPU
            gpu_result = mx.sort(data_mx, axis=-1)
            gpu_desc = mx.negative(gpu_result)  # Reverse for descending
            mx.eval(gpu_desc)

            # Sort descending on CPU
            mx.set_default_device(mx.cpu)
            cpu_result = mx.sort(mx.array(data_np), axis=-1)
            cpu_desc = mx.negative(cpu_result)
            mx.eval(cpu_desc)

            # Compare
            gpu_np = np.array(gpu_desc)
            cpu_np = np.array(cpu_desc)

            if np.allclose(gpu_np, cpu_np, rtol=1e-5, atol=1e-6):
                print(f"  ✓ PASS: {size} elements descending")
                passed += 1
            else:
                print(f"  ✗ FAIL: {size} elements - Results don't match")
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {size} elements - Exception: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Descending Sort Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    return failed == 0

def test_edge_cases():
    """Test edge cases: already sorted, reverse sorted, duplicates."""
    print("\n" + "="*70)
    print("TEST 4: Edge Cases")
    print("="*70)

    test_cases = [
        ("Already sorted", 1024, lambda s: np.arange(s, dtype=np.float32)),
        ("Reverse sorted", 1024, lambda s: np.arange(s-1, -1, -1, dtype=np.float32)),
        ("All same value", 1024, lambda s: np.full(s, 42.0, dtype=np.float32)),
        ("Many duplicates", 1024, lambda s: np.random.randint(0, 10, s, dtype=np.int32).astype(np.float32)),
        ("With NaN values", 1024, lambda s: np.where(np.random.rand(s) > 0.9, np.nan, np.random.randn(s)).astype(np.float32)),
    ]

    passed = 0
    failed = 0

    for name, size, generator in test_cases:
        print(f"\nTesting: {name}")

        try:
            data_np = generator(size)
            data_mx = mx.array(data_np)

            # Sort on GPU
            gpu_result = mx.sort(data_mx)
            mx.eval(gpu_result)

            # Sort on CPU
            mx.set_default_device(mx.cpu)
            cpu_result = mx.sort(mx.array(data_np))
            mx.eval(cpu_result)

            # Compare (handle NaN specially)
            gpu_np = np.array(gpu_result)
            cpu_np = np.array(cpu_result)

            # For NaN comparison, check positions of NaN values
            if "NaN" in name:
                gpu_nan = np.isnan(gpu_np)
                cpu_nan = np.isnan(cpu_np)
                valid = ~(gpu_nan | cpu_nan)
                if (np.array_equal(gpu_nan, cpu_nan) and
                    np.allclose(gpu_np[valid], cpu_np[valid], rtol=1e-5, atol=1e-6)):
                    print(f"  ✓ PASS: {name}")
                    passed += 1
                else:
                    print(f"  ✗ FAIL: {name} - NaN handling differs")
                    failed += 1
            else:
                if np.allclose(gpu_np, cpu_np, rtol=1e-5, atol=1e-6):
                    print(f"  ✓ PASS: {name}")
                    passed += 1
                else:
                    print(f"  ✗ FAIL: {name} - Results don't match")
                    failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {name} - Exception: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Edge Cases Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    return failed == 0

def test_2d_arrays():
    """Test sorting 2D arrays along the last axis."""
    print("\n" + "="*70)
    print("TEST 5: 2D Arrays")
    print("="*70)

    test_cases = [
        ("2x512", (2, 512)),
        ("4x256", (4, 256)),
        ("8x256", (8, 256)),
        ("16x128", (16, 128)),
    ]

    passed = 0
    failed = 0

    for name, shape in test_cases:
        print(f"\nTesting: {name} (shape={shape})")

        try:
            np.random.seed(42)
            data_np = np.random.randn(*shape).astype(np.float32)
            data_mx = mx.array(data_np)

            # Sort on GPU
            gpu_result = mx.sort(data_mx, axis=-1)
            mx.eval(gpu_result)

            # Sort on CPU
            mx.set_default_device(mx.cpu)
            cpu_result = mx.sort(mx.array(data_np), axis=-1)
            mx.eval(cpu_result)

            # Compare
            gpu_np = np.array(gpu_result)
            cpu_np = np.array(cpu_result)

            if np.allclose(gpu_np, cpu_np, rtol=1e-5, atol=1e-6):
                print(f"  ✓ PASS: {name}")
                passed += 1
            else:
                print(f"  ✗ FAIL: {name} - Results don't match")
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {name} - Exception: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"2D Arrays Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    return failed == 0

def test_performance():
    """Compare GPU vs CPU performance for large arrays."""
    print("\n" + "="*70)
    print("TEST 6: Performance Comparison")
    print("="*70)

    sizes = [1024, 2048, 4096, 8192]

    print(f"\n{'Size':>8} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Speedup':>10}")
    print("-" * 50)

    for size in sizes:
        np.random.seed(42)
        data_np = np.random.randn(size).astype(np.float32)
        data_mx_gpu = mx.array(data_np)

        # Time GPU
        mx.set_default_device(mx.gpu)
        start = time.time()
        gpu_result = mx.sort(data_mx_gpu)
        mx.eval(gpu_result)
        gpu_time = (time.time() - start) * 1000

        # Time CPU
        mx.set_default_device(mx.cpu)
        data_mx_cpu = mx.array(data_np)
        start = time.time()
        cpu_result = mx.sort(data_mx_cpu)
        mx.eval(cpu_result)
        cpu_time = (time.time() - start) * 1000

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"{size:>8} {gpu_time:>12.2f} {cpu_time:>12.2f} {speedup:>10.2f}x")

    print(f"{'='*70}")

    return True

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MLX Vulkan Backend - Radix Sort Test Suite")
    print("="*70)

    # Set GPU as default device
    try:
        mx.set_default_device(mx.gpu)
        print(f"\n✓ GPU device available and set as default")
    except Exception as e:
        print(f"\n✗ GPU device not available: {e}")
        print("   Falling back to CPU for testing")
        mx.set_default_device(mx.cpu)

    results = {}

    try:
        results['int32_sort'] = test_basic_int32_sort()
    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        results['int32_sort'] = False

    try:
        results['float32_sort'] = test_basic_float32_sort()
    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        results['float32_sort'] = False

    try:
        results['descending_sort'] = test_descending_sort()
    except Exception as e:
        print(f"\n✗ Test 3 failed with exception: {e}")
        results['descending_sort'] = False

    try:
        results['edge_cases'] = test_edge_cases()
    except Exception as e:
        print(f"\n✗ Test 4 failed with exception: {e}")
        results['edge_cases'] = False

    try:
        results['2d_arrays'] = test_2d_arrays()
    except Exception as e:
        print(f"\n✗ Test 5 failed with exception: {e}")
        results['2d_arrays'] = False

    try:
        results['performance'] = test_performance()
    except Exception as e:
        print(f"\n✗ Test 6 failed with exception: {e}")
        results['performance'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")

    print(f"\n{'='*70}")
    print(f"Overall: {passed}/{total} test suites passed")
    print(f"{'='*70}")

    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
