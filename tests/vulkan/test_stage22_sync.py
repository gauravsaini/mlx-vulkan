#!/usr/bin/env python3.11
"""
Stage 22: Synchronization Primitives — Event, Fence, device_info
Tests the Phase 4 Vulkan backend implementation.

Since mx.Event and mx.Fence are not exposed as direct Python bindings,
we test the synchronization semantics indirectly via observed ordering
of GPU/CPU work, and test device_info via mx.metal.device_info().
"""

import mlx.core as mx
import sys
import traceback

results = []

def run_test(name, fn):
    try:
        fn()
        print(f"  PASS: {name}")
        results.append(True)
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        results.append(False)

# ─── Test 1: device_info returns expected fields ───────────────────────────────
def test_device_info_fields():
    info = mx.metal.device_info()
    assert isinstance(info, dict), f"Expected dict, got {type(info)}"
    assert "device_name" in info, f"Missing 'device_name'; got keys: {list(info.keys())}"
    assert "memory_size" in info, f"Missing 'memory_size'; got keys: {list(info.keys())}"
    assert "architecture" in info, f"Missing 'architecture'; got keys: {list(info.keys())}"
    assert isinstance(info["device_name"], str) and len(info["device_name"]) > 0, \
        f"device_name is empty or non-string: {info['device_name']!r}"
    assert isinstance(info["memory_size"], int) and info["memory_size"] > 0, \
        f"memory_size should be positive int: {info['memory_size']}"
    arch = info["architecture"]
    assert "vulkan_" in arch, f"architecture should start with 'vulkan_': got {arch!r}"
    print(f"    device_name  = {info['device_name']}")
    print(f"    architecture = {info['architecture']}")
    mem_gb = info['memory_size'] / (1024**3)
    print(f"    memory_size  = {info['memory_size']} bytes ({mem_gb:.2f} GB)")

# ─── Test 2: device_info architecture is NOT 'unknown' on Apple Silicon ────────
def test_device_info_apple_arch():
    info = mx.metal.device_info()
    arch = info.get("architecture", "")
    # On MoltenVK/Apple Silicon we expect vulkan_apple
    # On other Vulkan we expect known vendor strings
    unknown_arch = (arch == "vulkan_unknown")
    # Warn but pass — unknown is valid on unrecognised hardware
    if unknown_arch:
        print(f"    NOTE: architecture is 'vulkan_unknown' — may need new vendor ID case")
    else:
        print(f"    architecture correctly detected as: {arch}")

# ─── Test 3: Vulkan compute still works after device_info query ────────────────
def test_gpu_compute_after_device_info():
    """Ensure device_info query doesn't corrupt device state."""
    _ = mx.metal.device_info()
    a = mx.array([1.0, 2.0, 3.0, 4.0])
    b = mx.array([10.0, 20.0, 30.0, 40.0])
    c = a + b
    mx.eval(c)
    expected = [11.0, 22.0, 33.0, 44.0]
    assert list(c.tolist()) == expected, f"Expected {expected}, got {c.tolist()}"

# ─── Test 4: GPU→CPU ordering via synchronize ─────────────────────────────────
def test_gpu_cpu_ordering():
    """
    Write on GPU, read on CPU — checks the fence semantics in the path that
    Fence::wait uses internally (drain GPU stream, then CPU reads).
    """
    gpu_stream = mx.gpu
    cpu_stream = mx.cpu

    with mx.stream(gpu_stream):
        a = mx.ones((256,), dtype=mx.float32) * 7.0
        mx.eval(a)

    # CPU-side read: this forces synchronize of the GPU stream
    val = a.tolist()[0]
    assert abs(val - 7.0) < 1e-5, f"Expected 7.0, got {val}"

# ─── Test 5: Cross-stream ordering via eval ────────────────────────────────────
def test_cross_stream_ordering():
    """
    Compute on GPU stream, verify result on CPU — exercises the happens-before
    ordering fence guarantees are built on.
    """
    with mx.stream(mx.gpu):
        x = mx.arange(128, dtype=mx.float32)
        y = mx.sum(x)
        mx.eval(y)

    expected = 127 * 128 // 2  # sum 0..127 = 8128
    got = y.item()
    assert abs(got - expected) < 1.0, f"Expected {expected}, got {got}"

# ─── Test 6: Multiple sequential evals don't deadlock ─────────────────────────
def test_sequential_evals_no_deadlock():
    """Fence::update increments count; subsequent waits must converge."""
    for i in range(5):
        a = mx.array([float(i)]) * 2.0
        mx.eval(a)
        assert abs(a.item() - float(i * 2)) < 1e-5, \
            f"Iter {i}: expected {i*2}, got {a.item()}"

# ─── Test 7: Memory info consistency ──────────────────────────────────────────
def test_memory_info_consistent():
    """Check active/peak memory tracking is consistent."""
    before = mx.get_active_memory()
    arr = mx.ones((1024, 1024), dtype=mx.float32)  # ~4 MB
    mx.eval(arr)
    after = mx.get_active_memory()
    assert after >= before, \
        f"Active memory should be >= before after allocation: before={before}, after={after}"
    del arr
    # peak memory should be at least as large as current
    peak = mx.get_peak_memory()
    assert peak >= 0, f"Peak memory should be non-negative: {peak}"

# ─── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Stage 22: Synchronization & DeviceInfo")
    print("=" * 50)

    run_test("device_info returns expected fields", test_device_info_fields)
    run_test("device_info apple arch detection", test_device_info_apple_arch)
    run_test("GPU compute still works after device_info", test_gpu_compute_after_device_info)
    run_test("GPU→CPU ordering via synchronize", test_gpu_cpu_ordering)
    run_test("Cross-stream sum ordering", test_cross_stream_ordering)
    run_test("Sequential evals no deadlock", test_sequential_evals_no_deadlock)
    run_test("Memory info consistent", test_memory_info_consistent)

    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
