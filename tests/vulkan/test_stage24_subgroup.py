#!/usr/bin/env python3
"""Stage 24: Workgroup size tuning infrastructure test.

Verifies that the Vulkan device correctly reports subgroup_size and
preferred_workgroup_size via device_info(), and that these values are
in the expected range for any real Vulkan physical device.
"""

import sys

try:
    import mlx.core as mx
except ImportError as e:
    print(f"FAIL: could not import mlx.core: {e}")
    sys.exit(1)


def test_subgroup_info_present():
    info = mx.metal.device_info()
    assert "subgroup_size" in info, (
        "FAIL: 'subgroup_size' key missing from device_info()"
    )
    assert "preferred_workgroup_size" in info, (
        "FAIL: 'preferred_workgroup_size' key missing from device_info()"
    )
    sg = int(info["subgroup_size"])
    wg = int(info["preferred_workgroup_size"])
    print(f"  subgroup_size            = {sg}")
    print(f"  preferred_workgroup_size = {wg}")
    assert sg > 0, f"FAIL: subgroup_size={sg} must be > 0"
    assert sg <= 128, f"FAIL: subgroup_size={sg} unrealistically large"
    # subgroup_size must be a power of 2
    assert (sg & (sg - 1)) == 0, f"FAIL: subgroup_size={sg} is not a power of 2"
    assert wg >= sg, f"FAIL: preferred_workgroup_size={wg} < subgroup_size={sg}"
    assert wg <= 256, f"FAIL: preferred_workgroup_size={wg} > 256 cap"
    # Must be a multiple of subgroup_size
    assert wg % sg == 0, (
        f"FAIL: preferred_workgroup_size={wg} not a multiple of subgroup_size={sg}"
    )
    return True


def test_device_info_existing_fields():
    """Ensure we didn't break existing device_info() fields."""
    info = mx.metal.device_info()
    for key in ("device_name", "architecture", "memory_size"):
        assert key in info, f"FAIL: existing key '{key}' missing from device_info()"
    return True


def test_basic_gpu_compute_still_works():
    """Sanity-check that GPU compute still executes correctly after the changes."""
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    mx.eval(c)
    expected = [5.0, 7.0, 9.0]
    for i, (got, exp) in enumerate(zip(c.tolist(), expected)):
        assert abs(got - exp) < 1e-5, f"FAIL: c[{i}]={got} != {exp}"
    return True


def main():
    tests = [
        ("subgroup info present in device_info()", test_subgroup_info_present),
        ("existing device_info() fields intact", test_device_info_existing_fields),
        ("basic GPU compute still works", test_basic_gpu_compute_still_works),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL ({name}): unexpected exception: {e}")
            failed += 1

    total = passed + failed
    print(f"\nResults: {passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
