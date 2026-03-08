#!/usr/bin/env python3
"""
Stage 25: Linux AMD bring-up smoke for the Vulkan backend.

Usage:
  python tests/vulkan/test_stage25_amd_bringup.py

Optional environment flags:
  MLX_VULKAN_REQUIRE_AMD=1
    Fail if the detected GPU is not AMD.
"""

import os
import platform
import sys
import importlib.util

import numpy as np


def info_to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def vendor_looks_amd(info):
    text = " ".join(info_to_str(v).lower() for v in info.values())
    return "amd" in text or "advanced micro devices" in text or "radeon" in text


def check(name, fn, errors):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as exc:
        errors.append(f"{name}: {exc}")


def main():
    try:
        core_so = os.environ.get("MLX_CORE_SO")
        if core_so:
            import mlx

            spec = importlib.util.spec_from_file_location("mlx.core", core_so)
            mx = importlib.util.module_from_spec(spec)
            sys.modules["mlx.core"] = mx
            spec.loader.exec_module(mx)
            mlx.core = mx
        else:
            import mlx.core as mx
    except Exception as exc:
        print(f"FAIL: Cannot import mlx.core: {exc}")
        print("Install/build MLX first, for example: pip install -e mlx-src/")
        return 1

    print("=======================================")
    print("  STAGE 25: AMD Vulkan Bring-up Smoke")
    print("=======================================")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")

    errors = []
    require_amd = os.environ.get("MLX_VULKAN_REQUIRE_AMD", "0") == "1"

    check("mlx import", lambda: None, errors)

    def check_gpu():
        if not mx.is_available(mx.gpu):
            raise RuntimeError("mx.is_available(mx.gpu) returned false")
        count = mx.device_count(mx.gpu)
        if count < 1:
            raise RuntimeError(f"mx.device_count(mx.gpu) returned {count}")
        info = mx.device_info(mx.gpu)
        if isinstance(info, list):
            first = info[0]
        else:
            first = info
        print(f"  GPU info: {first}")
        if require_amd and not vendor_looks_amd(first):
            raise RuntimeError(f"AMD GPU required, got: {first}")

    def check_core_ops():
        x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        y = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
        out = x @ y
        mx.eval(out)
        got = np.array(out.tolist(), dtype=np.float32)
        ref = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        if not np.allclose(got, ref, atol=1e-4):
            raise RuntimeError(f"matmul mismatch: got {got.tolist()}")

        s = mx.softmax(mx.array([1.0, 2.0, 3.0], dtype=mx.float32))
        mx.eval(s)
        got = np.array(s.tolist(), dtype=np.float32)
        ref = np.exp(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        ref = ref / ref.sum()
        if not np.allclose(got, ref, atol=1e-4):
            raise RuntimeError(f"softmax mismatch: got {got.tolist()}")

    def check_python_bridge():
        a = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        mv = memoryview(a)
        if not mv.readonly:
            raise RuntimeError("memoryview unexpectedly writable")
        if tuple(mv.shape) != (2, 2):
            raise RuntimeError(f"memoryview shape mismatch: got {mv.shape}")

        np_val = np.asarray(a)
        np_ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        if not np.allclose(np_val, np_ref, atol=1e-4):
            raise RuntimeError(f"numpy export mismatch: got {np_val.tolist()}")

        list_val = mx.arange(6, dtype=mx.int32).reshape(2, 3).tolist()
        if list_val != [[0, 1, 2], [3, 4, 5]]:
            raise RuntimeError(f"tolist mismatch: got {list_val}")

        scalar_val = mx.array(7, dtype=mx.int32).tolist()
        if scalar_val != 7:
            raise RuntimeError(f"scalar tolist mismatch: got {scalar_val}")

        np_in = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)
        from_np = mx.array(np_in)
        got_np_in = np.asarray(from_np)
        if not np.allclose(got_np_in, np_in, atol=1e-4):
            raise RuntimeError(f"numpy import mismatch: got {got_np_in.tolist()}")

        rev = mx.arange(5, dtype=mx.int32)[::-1]
        rev_ref = np.array([4, 3, 2, 1, 0], dtype=np.int32)
        if rev.tolist() != rev_ref.tolist():
            raise RuntimeError(f"reverse tolist mismatch: got {rev.tolist()}")
        if not np.array_equal(np.asarray(rev), rev_ref):
            raise RuntimeError(f"reverse numpy export mismatch: got {np.asarray(rev).tolist()}")

        strided = mx.arange(6, dtype=mx.int32).reshape(2, 3)[:, ::2]
        strided_ref = np.array([[0, 2], [3, 5]], dtype=np.int32)
        if strided.tolist() != strided_ref.tolist():
            raise RuntimeError(f"strided tolist mismatch: got {strided.tolist()}")
        if not np.array_equal(np.asarray(strided), strided_ref):
            raise RuntimeError(f"strided numpy export mismatch: got {np.asarray(strided).tolist()}")

    def check_cpu_fallback_outputs():
        x = mx.isinf(mx.array([1, 2, 3], dtype=mx.int32))
        mx.eval(x)
        if x.tolist() != [False, False, False]:
            raise RuntimeError(f"isinf(int) mismatch: got {x.tolist()}")

        a = mx.zeros((2, 0), dtype=mx.float32)
        b = mx.zeros((0, 3), dtype=mx.float32)
        out = mx.matmul(a, b)
        mx.eval(out)
        got = np.array(out.tolist(), dtype=np.float32)
        ref = np.zeros((2, 3), dtype=np.float32)
        if not np.allclose(got, ref, atol=0.0):
            raise RuntimeError(f"empty-K matmul mismatch: got {got.tolist()}")

    def check_linalg_fallbacks():
        a = mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)

        q, r = mx.linalg.qr(a)
        mx.eval(q, r)
        qr = np.array((q @ r).tolist(), dtype=np.float32)
        if not np.allclose(qr, np.array(a.tolist(), dtype=np.float32), atol=1e-4):
            raise RuntimeError("qr reconstruction mismatch")

        s = mx.linalg.svd(a, compute_uv=False)
        mx.eval(s)
        sval = np.array(s.tolist(), dtype=np.float32)
        sref = np.linalg.svd(np.array(a.tolist(), dtype=np.float32), compute_uv=False)
        if not np.allclose(sval, sref, atol=1e-4):
            raise RuntimeError(f"svd mismatch: got {sval.tolist()}")

        inv = mx.linalg.inv(a)
        mx.eval(inv)
        inv_ref = np.linalg.inv(np.array(a.tolist(), dtype=np.float32))
        inv_val = np.array(inv.tolist(), dtype=np.float32)
        if not np.allclose(inv_val, inv_ref, atol=1e-4):
            raise RuntimeError(f"inverse mismatch: got {inv_val.tolist()}")

    check("gpu detect", check_gpu, errors)
    check("core ops", check_core_ops, errors)
    check("python bridge", check_python_bridge, errors)
    check("cpu fallback outputs", check_cpu_fallback_outputs, errors)
    check("linalg fallbacks", check_linalg_fallbacks, errors)

    print()
    if errors:
        for err in errors:
            print(f"  FAIL {err}")
        print(f"\nFAIL: {len(errors)} checks failed")
        return 1

    print("PASS: Vulkan bring-up smoke succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
