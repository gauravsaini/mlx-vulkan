#!/usr/bin/env python3
"""
Stage 25: Linux Vulkan bring-up smoke for the backend.

Usage:
  python tests/vulkan/test_stage25_linux_vulkan_bringup.py

Optional environment flags:
  MLX_VULKAN_REQUIRE_VENDOR=amd|nvidia|intel
    Fail if the detected GPU vendor does not match the requested value.
  MLX_VULKAN_REQUIRE_AMD=1
    Legacy alias for MLX_VULKAN_REQUIRE_VENDOR=amd.
  MLX_VULKAN_REQUIRE_NVIDIA=1
    Legacy alias for MLX_VULKAN_REQUIRE_VENDOR=nvidia.
  MLX_VULKAN_INCLUDE_LINALG=1
    Also run the current linalg CPU-fallback smoke. This is disabled by default
    for first-pass Linux/NVIDIA bring-up because that path is not yet stable.
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


def vendor_looks_nvidia(info):
    text = " ".join(info_to_str(v).lower() for v in info.values())
    return "nvidia" in text or "geforce" in text or "tesla" in text or "quadro" in text


def vendor_matches(info, required_vendor):
    if required_vendor == "amd":
        return vendor_looks_amd(info)
    if required_vendor == "nvidia":
        return vendor_looks_nvidia(info)
    if required_vendor == "intel":
        text = " ".join(info_to_str(v).lower() for v in info.values())
        return "intel" in text or "arc" in text
    raise RuntimeError(f"Unsupported MLX_VULKAN_REQUIRE_VENDOR value: {required_vendor}")


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
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:
        print(f"FAIL: Cannot import mlx.core: {exc}")
        print("Install/build MLX first, for example: pip install -e mlx-src/")
        return 1

    print("========================================")
    print("  STAGE 25: Linux Vulkan Bring-up Smoke")
    print("========================================")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")

    errors = []
    required_vendor = os.environ.get("MLX_VULKAN_REQUIRE_VENDOR", "").strip().lower()
    include_linalg = os.environ.get("MLX_VULKAN_INCLUDE_LINALG", "0") == "1"
    if not required_vendor:
        if os.environ.get("MLX_VULKAN_REQUIRE_AMD", "0") == "1":
            required_vendor = "amd"
        elif os.environ.get("MLX_VULKAN_REQUIRE_NVIDIA", "0") == "1":
            required_vendor = "nvidia"

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
        if required_vendor and not vendor_matches(first, required_vendor):
            raise RuntimeError(f"{required_vendor} GPU required, got: {first}")

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

    def check_autograd():
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)

        def loss_fn(t):
            return mx.sum(t * t)

        loss, grad = mx.value_and_grad(loss_fn)(x)
        mx.eval(loss, grad)
        if not np.isclose(float(loss.tolist()), 14.0, atol=1e-4):
            raise RuntimeError(f"loss mismatch: got {loss.tolist()}")
        got = np.array(grad.tolist(), dtype=np.float32)
        ref = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        if not np.allclose(got, ref, atol=1e-4):
            raise RuntimeError(f"gradient mismatch: got {got.tolist()}")

    def check_optimizers():
        params = {"w": mx.array([1.0, -2.0], dtype=mx.float32)}
        grads = {"w": mx.array([0.5, -0.25], dtype=mx.float32)}

        sgd = optim.SGD(learning_rate=0.1)
        sgd.init(params)
        sgd_params = sgd.apply_gradients(grads, params)
        got_sgd = np.array(sgd_params["w"].tolist(), dtype=np.float32)
        ref_sgd = np.array([0.95, -1.975], dtype=np.float32)
        if not np.allclose(got_sgd, ref_sgd, atol=1e-5):
            raise RuntimeError(f"SGD mismatch: got {got_sgd.tolist()}")

        adam = optim.Adam(learning_rate=0.1)
        adam.init(params)
        adam_params = adam.apply_gradients(grads, params)
        got_adam = np.array(adam_params["w"].tolist(), dtype=np.float32)
        if np.allclose(got_adam, np.array(params["w"].tolist(), dtype=np.float32), atol=1e-6):
            raise RuntimeError("Adam parameters did not change")
        if not np.isfinite(got_adam).all():
            raise RuntimeError(f"Adam produced non-finite values: {got_adam.tolist()}")

    def check_tiny_mlp():
        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 8)
                self.l2 = nn.Linear(8, 1)

            def __call__(self, x):
                return self.l2(nn.relu(self.l1(x)))

        mx.random.seed(7)
        model = TinyMLP()
        optimizer = optim.SGD(learning_rate=0.1)
        optimizer.init(model.parameters())

        x = mx.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=mx.float32,
        )
        y = mx.sum(x, axis=1, keepdims=True)

        def loss_fn(params):
            model.update(params)
            pred = model(x)
            diff = pred - y
            return mx.mean(diff * diff)

        loss_and_grad = mx.value_and_grad(loss_fn)
        params = model.parameters()
        losses = []
        for _ in range(25):
            loss, grads = loss_and_grad(params)
            params = optimizer.apply_gradients(grads, params)
            mx.eval(loss)
            losses.append(float(loss.tolist()))

        if not losses[-1] < losses[0]:
            raise RuntimeError(f"MLP loss did not decrease: {losses[0]} -> {losses[-1]}")
        if min(losses[-5:]) > losses[0]:
            raise RuntimeError(f"MLP convergence too weak: {losses}")

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

        chol = mx.linalg.cholesky(a)
        mx.eval(chol)
        chol_val = np.array(chol.tolist(), dtype=np.float32)
        chol_ref = np.linalg.cholesky(np.array(a.tolist(), dtype=np.float32))
        if not np.allclose(chol_val, chol_ref, atol=1e-4):
            raise RuntimeError(f"cholesky mismatch: got {chol_val.tolist()}")

        eigvals, eigvecs = mx.linalg.eigh(a)
        mx.eval(eigvals, eigvecs)
        eigvals_val = np.array(eigvals.tolist(), dtype=np.float32)
        eigvals_ref, _ = np.linalg.eigh(np.array(a.tolist(), dtype=np.float32))
        if not np.allclose(eigvals_val, eigvals_ref, atol=1e-4):
            raise RuntimeError(f"eigh eigenvalues mismatch: got {eigvals_val.tolist()}")

    check("gpu detect", check_gpu, errors)
    check("core ops", check_core_ops, errors)
    check("autograd", check_autograd, errors)
    check("optimizers", check_optimizers, errors)
    check("tiny mlp", check_tiny_mlp, errors)
    check("python bridge", check_python_bridge, errors)
    check("cpu fallback outputs", check_cpu_fallback_outputs, errors)
    if include_linalg:
        check("linalg fallbacks", check_linalg_fallbacks, errors)
    else:
        print("  SKIP linalg fallbacks (set MLX_VULKAN_INCLUDE_LINALG=1 to enable)")

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
