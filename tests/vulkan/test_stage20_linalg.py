#!/usr/bin/env python3
"""Stage 20: Linear Algebra CPU fallbacks via Vulkan backend.

Tests that linalg ops (Inverse, Cholesky, QRF, SVD) run successfully on the
default GPU stream via CPU fallback on MoltenVK unified memory.

Notes:
  - mx.eye crashes in current Vulkan backend, so identity comparisons use numpy.
  - L @ L.T in Cholesky test uses numpy matmul because the Vulkan matmul
    has a pre-existing bug with non-contiguous (transposed) inputs. The
    Cholesky factor L itself is verified against the numpy reference instead.
"""
import mlx.core as mx
import numpy as np

results = []

# Test Inverse
try:
    A = mx.array([[2.0, 1.0], [1.0, 3.0]])
    inv_A = mx.linalg.inv(A)
    mx.eval(inv_A)
    I = A @ inv_A
    mx.eval(I)
    I_np = np.array(I.tolist())
    eye_np = np.eye(2)
    ok = bool(np.allclose(I_np, eye_np, atol=1e-4))
    results.append(("Inverse", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("Inverse", f"ERROR: {e}"))

# Test Cholesky -- verify L against numpy reference (avoids Vulkan L@L.T matmul bug)
try:
    A_np = np.array([[4.0, 2.0], [2.0, 3.0]])
    A = mx.array(A_np)
    L = mx.linalg.cholesky(A)
    mx.eval(L)
    L_np = np.array(L.tolist())
    L_ref = np.linalg.cholesky(A_np)
    ok = bool(np.allclose(L_np, L_ref, atol=1e-4))
    results.append(("Cholesky", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("Cholesky", f"ERROR: {e}"))

# Test QRF
try:
    A_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    A = mx.array(A_np)
    Q, R = mx.linalg.qr(A)
    mx.eval(Q, R)
    reconstructed = Q @ R
    mx.eval(reconstructed)
    rec_np = np.array(reconstructed.tolist())
    ok = bool(np.allclose(rec_np, A_np, atol=1e-4))
    results.append(("QRF", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("QRF", f"ERROR: {e}"))

# Test SVD
try:
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = mx.array(A_np)
    U, S, Vt = mx.linalg.svd(A)
    mx.eval(U, S, Vt)
    U_np = np.array(U.tolist())
    S_np = np.array(S.tolist())
    Vt_np = np.array(Vt.tolist())
    reconstructed_np = U_np @ np.diag(S_np) @ Vt_np
    ok = bool(np.allclose(reconstructed_np, A_np, atol=1e-4))
    results.append(("SVD", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("SVD", f"ERROR: {e}"))

passed = sum(1 for _, r in results if r == "PASS")
total = len(results)
print(f"\nResults: {passed}/{total}")
for name, r in results:
    print(f"  {name}: {r}")
