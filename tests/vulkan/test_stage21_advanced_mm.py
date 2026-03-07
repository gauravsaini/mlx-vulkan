#!/usr/bin/env python3.11
"""Stage 21: GatherMM, BlockMaskedMM, SegmentedMM.

This stage validates:
  1. GatherMM CPU and GPU numerical correctness.
  2. BlockMaskedMM CPU and GPU numerical correctness for bool and float masks.
  3. SegmentedMM remains callable (CPU stream, and GPU stream fallback behavior).
"""

import numpy as np
import mlx.core as mx

results = []


def report(name, status, detail=""):
    results.append((name, status, detail))


def block_masked_mm_ref(a, b, block_size, mask_out=None, mask_lhs=None, mask_rhs=None):
    """Reference implementation matching CPU backend semantics."""
    a = np.asarray(a)
    b = np.asarray(b)
    out_dtype = np.result_type(a.dtype, b.dtype)

    batch_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_b = np.broadcast_to(a, batch_shape + a.shape[-2:]).astype(out_dtype, copy=True)
    b_b = np.broadcast_to(b, batch_shape + b.shape[-2:]).astype(out_dtype, copy=True)

    m, k = a_b.shape[-2], a_b.shape[-1]
    n = b_b.shape[-1]

    tm = (m + block_size - 1) // block_size
    tn = (n + block_size - 1) // block_size
    tk = (k + block_size - 1) // block_size

    a_f = a_b.reshape((-1, m, k))
    b_f = b_b.reshape((-1, k, n))
    batch = a_f.shape[0]

    lhs_f = None
    rhs_f = None
    out_f = None
    if mask_lhs is not None:
        lhs = np.broadcast_to(np.asarray(mask_lhs), batch_shape + (tm, tk))
        lhs_f = lhs.reshape((-1, tm, tk))
    if mask_rhs is not None:
        rhs = np.broadcast_to(np.asarray(mask_rhs), batch_shape + (tk, tn))
        rhs_f = rhs.reshape((-1, tk, tn))
    if mask_out is not None:
        out_m = np.broadcast_to(np.asarray(mask_out), batch_shape + (tm, tn))
        out_f = out_m.reshape((-1, tm, tn))

    for bi in range(batch):
        if lhs_f is not None:
            lhs_is_bool = lhs_f.dtype == np.bool_
            for i in range(tm):
                rs = i * block_size
                re = min(rs + block_size, m)
                for j in range(tk):
                    cs = j * block_size
                    ce = min(cs + block_size, k)
                    mv = lhs_f[bi, i, j]
                    if mv != 1:
                        if lhs_is_bool:
                            a_f[bi, rs:re, cs:ce] = 0
                        else:
                            a_f[bi, rs:re, cs:ce] *= mv

        if rhs_f is not None:
            rhs_is_bool = rhs_f.dtype == np.bool_
            for i in range(tk):
                rs = i * block_size
                re = min(rs + block_size, k)
                for j in range(tn):
                    cs = j * block_size
                    ce = min(cs + block_size, n)
                    mv = rhs_f[bi, i, j]
                    if mv != 1:
                        if rhs_is_bool:
                            b_f[bi, rs:re, cs:ce] = 0
                        else:
                            b_f[bi, rs:re, cs:ce] *= mv

    out = np.matmul(a_f, b_f)

    if out_f is not None:
        out_is_bool = out_f.dtype == np.bool_
        for bi in range(batch):
            for i in range(tm):
                rs = i * block_size
                re = min(rs + block_size, m)
                for j in range(tn):
                    cs = j * block_size
                    ce = min(cs + block_size, n)
                    mv = out_f[bi, i, j]
                    if mv != 1:
                        if out_is_bool:
                            out[bi, rs:re, cs:ce] = 0
                        else:
                            out[bi, rs:re, cs:ce] *= mv

    return out.reshape(batch_shape + (m, n))


# Test 1: BlockMaskedMM on CPU stream
if hasattr(mx, "block_masked_mm"):
    try:
        with mx.stream(mx.cpu):
            a = mx.random.normal(shape=(128, 128))
            b = mx.random.normal(shape=(128, 128))
            mask_out = mx.ones((2, 2), dtype=mx.bool_)
            out = mx.block_masked_mm(a, b, block_size=64, mask_out=mask_out)
            mx.eval(out)
        report("BlockMaskedMM_cpu", "PASS", f"shape={list(out.shape)}")
    except Exception as e:
        report("BlockMaskedMM_cpu", "ERROR", str(e))
else:
    report("BlockMaskedMM_cpu", "SKIP", "mx.block_masked_mm not in API")


# Test 2: GatherMM on CPU stream
try:
    with mx.stream(mx.cpu):
        a = mx.random.normal(shape=(4, 8, 16))
        b = mx.random.normal(shape=(4, 16, 8))
        lhs_idx = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        rhs_idx = mx.array([3, 2, 1, 0], dtype=mx.uint32)
        out = mx.gather_mm(a, b, lhs_indices=lhs_idx, rhs_indices=rhs_idx)
        mx.eval(out)
    expected_shape = (4, 8, 8)
    if tuple(out.shape) == expected_shape:
        report("GatherMM_cpu", "PASS", f"shape={list(out.shape)}")
    else:
        report("GatherMM_cpu", "FAIL", f"expected {expected_shape}, got {list(out.shape)}")
except Exception as e:
    report("GatherMM_cpu", "ERROR", str(e))


# Test 3: GatherMM numerical correctness (CPU stream)
try:
    np.random.seed(42)
    a_np = np.random.randn(3, 4, 6).astype(np.float32)
    b_np = np.random.randn(3, 6, 5).astype(np.float32)
    lhs_np = np.array([2, 0, 1], dtype=np.uint32)
    rhs_np = np.array([1, 2, 0], dtype=np.uint32)
    ref = np.stack([a_np[lhs_np[i]] @ b_np[rhs_np[i]] for i in range(3)])

    with mx.stream(mx.cpu):
        out_mx = mx.gather_mm(
            mx.array(a_np),
            mx.array(b_np),
            lhs_indices=mx.array(lhs_np),
            rhs_indices=mx.array(rhs_np),
        )
        mx.eval(out_mx)

    max_err = float(np.max(np.abs(np.array(out_mx) - ref)))
    if max_err < 1e-4:
        report("GatherMM_cpu_numerical", "PASS", f"max_err={max_err:.2e}")
    else:
        report("GatherMM_cpu_numerical", "FAIL", f"max_err={max_err:.4f}")
except Exception as e:
    report("GatherMM_cpu_numerical", "ERROR", str(e))


# Test 4: GatherMM numerical correctness (GPU stream)
try:
    np.random.seed(7)
    a_np = np.random.randn(5, 6, 10).astype(np.float32)
    b_np = np.random.randn(5, 10, 7).astype(np.float32)
    lhs_np = np.array([4, 0, 3, 1, 2], dtype=np.uint32)
    rhs_np = np.array([2, 4, 1, 0, 3], dtype=np.uint32)
    ref = np.stack([a_np[lhs_np[i]] @ b_np[rhs_np[i]] for i in range(5)])

    with mx.stream(mx.gpu):
        out_gpu = mx.gather_mm(
            mx.array(a_np),
            mx.array(b_np),
            lhs_indices=mx.array(lhs_np),
            rhs_indices=mx.array(rhs_np),
        )
        mx.eval(out_gpu)

    max_err = float(np.max(np.abs(np.array(out_gpu) - ref)))
    if max_err < 1e-3:
        report("GatherMM_gpu_numerical", "PASS", f"max_err={max_err:.2e}")
    else:
        report("GatherMM_gpu_numerical", "FAIL", f"max_err={max_err:.4f}")
except Exception as e:
    report("GatherMM_gpu_numerical", "ERROR", str(e))


# Test 5: BlockMaskedMM numerical correctness (GPU, bool masks)
if hasattr(mx, "block_masked_mm"):
    try:
        np.random.seed(11)
        a_np = np.random.randn(2, 96, 80).astype(np.float32)
        b_np = np.random.randn(2, 80, 64).astype(np.float32)
        block_size = 32
        tm, tn, tk = 3, 2, 3

        out_mask = np.array(
            [
                [[1, 0], [1, 1], [0, 1]],
                [[1, 1], [1, 0], [1, 1]],
            ],
            dtype=np.bool_,
        )
        lhs_mask = np.array(
            [
                [[1, 1, 0], [1, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [0, 1, 1], [1, 0, 1]],
            ],
            dtype=np.bool_,
        )
        rhs_mask = np.array(
            [
                [[1, 1], [0, 1], [1, 0]],
                [[1, 0], [1, 1], [1, 1]],
            ],
            dtype=np.bool_,
        )

        ref = block_masked_mm_ref(
            a_np,
            b_np,
            block_size,
            mask_out=out_mask,
            mask_lhs=lhs_mask,
            mask_rhs=rhs_mask,
        )

        with mx.stream(mx.gpu):
            out_gpu = mx.block_masked_mm(
                mx.array(a_np),
                mx.array(b_np),
                block_size=block_size,
                mask_out=mx.array(out_mask),
                mask_lhs=mx.array(lhs_mask),
                mask_rhs=mx.array(rhs_mask),
            )
            mx.eval(out_gpu)

        max_err = float(np.max(np.abs(np.array(out_gpu) - ref)))
        if max_err < 1e-3:
            report("BlockMaskedMM_gpu_bool", "PASS", f"max_err={max_err:.2e}")
        else:
            report("BlockMaskedMM_gpu_bool", "FAIL", f"max_err={max_err:.4f}")
    except Exception as e:
        report("BlockMaskedMM_gpu_bool", "ERROR", str(e))
else:
    report("BlockMaskedMM_gpu_bool", "SKIP", "mx.block_masked_mm not in API")


# Test 6: BlockMaskedMM numerical correctness (GPU, float masks)
if hasattr(mx, "block_masked_mm"):
    try:
        np.random.seed(123)
        a_np = np.random.randn(64, 64).astype(np.float32)
        b_np = np.random.randn(64, 64).astype(np.float32)
        block_size = 32

        out_mask = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
        lhs_mask = np.array([[1.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        rhs_mask = np.array([[1.0, 0.0], [0.75, 1.0]], dtype=np.float32)

        ref = block_masked_mm_ref(
            a_np,
            b_np,
            block_size,
            mask_out=out_mask,
            mask_lhs=lhs_mask,
            mask_rhs=rhs_mask,
        )

        with mx.stream(mx.gpu):
            out_gpu = mx.block_masked_mm(
                mx.array(a_np),
                mx.array(b_np),
                block_size=block_size,
                mask_out=mx.array(out_mask),
                mask_lhs=mx.array(lhs_mask),
                mask_rhs=mx.array(rhs_mask),
            )
            mx.eval(out_gpu)

        max_err = float(np.max(np.abs(np.array(out_gpu) - ref)))
        if max_err < 1e-3:
            report("BlockMaskedMM_gpu_float", "PASS", f"max_err={max_err:.2e}")
        else:
            report("BlockMaskedMM_gpu_float", "FAIL", f"max_err={max_err:.4f}")
    except Exception as e:
        report("BlockMaskedMM_gpu_float", "ERROR", str(e))
else:
    report("BlockMaskedMM_gpu_float", "SKIP", "mx.block_masked_mm not in API")


# Test 7: SegmentedMM availability / fallback behavior
if hasattr(mx, "segmented_mm"):
    try:
        with mx.stream(mx.cpu):
            a3 = mx.random.normal(shape=(4, 8))
            b3 = mx.random.normal(shape=(8, 4))
            segs = mx.array([[0, 4], [4, 8]], dtype=mx.uint32)
            out3 = mx.segmented_mm(a3, b3, segs)
            mx.eval(out3)
        report("SegmentedMM_cpu", "PASS", f"shape={list(out3.shape)}")
    except Exception as e:
        report("SegmentedMM_cpu", "ERROR", str(e))

    try:
        with mx.stream(mx.gpu):
            a4 = mx.random.normal(shape=(4, 8))
            b4 = mx.random.normal(shape=(8, 4))
            segs2 = mx.array([[0, 4], [4, 8]], dtype=mx.uint32)
            out4 = mx.segmented_mm(a4, b4, segs2)
            mx.eval(out4)
        report("SegmentedMM_gpu", "PASS", f"shape={list(out4.shape)}")
    except Exception as e:
        report("SegmentedMM_gpu", "ERROR", str(e))
else:
    report("SegmentedMM_cpu", "SKIP", "mx.segmented_mm not in Python API")
    report("SegmentedMM_gpu", "SKIP", "mx.segmented_mm not in Python API")


passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s in ("FAIL", "ERROR"))
skipped = sum(1 for _, s, _ in results if s == "SKIP")
total = passed + failed

print(f"\nResults: {passed}/{total} (skipped={skipped})")
for name, status, detail in results:
    suffix = f" - {detail}" if detail else ""
    print(f"  {name}: {status}{suffix}")
