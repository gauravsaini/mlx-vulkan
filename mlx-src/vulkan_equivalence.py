#!/usr/bin/env python3
"""
vulkan_equivalence.py — Vulkan Backend Numerical Equivalence Test Suite
=======================================================================
Tests that Vulkan GPU results match CPU reference values across all
supported dtypes and ops. Run with:

    PYTHONPATH=python python3 vulkan_equivalence.py [--verbose] [--filter SUBSTR]

Structure
---------
Each test function is registered via @test decorator. It receives a
``ctx`` (TestContext) and should call ctx.check() to assert numerical
equivalence between a Vulkan result and a reference.

Exit codes:  0 = all pass,  1 = failures exist,  2 = no tests matched.
"""

import sys
import os
import time
import traceback
import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# ── Always use in-tree build ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import mlx.core as mx

# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    skipped: bool = False


class TestContext:
    """Passed to each test function. Collects sub-check failures."""

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self._failures: List[str] = []

    # ------------------------------------------------------------------
    def check(
        self,
        label: str,
        gpu_val,          # mlx array computed on GPU
        ref_val,          # numpy array or scalar reference
        atol: float = 1e-3,
        rtol: float = 1e-3,
        dtype_cast=None,  # cast gpu_val before comparison (e.g. mx.float32)
    ):
        """Assert that ``gpu_val`` numerically matches ``ref_val``."""
        try:
            if dtype_cast is not None:
                evaled = np.array(gpu_val.astype(dtype_cast), dtype=np.float32)
            else:
                evaled = np.array(gpu_val, dtype=np.float64)

            ref = np.asarray(ref_val, dtype=np.float64)

            # Handle NaN matching
            if not np.all(
                np.isclose(evaled, ref, atol=atol, rtol=rtol, equal_nan=True)
            ):
                max_err = np.max(np.abs(evaled - ref))
                self._failures.append(
                    f"  {label}: max_err={max_err:.4g} "
                    f"(atol={atol}) | gpu={evaled.flat[:4]} ref={ref.flat[:4]}"
                )
            elif self.verbose:
                print(f"    {GREEN}✓{RESET} {label}")
        except Exception as e:
            self._failures.append(f"  {label}: EXCEPTION — {e}")

    def check_shape(self, label: str, arr, expected_shape: tuple):
        if tuple(arr.shape) != expected_shape:
            self._failures.append(
                f"  {label}: shape {tuple(arr.shape)} != {expected_shape}"
            )
        elif self.verbose:
            print(f"    {GREEN}✓{RESET} {label} shape={expected_shape}")

    def check_dtype(self, label: str, arr, expected_dtype):
        if arr.dtype != expected_dtype:
            self._failures.append(
                f"  {label}: dtype {arr.dtype} != {expected_dtype}"
            )
        elif self.verbose:
            print(f"    {GREEN}✓{RESET} {label} dtype={expected_dtype}")

    @property
    def ok(self) -> bool:
        return len(self._failures) == 0

    @property
    def failures(self) -> List[str]:
        return self._failures


# ─────────────────────────────────────────────────────────────────────────────
# Test registry
# ─────────────────────────────────────────────────────────────────────────────

_TESTS: List[Tuple[str, Callable]] = []

def test(fn: Callable) -> Callable:
    """Decorator: register a test function."""
    _TESTS.append((fn.__name__, fn))
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GPU = mx.gpu
CPU = mx.cpu

def _np_for(dtype) -> np.dtype:
    """Return the closest numpy dtype for an MLX dtype."""
    _MAP = {
        mx.float32:  np.float32,
        mx.float16:  np.float16,
        mx.bfloat16: np.float32,   # numpy has no bf16; cast first
        mx.int32:    np.int32,
        mx.int16:    np.int16,
        mx.int8:     np.int8,
        mx.uint32:   np.uint32,
        mx.uint16:   np.uint16,
        mx.uint8:    np.uint8,
        mx.bool_:    np.bool_,
    }
    return _MAP.get(dtype, np.float32)


def _to_f32_np(arr) -> np.ndarray:
    """Evaluate MLX array and convert to float32 numpy array."""
    mx.eval(arr)
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float32), dtype=np.float32)
    return np.array(arr, dtype=np.float32)


def mk(data, dtype, stream=GPU):
    """Create an MLX array from python list on the given stream."""
    return mx.array(data, dtype=dtype, stream=stream)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  UNARY OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

FLOAT_DTYPES  = [mx.float32, mx.float16, mx.bfloat16]
INT_DTYPES    = [mx.int32, mx.int8]
UINT_DTYPES   = [mx.uint32, mx.uint8]
ALL_DTYPES    = FLOAT_DTYPES + INT_DTYPES + UINT_DTYPES + [mx.bool_]


def _unary_float_data():
    return [0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 1e-3]


@test
def test_unary_neg(ctx):
    data = [1.0, -2.5, 0.0, 3.14]
    ref = np.array([-x for x in data], dtype=np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.negative(a)
        mx.eval(out)
        ctx.check(f"neg/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_abs(ctx):
    data = [1.0, -2.5, 0.0, -3.14]
    ref = np.abs(data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.abs(a)
        mx.eval(out)
        ctx.check(f"abs/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_sqrt(ctx):
    data = [0.0, 1.0, 4.0, 9.0, 0.25]
    ref = np.sqrt(data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.sqrt(a)
        mx.eval(out)
        ctx.check(f"sqrt/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_exp(ctx):
    data = [0.0, 1.0, -1.0, 2.0, -2.0]
    ref = np.exp(data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.exp(a)
        mx.eval(out)
        ctx.check(f"exp/{dt}", out, ref, atol=0.1, dtype_cast=mx.float32)


@test
def test_unary_log(ctx):
    data = [1.0, 2.0, math.e, 10.0, 0.5]
    ref = np.log(data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.log(a)
        mx.eval(out)
        ctx.check(f"log/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_sin_cos(ctx):
    data = [0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        s = mx.sin(a)
        c = mx.cos(a)
        mx.eval(s, c)
        ctx.check(f"sin/{dt}", s, np.sin(data).astype(np.float32), atol=5e-2, dtype_cast=mx.float32)
        ctx.check(f"cos/{dt}", c, np.cos(data).astype(np.float32), atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_tanh(ctx):
    data = [0.0, 1.0, -1.0, 2.0, -2.0]
    ref = np.tanh(data)
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.tanh(a)
        mx.eval(out)
        ctx.check(f"tanh/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_sigmoid(ctx):
    data = [0.0, 1.0, -1.0, 2.0, -4.0]
    ref = 1.0 / (1.0 + np.exp(-np.array(data)))
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.sigmoid(a)
        mx.eval(out)
        ctx.check(f"sigmoid/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_ceil_floor_round(ctx):
    data = [0.1, 0.5, 0.9, 1.5, -0.5, -1.5]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        ctx.check(f"ceil/{dt}",  mx.ceil(a),  np.ceil(data),  atol=1e-1, dtype_cast=mx.float32)
        ctx.check(f"floor/{dt}", mx.floor(a), np.floor(data), atol=1e-1, dtype_cast=mx.float32)


@test
def test_unary_square_rsqrt(ctx):
    data = [1.0, 2.0, 4.0, 0.5]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        sq  = mx.square(a)
        rq  = mx.rsqrt(a)
        mx.eval(sq, rq)
        ctx.check(f"square/{dt}", sq, np.array(data)**2,       atol=0.1, dtype_cast=mx.float32)
        ctx.check(f"rsqrt/{dt}",  rq, 1.0/np.sqrt(data),      atol=0.1, dtype_cast=mx.float32)


@test
def test_unary_erf(ctx):
    import math as _math
    data = [0.0, 0.5, 1.0, -0.5, -1.0]
    ref = [_math.erf(x) for x in data]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out = mx.erf(a)
        mx.eval(out)
        ctx.check(f"erf/{dt}", out, ref, atol=5e-2, dtype_cast=mx.float32)


@test
def test_unary_log2_log10(ctx):
    data = [1.0, 2.0, 8.0, 100.0, 0.5]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        out2  = mx.log2(a)
        out10 = mx.log10(a)
        mx.eval(out2, out10)
        ctx.check(f"log2/{dt}",  out2,  np.log2(data),  atol=5e-2, dtype_cast=mx.float32)
        ctx.check(f"log10/{dt}", out10, np.log10(data), atol=5e-2, dtype_cast=mx.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BINARY OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_binary_add(ctx):
    a_data = [1.0, 2.0, 3.0, -1.0]
    b_data = [4.0, -1.0, 0.5, 2.0]
    ref = np.array(a_data, np.float32) + np.array(b_data, np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        c = a + b
        mx.eval(c)
        ctx.check(f"add/{dt}", c, ref, atol=0.1, dtype_cast=mx.float32)


@test
def test_binary_sub(ctx):
    a_data = [5.0, 2.0, 3.0, -1.0]
    b_data = [1.0, -2.0, 0.5, 2.0]
    ref = np.array(a_data, np.float32) - np.array(b_data, np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        c = a - b
        mx.eval(c)
        ctx.check(f"sub/{dt}", c, ref, atol=0.1, dtype_cast=mx.float32)


@test
def test_binary_mul(ctx):
    a_data = [2.0, 3.0, -1.0, 0.5]
    b_data = [4.0, 0.5, 2.0, -2.0]
    ref = np.array(a_data, np.float32) * np.array(b_data, np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        c = a * b
        mx.eval(c)
        ctx.check(f"mul/{dt}", c, ref, atol=0.1, dtype_cast=mx.float32)


@test
def test_binary_div(ctx):
    a_data = [4.0, 6.0, -2.0, 1.0]
    b_data = [2.0, 3.0, 1.0, 4.0]
    ref = np.array(a_data, np.float32) / np.array(b_data, np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        c = a / b
        mx.eval(c)
        ctx.check(f"div/{dt}", c, ref, atol=0.1, dtype_cast=mx.float32)


@test
def test_binary_max_min(ctx):
    a_data = [1.0, 5.0, -1.0, 3.0]
    b_data = [2.0, 3.0, 0.5, -2.0]
    ref_max = np.maximum(a_data, b_data).astype(np.float32)
    ref_min = np.minimum(a_data, b_data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        ctx.check(f"maximum/{dt}", mx.maximum(a, b), ref_max, atol=0.1, dtype_cast=mx.float32)
        ctx.check(f"minimum/{dt}", mx.minimum(a, b), ref_min, atol=0.1, dtype_cast=mx.float32)


@test
def test_binary_pow(ctx):
    a_data = [2.0, 3.0, 1.5, 0.5]
    b_data = [2.0, 3.0, 2.0, 3.0]
    ref = np.power(a_data, b_data).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        c = mx.power(a, b)
        mx.eval(c)
        ctx.check(f"pow/{dt}", c, ref, atol=0.2, dtype_cast=mx.float32)


@test
def test_binary_compare(ctx):
    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [1.0, 1.0, 4.0, 3.0]
    a32 = np.array(a_data, np.float32)
    b32 = np.array(b_data, np.float32)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        b = mk(b_data, dt)
        mx.eval(a, b)
        ctx.check(f"eq/{dt}",  mx.equal(a, b),         (a32 == b32).astype(np.float32), atol=0.5, dtype_cast=mx.float32)
        ctx.check(f"lt/{dt}",  mx.less(a, b),          (a32 <  b32).astype(np.float32), atol=0.5, dtype_cast=mx.float32)
        ctx.check(f"gt/{dt}",  mx.greater(a, b),       (a32 >  b32).astype(np.float32), atol=0.5, dtype_cast=mx.float32)
        ctx.check(f"le/{dt}",  mx.less_equal(a, b),    (a32 <= b32).astype(np.float32), atol=0.5, dtype_cast=mx.float32)
        ctx.check(f"ge/{dt}",  mx.greater_equal(a, b), (a32 >= b32).astype(np.float32), atol=0.5, dtype_cast=mx.float32)


@test
def test_binary_broadcast(ctx):
    """Test broadcasting with a scalar-like (size-1) operand."""
    a_data = [1.0, 2.0, 3.0, 4.0]
    scalar = 2.0
    ref_add = (np.array(a_data, np.float32) + scalar)
    ref_mul = (np.array(a_data, np.float32) * scalar)
    for dt in FLOAT_DTYPES:
        a = mk(a_data, dt)
        s = mx.array(scalar, dtype=dt, stream=GPU)
        mx.eval(a, s)
        ctx.check(f"broadcast_add/{dt}", a + s, ref_add, atol=0.15, dtype_cast=mx.float32)
        ctx.check(f"broadcast_mul/{dt}", a * s, ref_mul, atol=0.15, dtype_cast=mx.float32)


@test
def test_binary_int_ops(ctx):
    a_data = [4, 10, -3, 7]
    b_data = [2, 3, 2, 3]
    a32 = np.array(a_data, np.int32)
    b32 = np.array(b_data, np.int32)
    a = mk(a_data, mx.int32)
    b = mk(b_data, mx.int32)
    mx.eval(a, b)
    ctx.check("int_add", a + b, a32 + b32, atol=0.5)
    ctx.check("int_sub", a - b, a32 - b32, atol=0.5)
    ctx.check("int_mul", a * b, a32 * b32, atol=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  REDUCTION OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_reduce_sum(ctx):
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ref = float(sum(data))
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        s = mx.sum(a)
        mx.eval(s)
        ctx.check(f"sum1d/{dt}", s, ref, atol=0.5, dtype_cast=mx.float32)


@test
def test_reduce_sum_2d_axis0(ctx):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ref = np.sum(data, axis=0).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mx.array(data, dtype=dt, stream=GPU)
        mx.eval(a)
        s = mx.sum(a, axis=0)
        mx.eval(s)
        ctx.check(f"sum2d_axis0/{dt}", s, ref, atol=0.5, dtype_cast=mx.float32)


@test
def test_reduce_sum_2d_axis1(ctx):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ref = np.sum(data, axis=1).astype(np.float32)
    for dt in FLOAT_DTYPES:
        a = mx.array(data, dtype=dt, stream=GPU)
        mx.eval(a)
        s = mx.sum(a, axis=1)
        mx.eval(s)
        ctx.check(f"sum2d_axis1/{dt}", s, ref, atol=0.5, dtype_cast=mx.float32)


@test
def test_reduce_max_min(ctx):
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        mx_max = mx.max(a)
        mx_min = mx.min(a)
        mx.eval(mx_max, mx_min)
        ctx.check(f"max/{dt}", mx_max, max(data), atol=0.1, dtype_cast=mx.float32)
        ctx.check(f"min/{dt}", mx_min, min(data), atol=0.1, dtype_cast=mx.float32)


@test
def test_reduce_prod(ctx):
    data = [2.0, 0.5, 2.0, 3.0]
    ref = float(np.prod(data))
    for dt in FLOAT_DTYPES:
        a = mk(data, dt)
        mx.eval(a)
        p = mx.prod(a)
        mx.eval(p)
        ctx.check(f"prod/{dt}", p, ref, atol=0.2, dtype_cast=mx.float32)


@test
def test_reduce_sum_int(ctx):
    data = [1, 2, 3, 4, 5, 6]
    ref = sum(data)
    a = mk(data, mx.int32)
    mx.eval(a)
    s = mx.sum(a)
    mx.eval(s)
    ctx.check("sum_int32", s, ref, atol=0.5)


@test
def test_reduce_any_all(ctx):
    trues  = [1, 1, 1, 1]
    falses = [0, 0, 0, 0]
    mixed  = [0, 1, 0, 1]
    for label, data, exp_all, exp_any in [
        ("all_trues",  trues,  True, True),
        ("all_falses", falses, False, False),
        ("mixed",      mixed,  False, True),
    ]:
        a = mx.array(data, dtype=mx.bool_, stream=GPU)
        mx.eval(a)
        ctx.check(f"all_{label}", mx.all(a), float(exp_all), atol=0.5, dtype_cast=mx.float32)
        ctx.check(f"any_{label}", mx.any(a), float(exp_any), atol=0.5, dtype_cast=mx.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  TYPE CASTING & COPY OPS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_cast_float32_to_bf16(ctx):
    data = [1.0, -2.0, 0.5, 100.0, -0.125]
    a32 = mx.array(data, dtype=mx.float32, stream=GPU)
    mx.eval(a32)
    abf = a32.astype(mx.bfloat16)
    mx.eval(abf)
    ctx.check_dtype("cast_f32_to_bf16 dtype", abf, mx.bfloat16)
    ctx.check("cast_f32_to_bf16 values", abf, data, atol=0.1, dtype_cast=mx.float32)


@test
def test_cast_bf16_to_float32(ctx):
    data = [1.0, -2.0, 0.5, 100.0]
    abf = mx.array(data, dtype=mx.bfloat16, stream=GPU)
    mx.eval(abf)
    a32 = abf.astype(mx.float32)
    mx.eval(a32)
    ctx.check_dtype("cast_bf16_to_f32 dtype", a32, mx.float32)
    ctx.check("cast_bf16_to_f32 values", a32, data, atol=0.1)


@test
def test_cast_float16_to_float32(ctx):
    data = [1.0, -2.0, 0.5, 3.14]
    af16 = mx.array(data, dtype=mx.float16, stream=GPU)
    mx.eval(af16)
    a32 = af16.astype(mx.float32)
    mx.eval(a32)
    ctx.check_dtype("cast_f16_to_f32 dtype", a32, mx.float32)
    ctx.check("cast_f16_to_f32 values", a32, data, atol=0.01)


@test
def test_cast_roundtrip(ctx):
    """float32 → bf16 → float32 should be close (BF16 has ~2 decimal places)."""
    data = [1.0, 2.5, -0.125, 100.0, -3.0]
    a = mx.array(data, dtype=mx.float32, stream=GPU)
    mx.eval(a)
    b = a.astype(mx.bfloat16).astype(mx.float32)
    mx.eval(b)
    ctx.check("f32_bf16_f32_roundtrip", b, data, atol=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_matmul_f32(ctx):
    A = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    B = np.random.default_rng(1).standard_normal((16, 8)).astype(np.float32)
    ref = A @ B
    a = mx.array(A, stream=GPU)
    b = mx.array(B, stream=GPU)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    mx.eval(c)
    ctx.check("matmul_f32", c, ref, atol=1e-3, rtol=1e-3)


@test
def test_matmul_f16(ctx):
    A = np.random.default_rng(2).standard_normal((8, 8)).astype(np.float16)
    B = np.random.default_rng(3).standard_normal((8, 8)).astype(np.float16)
    ref = (A.astype(np.float32) @ B.astype(np.float32))
    a = mx.array(A, stream=GPU)
    b = mx.array(B, stream=GPU)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    mx.eval(c)
    ctx.check("matmul_f16", c, ref, atol=0.5, dtype_cast=mx.float32)


@test
def test_matmul_bf16(ctx):
    rng = np.random.default_rng(4)
    A = rng.standard_normal((8, 8)).astype(np.float32)
    B = rng.standard_normal((8, 8)).astype(np.float32)
    ref = A @ B
    a = mx.array(A, dtype=mx.bfloat16, stream=GPU)
    b = mx.array(B, dtype=mx.bfloat16, stream=GPU)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    mx.eval(c)
    ctx.check("matmul_bf16", c, ref, atol=1.0, dtype_cast=mx.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  SHAPE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_reshape(ctx):
    data = list(range(12))
    a = mx.array(data, dtype=mx.float32, stream=GPU)
    mx.eval(a)
    b = mx.reshape(a, (3, 4))
    mx.eval(b)
    ctx.check_shape("reshape_3x4", b, (3, 4))
    ctx.check("reshape_values", b, np.array(data, np.float32).reshape(3, 4))


@test
def test_transpose(ctx):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = mx.array(data, stream=GPU)
    mx.eval(a)
    b = mx.transpose(a)
    mx.eval(b)
    ctx.check_shape("transpose_4x3", b, (4, 3))
    ctx.check("transpose_values", b, data.T)


@test
def test_slice(ctx):
    data = list(range(16))
    a = mx.array(data, dtype=mx.float32, stream=GPU)
    mx.eval(a)
    b = a[4:8]
    mx.eval(b)
    ctx.check("slice_4_8", b, np.array(data[4:8], np.float32))


@test
def test_concatenate(ctx):
    x = np.array([1.0, 2.0, 3.0], np.float32)
    y = np.array([4.0, 5.0, 6.0], np.float32)
    for dt in FLOAT_DTYPES:
        a = mx.array(x, dtype=dt, stream=GPU)
        b = mx.array(y, dtype=dt, stream=GPU)
        mx.eval(a, b)
        c = mx.concatenate([a, b])
        mx.eval(c)
        ctx.check(f"cat/{dt}", c, np.concatenate([x, y]), atol=0.1, dtype_cast=mx.float32)


@test
def test_stack(ctx):
    x = np.array([1.0, 2.0, 3.0], np.float32)
    y = np.array([4.0, 5.0, 6.0], np.float32)
    a = mx.array(x, stream=GPU)
    b = mx.array(y, stream=GPU)
    mx.eval(a, b)
    c = mx.stack([a, b])
    mx.eval(c)
    ctx.check_shape("stack_2x3", c, (2, 3))
    ctx.check("stack_values", c, np.stack([x, y]))


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BF16 SPECIFIC STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_bf16_large_vector(ctx):
    """BF16 add on a larger vector (tests multi-word addressing)."""
    N = 512
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(N).astype(np.float32)
    b_np = rng.standard_normal(N).astype(np.float32)
    ref = a_np + b_np

    a = mx.array(a_np, dtype=mx.bfloat16, stream=GPU)
    b = mx.array(b_np, dtype=mx.bfloat16, stream=GPU)
    mx.eval(a, b)
    c = a + b
    mx.eval(c)
    ctx.check("bf16_large_add", c, ref, atol=0.5, dtype_cast=mx.float32)


@test
def test_bf16_odd_size_vector(ctx):
    """BF16 add on an ODD-length vector (tests boundary word handling)."""
    data_a = [1.0, 2.0, 3.0]   # 3 elements — last uint32 word is partially used
    data_b = [0.5, 1.5, 2.5]
    ref = np.array(data_a) + np.array(data_b)

    a = mk(data_a, mx.bfloat16)
    b = mk(data_b, mx.bfloat16)
    mx.eval(a, b)
    c = a + b
    mx.eval(c)
    ctx.check("bf16_odd_add", c, ref, atol=0.2, dtype_cast=mx.float32)


@test
def test_bf16_all_unary_ops(ctx):
    """Spot-check every unary op in BF16 against a float32 reference."""
    data = [0.25, 0.5, 1.0, 2.0]
    a_bf = mk(data, mx.bfloat16)
    a_f32 = mk(data, mx.float32)
    mx.eval(a_bf, a_f32)

    OPS = [
        ("abs",     mx.abs,     mx.abs),
        ("neg",     mx.negative, mx.negative),
        ("sqrt",    mx.sqrt,    mx.sqrt),
        ("exp",     mx.exp,     mx.exp),
        ("log",     mx.log,     mx.log),
        ("sin",     mx.sin,     mx.sin),
        ("cos",     mx.cos,     mx.cos),
        ("tanh",    mx.tanh,    mx.tanh),
        ("sigmoid", mx.sigmoid, mx.sigmoid),
        ("square",  mx.square,  mx.square),
    ]
    for name, bf16_op, f32_op in OPS:
        r_bf  = bf16_op(a_bf)
        r_ref = f32_op(a_f32)
        mx.eval(r_bf, r_ref)
        ref_np = np.array(r_ref, dtype=np.float32)
        ctx.check(f"bf16_{name}", r_bf, ref_np, atol=0.1, dtype_cast=mx.float32)


@test
def test_bf16_chain_ops(ctx):
    """Chain: exp(abs(a)) via BF16, compare to float32 reference."""
    data = [-1.0, 0.5, 1.5, -0.25]
    a_bf  = mk(data, mx.bfloat16)
    a_f32 = mk(data, mx.float32)
    mx.eval(a_bf, a_f32)

    r_bf  = mx.exp(mx.abs(a_bf))
    r_ref = mx.exp(mx.abs(a_f32))
    mx.eval(r_bf, r_ref)

    ctx.check("bf16_chain_exp_abs", r_bf, np.array(r_ref, np.float32), atol=0.2, dtype_cast=mx.float32)


@test
def test_bf16_sum_reduce(ctx):
    """Sum reduction in BF16; accumulator should stay f32 internally."""
    data = [1.0, 2.0, 3.0, 4.0]
    ref = sum(data)
    a = mk(data, mx.bfloat16)
    mx.eval(a)
    s = mx.sum(a)
    mx.eval(s)
    ctx.check("bf16_sum", s, ref, atol=0.5, dtype_cast=mx.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  CORRECTNESS EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_zeros_ones(ctx):
    for dt in FLOAT_DTYPES:
        z = mx.zeros((4,), dtype=dt, stream=GPU)
        o = mx.ones((4,), dtype=dt, stream=GPU)
        mx.eval(z, o)
        ctx.check(f"zeros/{dt}", z, [0.0]*4, atol=1e-6, dtype_cast=mx.float32)
        ctx.check(f"ones/{dt}",  o, [1.0]*4, atol=1e-6, dtype_cast=mx.float32)


@test
def test_single_element(ctx):
    """Operations on size-1 tensors (scalar-like)."""
    for dt in FLOAT_DTYPES:
        a = mx.array([3.0], dtype=dt, stream=GPU)
        b = mx.array([2.0], dtype=dt, stream=GPU)
        mx.eval(a, b)
        ctx.check(f"add_scalar/{dt}", a + b, [5.0], atol=0.2, dtype_cast=mx.float32)
        ctx.check(f"exp_scalar/{dt}", mx.exp(a), [np.exp(3.0)], atol=0.2, dtype_cast=mx.float32)


@test
def test_nan_propagation(ctx):
    """NaN in inputs should propagate through ops."""
    nan_data = [float("nan"), 1.0, 2.0, 3.0]
    a = mx.array(nan_data, dtype=mx.float32, stream=GPU)
    mx.eval(a)
    r = a + mx.zeros_like(a)
    mx.eval(r)
    r_np = np.array(r, np.float32)
    ctx.check("nan_propagate", r_np[:1], np.array([float("nan")]), atol=0.0)


@test
def test_inf_handling(ctx):
    """Inf values should propagate correctly."""
    data = [float("inf"), float("-inf"), 1.0, 0.0]
    a = mx.array(data, dtype=mx.float32, stream=GPU)
    mx.eval(a)
    r = a * mx.array([1.0, 1.0, 1.0, 1.0], stream=GPU)
    mx.eval(r)
    r_np = np.array(r, np.float32)
    assert np.isinf(r_np[0]) and r_np[0] > 0, "Expected +inf at index 0"
    assert np.isinf(r_np[1]) and r_np[1] < 0, "Expected -inf at index 1"


@test
def test_large_tensor_f32(ctx):
    """Stress test: element-wise multiply on a large (65536) tensor."""
    N = 65536
    rng = np.random.default_rng(7)
    a_np = rng.standard_normal(N).astype(np.float32)
    b_np = rng.standard_normal(N).astype(np.float32)
    ref = a_np * b_np
    a = mx.array(a_np, stream=GPU)
    b = mx.array(b_np, stream=GPU)
    mx.eval(a, b)
    c = a * b
    mx.eval(c)
    ctx.check("large_mul_f32", c, ref, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def run_all(verbose: bool = False, filter_str: Optional[str] = None) -> int:
    """Run all registered tests. Returns exit code."""
    tests = _TESTS
    if filter_str:
        tests = [(n, fn) for n, fn in tests if filter_str in n]
    if not tests:
        print(f"{YELLOW}No tests matched filter '{filter_str}'{RESET}")
        return 2

    results: List[TestResult] = []
    total_duration = 0.0

    print(f"\n{BOLD}{CYAN}{'═'*62}{RESET}")
    print(f"{BOLD}{CYAN}  MLX Vulkan Equivalence Suite  —  {len(tests)} tests{RESET}")
    print(f"{BOLD}{CYAN}{'═'*62}{RESET}\n")

    for name, fn in tests:
        ctx = TestContext(name, verbose=verbose)
        if verbose:
            print(f"\n{BOLD}▶ {name}{RESET}")

        t0 = time.perf_counter()
        exc = None
        try:
            fn(ctx)
        except Exception as e:
            exc = e
            ctx._failures.append(f"  UNCAUGHT: {e}\n  {''.join(traceback.format_exception(type(e), e, e.__traceback__))}")

        dur = (time.perf_counter() - t0) * 1000
        total_duration += dur

        passed = ctx.ok and exc is None
        r = TestResult(name=name, passed=passed, duration_ms=dur,
                       message="\n".join(ctx.failures))

        if passed:
            print(f"  {GREEN}PASS{RESET}  {name:<48} {DIM}{dur:6.1f}ms{RESET}")
        else:
            print(f"  {RED}FAIL{RESET}  {name:<48} {DIM}{dur:6.1f}ms{RESET}")
            if ctx.failures:
                for line in ctx.failures:
                    print(f"       {YELLOW}{line}{RESET}")

        results.append(r)

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)

    print(f"\n{BOLD}{'─'*62}{RESET}")
    print(f"  {BOLD}Total:{RESET}  {len(results)} tests  │  "
          f"{GREEN}{n_pass} passed{RESET}  │  "
          f"{(RED + str(n_fail) + ' failed' + RESET) if n_fail else DIM + '0 failed' + RESET}")
    print(f"  Time:   {total_duration:.0f}ms")
    print(f"{BOLD}{'─'*62}{RESET}\n")

    return 0 if n_fail == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLX Vulkan backend numerical equivalence tests"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each sub-check result")
    parser.add_argument("-f", "--filter", metavar="SUBSTR", default=None,
                        help="Only run tests whose name contains SUBSTR")
    parser.add_argument("--list", action="store_true",
                        help="List all registered test names and exit")
    args = parser.parse_args()

    if args.list:
        print("Registered tests:")
        for name, _ in _TESTS:
            print(f"  {name}")
        sys.exit(0)

    sys.exit(run_all(verbose=args.verbose, filter_str=args.filter))
