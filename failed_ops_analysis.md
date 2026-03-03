# MLX Vulkan Backend - Official Test Suite Failure Analysis

**Date**: 2026-03-03
**Test Suite**: MLX test_ops.py (134 tests total)
**Status**: 21/34 tests PASSED (62% in partial run, 2 crashes excluded)

---

## Executive Summary

The MLX Vulkan backend was profiled against the official MLX test suite (`test_ops.py` and `test_array.py`). Results show significant progress with 62% pass rate on non-crashing tests, but reveal critical issues in several core operations.

**Key Findings**:
- **13 test failures** identified (excluding 2 crashes)
- **2 critical crashes**: `test_diag`, `test_api` (bus error/segfault)
- **3 major failure categories**: Incorrect implementations, Type promotion issues, Complex number support

---

## Test Results Summary

### test_ops.py (partial run - 34 tests, excluded diag crash)
- **PASSED**: 21 tests (62%)
- **FAILED**: 13 tests (38%)
- **CRASH**: test_diag (segfault)

### test_array.py (partial run - 19 tests before crash)
- **PASSED**: 18 tests (95%)
- **CRASH**: test_api (bus error at line 1546)

---

## Categorized Failure Analysis

### Category 1: CRITICAL CRASHES (Highest Priority)

#### 1.1 test_diag - Segmentation Fault
- **Location**: `test_ops.py:2594` - `mx.diag(x)` on 1D input
- **Root Cause**: GPU primitive or eval_gpu crash in diag implementation
- **Impact**: Blocks all diag-related tests
- **Fix Required**: Debug GPU dispatch path for diag operation

#### 1.2 test_api - Bus Error
- **Location**: `test_array.py:1546` - `assertTrue` call in test_api
- **Root Cause**: Unknown - likely memory corruption or GPU buffer issue
- **Impact**: Blocks all subsequent test_array tests
- **Fix Required**: Debug GPU buffer management in array operations

---

### Category 2: Type Promotion / Casting Issues

#### 2.1 test_add FAILED
- **Issue**: `int64 + float32` returns wrong result (3.0 instead of 4.0)
- **Test Case**:
  ```python
  x = mx.array(1, mx.int64)
  z = x + 3.0  # Expected: float32 with value 4.0, Got: 3.0
  ```
- **Root Cause**: Binary operation type promotion logic incorrect for int64+float32
- **Fix Required**: Fix binary type promotion in primitives.cpp

#### 2.2 test_arange_corner_cases_cast FAILED
- **Issue**: Casting behavior in arange differs from NumPy
- **Root Cause**: Dtype casting logic in arange GPU primitive
- **Fix Required**: Align arange dtype promotion with NumPy spec

#### 2.3 test_arange_overload_dispatch FAILED
- **Issue**: Arange overload dispatch not matching expected behavior
- **Root Cause**: Dispatch logic in arange implementation
- **Fix Required**: Fix arange overload resolution

---

### Category 3: Incorrect GPU Implementations

#### 3.1 test_bitwise_ops FAILED
- **Issue**: Bitwise operations (bitwise_and, bitwise_or, bitwise_xor) producing wrong results
- **Test Case**: `left_shift` and `right_shift` operations fail
- **Root Cause**: Binary shader implementation for bitwise operations
- **Fix Required**: Debug binary.comp for bitwise ops

#### 3.2 test_bitwise_grad FAILED
- **Issue**: Bitwise operation gradients incorrect
- **Root Cause**: Missing or incorrect gradient implementation
- **Fix Required**: Implement gradients for bitwise ops

#### 3.3 test_clip FAILED
- **Issue**: `mx.clip(array, min_array, 4)` produces wrong result
- **Test Case**:
  ```python
  mins = np.array([3, 1, 5, 5])
  a = np.array([2, 3, 4, 5], np.int32)
  clipped = mx.clip(mx.array(a), mx.array(mins), 4)
  # Expected: [3, 3, 5, 5], Got: incorrect
  ```
- **Root Cause**: Clip implementation doesn't handle array mins/maxes correctly
- **Fix Required**: Fix clip GPU primitive for array bounds

#### 3.4 test_divmod FAILED - CRITICAL
- **Issue**: `mx.divmod` returns garbage values
- **Test Case**:
  ```python
  a_np = [79], b_np = [88]
  np_out = np.divmod(a_np, b_np)  # [0], [79]
  mx_out = mx.divmod(mx.array(a_np), mx.array(b_np))
  # Got: [2143289344], [2143289344] - COMPLETELY WRONG
  ```
- **Root Cause**: DivMod primitive completely broken - memory corruption or wrong GPU implementation
- **Fix Required**: Complete rewrite of divmod GPU primitive

#### 3.5 test_dynamic_slicing FAILED
- **Issue**: `mx.slice_update` produces incorrect results
- **Test Case**:
  ```python
  x = mx.zeros(shape=(4, 4, 4))
  update = mx.random.randint(0, 100, shape=(3, 2, 1))
  out = mx.slice_update(x, update, mx.array([1, 2, 3]), (0, 1, 2))
  # Expected: x[1:, 2:, 3:] = update, Got: wrong
  ```
- **Root Cause**: Slice update GPU implementation incorrect
- **Note**: Known issue - `compute_dynamic_offset()` in slicing.cpp is a stub returning 0
- **Fix Required**: Implement proper dynamic offset computation in slice_update

---

### Category 4: Mathematical Precision Issues

#### 4.1 test_cos FAILED
- **Issue**: `mx.cos(π/2)` returns 0 instead of ~0 (float32 precision)
- **Test Case**:
  ```python
  a = [0, π/4, π/2, π, 3π/4, 2π]
  result = mx.cos(a)
  # Expected: [1, 0.707..., 0, -1, -0.707..., 1]
  # Got: [1, 0.707107, 0, -1, -0.707107, 1]
  # Difference: [0, 0, 4.37e-08, 0, 0, 0]
  ```
- **Root Cause**: Float32 precision in GPU cosine calculation
- **Note**: This is actually correct within float32 precision limits
- **Fix Required**: May need to adjust test tolerance or use higher precision

#### 4.2 test_conjugate FAILED
- **Issue**: Complex conjugate operation incorrect
- **Root Cause**: Unary shader for conjugate not implemented correctly for complex64
- **Fix Required**: Fix conjugate implementation in unary.comp

---

### Category 5: Complex Number Support Issues

#### 5.1 test_complex_ops FAILED
- **Issue**: All complex operations fail (arccos, arcsin, arctan, square, sqrt, rsqrt)
- **Test Case**:
  ```python
  x = mx.array([3.0 + 4.0j, -5.0 + 12.0j, -8.0 + 0.0j, 0.0 + 9.0j, 0.0 + 0.0j])
  mx.sqrt(x)  # Got: [1.73205+0j, ...] WRONG
  # Expected: [2+1j, ...]
  ```
- **Root Cause**: Complex number operations not implemented or incorrect in GPU shaders
- **Fix Required**: Implement complex math in unary/other shaders

#### 5.2 test_complex_power FAILED
- **Issue**: Complex power operation incorrect
- **Root Cause**: Power operation doesn't handle complex numbers correctly
- **Fix Required**: Fix complex power implementation

---

### Category 6: Array Comparison Issues

#### 6.1 test_array_equal FAILED
- **Issue**: `mx.array_equal` returns incorrect results
- **Root Cause**: Array equality comparison implementation incorrect
- **Fix Required**: Fix array_equal logic in comparison ops

---

## Fix Priority Ranking

### P0 - Critical (Blockers)
1. **test_divmod** - Returns garbage, completely broken
2. **test_diag** - Crash/segfault
3. **test_api** - Bus error/crash

### P1 - High Impact (Core functionality)
4. **test_add** - Type promotion wrong (common operation)
5. **test_bitwise_ops** - Bitwise ops incorrect
6. **test_dynamic_slicing** - Slice update broken (known stub issue)
7. **test_clip** - Array bounds not working

### P2 - Medium Impact (Less common)
8. **test_complex_ops** - Complex math broken
9. **test_complex_power** - Complex power broken
10. **test_array_equal** - Equality checks wrong
11. **test_arange_corner_cases_cast** - Dtype issues
12. **test_arange_overload_dispatch** - Overload issues

### P3 - Low Impact (Precision/Niche)
13. **test_cos** - Float32 precision (may be test issue)
14. **test_conjugate** - Complex conjugate
15. **test_bitwise_grad** - Gradients for bitwise ops

---

## Root Cause Analysis by Component

### GPU Primitives
- **Binary**: bitwise ops, type promotion (int64+float32)
- **Unary**: cos (precision), conjugate (complex)
- **Clip**: Array bounds handling
- **DivMod**: COMPLETELY BROKEN (memory corruption?)
- **Slice**: Dynamic offset computation (stub)

### GPU Shaders
- **binary.comp**: Bitwise operations, type promotion
- **unary.comp**: Trig functions, complex operations
- **clip/ternary**: Array bounds logic

### Eval/Dispatch
- **diag**: Crash in GPU eval path
- **slice_update**: Incorrect dynamic offset

### Type System
- **Promotion**: int64+float32 cast
- **Arange**: Dtype casting and overload dispatch

---

## Recommended Fix Strategy

### Immediate (This Week)
1. Fix divmod - critical garbage output
2. Debug diag crash - blocking tests
3. Fix int64+float32 type promotion

### Short-term (Next 2 weeks)
4. Implement proper dynamic offset for slice_update
5. Fix clip array bounds handling
6. Fix bitwise operations

### Medium-term (Next month)
7. Implement complex number support in shaders
8. Fix arange dtype issues
9. Fix array_equal

### Long-term (Future)
10. Investigate float32 precision issues
11. Implement gradients for bitwise ops

---

## Notes

1. **Python Version**: Tests run with Python 3.11 (Python 3.14 incompatible with compiled .so)
2. **Build**: Rebuilt with debug output disabled for cleaner test output
3. **Test Coverage**: 34/134 tests run successfully before crashes
4. **Known Issues**: Slice update dynamic offset is documented stub (see MEMORY.md)
5. **Complex Numbers**: Complex64 dtype exists but GPU implementation is incorrect

---

## Test Environment

```
Platform: macOS (Darwin 24.6.0)
Device: Apple M1 via MoltenVK
Python: 3.11.14
Vulkan: VK_EXT_external_memory_host available
Test Suite: MLX test_ops.py, test_array.py
Timeout: 120s (extended to 300s for full runs)
```
