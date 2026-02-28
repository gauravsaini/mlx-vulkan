# MLX Vulkan Backend — Change Timeline

---

## UPDATED ON : 2026-02-28

### fix (2026-02-28) — Binary broadcast stride-based indexing + pipeline cache v4

1. **Binary broadcast hang fixed — stride-based ND indexing in `binary.comp`**:
   - Root cause: binary shader used flat `idx` for both inputs. Broadcast arrays (e.g. `(2,1)` broadcast
     to `(2,3)`) have `data_size=2` but shader indexed `b[0..5]` → OOB GPU hang/deadlock.
   - Multiple approaches tried: modulo (`idx % data_size`) — only worked for scalar/row broadcast, not
     column; `expand_broadcast` via `copy_gpu_inplace` — copy completed but subsequent binary shader hung
     (MoltenVK command-buffer-level GPU conflict); stale `.spv` cache contributed to debugging difficulty.
   - Final fix: **stride-based ND broadcast indexing in the shader**. Push constants expanded from 28→80 bytes
     (well within 128-byte limit): `ndim`, `out_shape[4]`, `a_strides[4]`, `b_strides[4]`. Shader decomposes
     flat output index into ND coordinates via `out_shape`, then dots with per-input strides. stride=0 for
     broadcast dimensions. Handles scalar, row, column, and arbitrary ND broadcast patterns. No pre-copy needed.
   - `compute_broadcast_strides()` helper added to `primitives.cpp`: computes zero-stride for dims where
     `in.shape(i)==1`, uses original stride otherwise. Left-pads with 0 for fewer input dims.

2. **Pipeline cache bumped v3→v4** (`device.cpp`): binary.comp push constant layout changed (28→80 bytes).

3. **BINARY_GPU macro simplified**: removed `NEEDS_EXPAND` / `contiguous_copy_gpu()` pre-copy logic.
   All broadcast handling is now in-shader. Direct dispatch from eval_gpu to dispatch_binary.

4. **Tests** (before → after):
   - `(2,3)+(2,1)` column broadcast: ❌ GPU hang → ✅ `[[11,12,13],[24,25,26]]`
   - `(2,3)+(1,3)` row broadcast: ✅ (already worked with modulo, still works)
   - `(2,3,4)+(2,1,1)` 3D broadcast: ❌ hang → ✅ correct
   - `logsumexp` (uses broadcast internally): ✅ correct
   - All 25 existing stage tests (stages 13–18): ✅ zero regressions

5. **Files changed**: `primitives.cpp`, `binary.comp`, `device.cpp`

6. **Decision rationale — why stride-based indexing**:
   - **Modulo rejected**: `idx % data_size` only works when the broadcast dimension is the last axis.
     For column broadcast `(2,1)→(2,3)`, modulo gives `0,1,0,1,0,1` but correct is `0,0,0,1,1,1`.
     The flat-index-to-element mapping differs per broadcast axis position, so a 1D modulo is insufficient.
   - **expand_broadcast rejected**: dispatching a copy shader + binary shader within a single `eval_gpu`
     call deadlocked on MoltenVK. The copy completed (confirmed via `fprintf` traces), but the GPU hung
     on the subsequent binary dispatch. This appears to be a MoltenVK-specific command buffer contention
     issue when two compute dispatches share the same encoder within a single primitive eval.
   - **Stride-based chosen**: single shader dispatch, no pre-copy, push constants stay within 128-byte
     Vulkan limit (80 bytes for 20 fields). Mirrors `copy.comp`'s `strided_offset_src` pattern.
     `compute_broadcast_strides()` computes zero-stride for broadcast dims, matching MLX's internal
     stride representation. Handles 0D through 4D broadcasts uniformly.
   - **Gotcha**: `setup.py build_ext` does NOT recompile `.comp→.spv` shaders automatically.
     Manual `glslc` is required. Stale `binary.spv` from Feb 27 caused hours of debugging.

---

## UPDATED ON : 2026-03-01

### fix (2026-03-01) — Reduce any-axis + logcumsumexp GPU support

1. **Reduce any-axis fixed (was axis=0 wrong)**:
   - Root cause: `reduce.comp` used `in_data[out_idx * reduce_size + j]` — only correct for last-axis reduce
   - Fix: added `inner` + `outer_stride` to push constant (24→32 bytes)
   - Shader now uses: `in_idx = (out_idx / inner) * outer_stride + j * inner + (out_idx % inner)`
   - Handles any single/multi-axis reduction from a row-contiguous input; `inner=1` = last-axis (unchanged behaviour)

2. **logcumsumexp (LogAddExp scan) GPU support added**:
   - `scan.comp`: added `SCAN_LOGADDEXP` (op=4) with numerically stable combine: `max(a,b) + log(1+exp(min-max))`
   - `primitives.cpp`: removed throw for `Scan::ReduceType::LogAddExp`, mapped to op=4

3. **Pipeline cache bumped v2 → v3**:
   - `reduce.comp` push constant layout changed; old binary cache invalid
   - `kPipelineCacheVersion = 3` in `device.cpp` forces new cache path

4. **Tests (before → after)**:
   - Stage 10 Reduce: 8/11 → **11/11** ✅ (sum/max/min axis=0 were wrong, now correct)
   - Stage 15 Scan: 5/5 ✅ (unchanged — logcumsumexp also now works in test_api)

5. **Files changed**: `kernels/reduce.comp`, `kernels/scan.comp`, `primitives.cpp`, `device.cpp`

---

## UPDATED ON : 2026-02-28

### feat (2026-02-28) — Gather ND, Scan GPU dispatch, .gitignore

1. **Gather generalized to any axis/rank** (`mx.take(src, idx, axis=N)`):
   - Added `INDEX_GATHER_GEN` (op=3) to `indexing.comp` — general ND gather
   - Formula: `j = tid/slice_total`, `outer_off = within/inner`, `src_pos = outer_off * src_outer_stride + idx[j] * inner + inner_off`
   - Works for any source rank and any gather axis, not just 1D/axis=0

2. **GatherAxis / ScatterAxis unified to 44-byte push constant**:
   - Shared `IndexPushConst` (11 fields = 44 bytes) + `indexing_dispatch()` helper
   - Eliminated per-function boilerplate and push-constant size mismatches
   - Fixed ScatterAxis negative-index wrap: was using `idx_size`, now uses `src_ax_size`

3. **MoltenVK stale pipeline cache crash fixed**:
   - Root cause: changing push constant layout 36→44 bytes corrupted on-disk binary cache
   - MoltenVK SIGKILL'd the process at dlopen time (not at pipeline creation)
   - Fix: versioned cache path `mlx_vulkan_pipeline_cache_v2.bin` in `device.cpp`

4. **Scan GPU dispatch implemented** (`scan.comp` — two-level Hillis-Steele):
   - Phase 1: serial inclusive scan within each thread's chunk (chunk_size = ⌈scan_size/256⌉)
   - Phase 2: parallel exclusive prefix scan across 256 chunk totals (Hillis-Steele)
   - Phase 3: propagate chunk prefix; inclusive/exclusive conversion at writeback
   - Supports scan_size ≤ 1024; Sum/Prod/Min/Max; inclusive + exclusive; reverse

5. **Tests (before → after)**:
   - Stage 13 Indexing: 4/7 → **7/7** ✅ (`test_take_axis` now passes)
   - Stage 15 Scan: 0/5 → **5/5** ✅ (was unconditional throw)
   - Stages 14, 16, 17, 18: unchanged (6/6, 8/8, 3/3, 3/3) ✅

6. **Files changed**: `kernels/indexing.comp`, `kernels/scan.comp`, `primitives.cpp`, `device.cpp`, `.gitignore`, `PLAN.md`, `CLAUDE.md`

---

## UPDATED ON : 2026-02-26

### feat (2026-02-26) — AddMM, Convolution, RandomBits GPU dispatch

1. **AddMM (alpha·A@B + beta·C)**: CPU fallback via `add_mm` (BLAS); GPU dispatches matmul + scalar-multiply binary shaders
2. **Convolution**: CPU fallback for full convolution; GPU path dispatches `conv.comp` shader for 1D/2D cases
3. **RandomBits**: New `rbits.comp` shader implementing Philox RNG (4-round counter-based); generates random uint32 bits on GPU; supports different seeds and key splitting
4. **Tests**: `test_stage17_addmm_conv_rbits.py` added — covers AddMM dimensions, conv correctness, random shape/seed behaviour
5. **Files changed**: `kernels/rbits.comp`, `primitives.cpp`, `CMakeLists.txt`, `PLAN.md`, `tests/vulkan/test_stage17_addmm_conv_rbits.py`

---

## UPDATED ON : 2026-02-25

### feat (2026-02-25) — MLX Vulkan backend foundation

1. **Full Vulkan backend scaffold**: Device init (VkInstance → physical device → logical device), VMA allocator, per-stream CommandEncoder, pipeline cache (load/save binary), descriptor pool
2. **20+ GLSL compute shaders**: `unary.comp`, `binary.comp`, `binary_two.comp`, `copy.comp`, `reduce.comp`, `arg_reduce.comp`, `matmul.comp`, `softmax.comp`, `arange.comp`, `indexing.comp`, `scan.comp`, `sort.comp`, `normalization.comp`, `rope.comp`, `logsumexp.comp`, `random.comp`, `conv.comp`, `quantized.comp`
3. **GPU dispatch implemented** for: Unary (28 ops), Binary (18 ops), Ternary/Select, Arange, Reduce (Sum/Max/Min/Prod/And/Or), ArgReduce, Matmul, Softmax, LogSumExp, Copy, Gather (1D), Sort
4. **MoltenVK / Apple M1**: Zero-copy path via `VK_EXT_external_memory_host`; pipeline cache persisted to `~/.cache/`
5. **Test suite (stages 1–16)**: Device, build, shader, allocator, unified-mem, workgroup, CPU smoke, copy, unary, binary, reduce, matmul, NN ops, indexing, sort, scan, NN-extended stages
6. **Files changed**: entire `mlx-src/mlx/backend/vulkan/`, `tests/vulkan/` (18 test files), `PLAN.md`, `REFERENCES.md`

---

## UPDATED ON : 2026-02-27

### feat (2026-02-27) — FFT/RFFT Vulkan GPU implementation

1. **FFT Stockham radix-2/4/8 fully working**:
   - Fixed dispatch geometry: `vkCmdDispatch(batch_size, 1, 1)` not `(batch_size, tg_batch, 1)`
   - Added missing radix-8 codelet for n ≥ 512
   - Fixed `FFTPushConstants` array: `stockham[9]` not `stockham[8]` (9 supported radices)
   - Fixed `encoder.op_count++` missing — all GPU commands were silently dropped

2. **RFFT support added**:
   - Push constant overflow fixed: struct was 132 bytes (33×uint32), Vulkan limit is 128
   - Removed unused `rader[9]` field → struct drops to 96 bytes, `real` field now in-range
   - Added binding 2 (`float src_real[]`) to `fft.comp` for float32 → complex64 RFFT input
   - Truncates output to `n/2+1` when `params.real == 1`

3. **Tests**: Stage 17 FFT: 0/3 → **3/3** ✅

4. **Files changed**: `kernels/fft.comp`, `fft.cpp`, `PLAN.md`

---

## UPDATED ON : 2026-03-01

### fix (2026-03-01) — Unary shader enum mismatch + float16 arithmetic correctness

1. **Enum mismatch fixed in `unary.comp`**:
   - Root cause: `unary.comp` declared `UNARY_LOG2=10` after `LOG1P=9`, pushing `Sin=11` etc. But `primitives.cpp::UnaryOp` has `Sin=10, Cos=11, ...`. Result: 15 operations dispatched to the wrong opcode — `sin(0.5)` returned `log2(0.5)=-1`.
   - Fix: rewrote `unary.comp` constants to exactly match `primitives.cpp` enum values. All 27 subtests now pass.

2. **float16 arithmetic correctness in `unary.comp`**:
   - Root cause: float16 input was stored as 16-bit in a uint32 buffer. The shader read it with `uintBitsToFloat()` which reinterprets raw bits as float32 — producing garbage (exp(2.0 f16) returned 0.0).
   - Fix: added `input_elem_bytes` push constant to `dispatch_unary`. When `input_elem_bytes==2`, shader uses `unpackHalf2x16()`/`packHalf2x16()` and processes 2 f16 elements per thread. exp(2.0 f16)=7.39 ✅

3. **Log2/Log10 dispatch fixed**:
   - Root cause: `log2` and `log10` are handled by a single `Log` C++ class with a `Base` enum (e/two/ten). `UNARY_GPU(Log, Log)` generated only `Log::eval_gpu` with `op=Log`, discarding log2/log10.
   - Fix: removed `UNARY_GPU(Log, Log)` and replaced with explicit `Log::eval_gpu` that checks `state()` (the base) to select `UnaryOp::Log`, `UnaryOp::Log2`, or `UnaryOp::Log10`.

4. **Allocator 4-byte alignment**: `VulkanAllocator::malloc` and `alloc_staging` now pad all allocations to 4-byte boundary. Prevents Metal out-of-bounds when reading uint32 words from a float16 buffer sized as 2N bytes.

5. **Pipeline cache version v4→v7**: Bumped `kPipelineCacheVersion` to force recompilation after shader layout changes.

6. **Tests** (before → after for `pytest python/tests/test_ops.py -k unary`):
   - float32 sin/cos/tan/sinh/cosh/tanh/arcsin/arccos/arctan/arcsinh/arccosh/arctanh: ❌ wrong → ✅ correct
   - log2(0.5)=-1.0, log10(0.5)=-0.301: ❌ not dispatched → ✅ correct
   - float16 exp(2.0)=7.39, sin(1.0)=0.841: ❌ garbage → ✅ correct
   - `2 passed, 0 failed` ✅

7. **Files changed**: `kernels/unary.comp`, `backend/vulkan/primitives.cpp`, `backend/vulkan/device.cpp`, `backend/vulkan/allocator.cpp`

