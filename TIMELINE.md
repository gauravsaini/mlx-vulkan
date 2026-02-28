# MLX Vulkan Backend — Change Timeline

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
