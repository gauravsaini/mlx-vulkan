# MLX Vulkan Backend — Task Tracker

## Context

Implement a Vulkan compute backend for MLX (ml-explore/mlx) to enable the framework on Linux
with any Vulkan-capable GPU (AMD, NVIDIA, Intel). Mirrors the existing CUDA backend structure.
Target: Linux-first. macOS via MoltenVK deferred. Full primitive coverage. AOT SPIR-V kernels.

**Key contract**: `mlx/backend/gpu/eval.h` — 4 functions all GPU backends must implement:

- `gpu::new_stream(Stream)`, `gpu::eval(array&)`, `gpu::finalize(Stream)`, `gpu::synchronize(Stream)`

**Reference backends**: `mlx/backend/cuda/` (structure), `mlx/backend/metal/` (kernel patterns)

**Current device**: Apple M1 via MoltenVK (macOS development)
**Last verified**: 2026-02-28

---

## Build Status (as of 2026-02-28)

| Step                                                                                   | Status                  |
| -------------------------------------------------------------------------------------- | ----------------------- |
| `cmake -B build_vulkan -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON` | ✅ PASSES               |
| `cmake --build build_vulkan -j4`                                                       | ✅ PASSES (zero errors) |
| All SPIR-V shaders pass `spirv-val`                                                    | ✅                      |
| Python bindings (`mlx.core` importable)                                                | ✅                      |
| `test_stage13_indexing.py` (Gather/GatherAxis/ScatterAxis)                             | ✅ 7/7 PASS             |
| `test_stage14_sort.py` (Sort)                                                          | ✅ 6/6 PASS             |
| `test_stage15_scan.py` (Scan / prefix ops)                                             | ✅ 5/5 PASS             |
| `test_stage16_nn_extended.py` (LayerNorm, RMSNorm, RoPE, SoftMax)                     | ✅ 8/8 PASS             |
| `test_stage17_fft.py` (FFT/RFFT)                                                      | ✅ 3/3 PASS             |
| `test_stage18_concat.py` (Concatenate)                                                 | ✅ 3/3 PASS             |

**Post-build workflow** (required after any `.cpp` change):

```bash
cmake --build build_vulkan -j4
cp build_vulkan/core.cpython-314-darwin.so python/mlx/core.cpython-314-darwin.so
codesign --sign - --force python/mlx/core.cpython-314-darwin.so
```

**After any `.comp` shader change**, either:

```bash
glslc --target-env=vulkan1.2 mlx/backend/vulkan/kernels/FOO.comp -o build_vulkan/mlx/backend/vulkan/kernels/FOO.spv
```

or simply run the full `cmake --build build_vulkan -j4` (rebuilds shaders too).

---

## Known Issues / Critical Bugs Fixed

### Fixed (2026-02-26)

1. **eval.cpp — SIGSEGV on eval_gpu() deletion**: `eval_gpu()` call was accidentally deleted by a `sed` command. Restored manually. This was the root cause of segfaults during GPU dispatch.

2. **binary.comp — int32 arithmetic corruption on Apple Silicon**: Apple Silicon GPU flushes denormals-to-zero, meaning int32 values reinterpreted as float bits via `uintBitsToFloat()` were silently flushed to 0. Fixed by adding dtype-aware arithmetic paths (`DTYPE_FLOAT` / `DTYPE_INT` / `DTYPE_UINT`) so integer ops never pass through float interpretation.

3. **fft.cpp — broken FFT implementation**: Broken `RFFT`/`IRFFT` class names and `is_power_of_2` ambiguity caused compile errors. Replaced with minimal stubs that throw `runtime_error`.

4. **Stale Python extension after rebuild**: `build_vulkan/core.cpython-314-darwin.so` must be manually copied to `python/mlx/` and re-signed with `codesign` after each build — this is not automated by CMake.

### Fixed (2026-02-27) — FFT Implementation

5. **fft.comp — wrong dispatch geometry**: Vulkan was dispatching `vkCmdDispatch(batch_size, threadgroup_batch_size, 1)` with `local_size_x=256`. Metal dispatches 1 workgroup per batch item with `threads_per_fft × tg_batch_size` threads. Fixed: `vkCmdDispatch(batch_size, 1, 1)` with `local_size_x=1024`.

6. **fft.cpp — FFTPushConstants::stockham[8] off-by-one**: `supported_radices = {13,11,8,7,6,5,4,3,2}` has 9 entries. `stockham[8]` (radix-2 steps) was never stored — array was only 8 elements. Fixed: extended to `stockham[9]`, loop limit `i < 9`.

7. **fft_op — encoder.op_count never incremented**: `Device::commit()` returns early if `op_count == 0`. All GPU commands were silently dropped — FFT output was all zeros. Fixed: added `encoder.op_count++` after `vkCmdDispatch`.

8. **fft.comp — radix-8 not implemented**: For n≥512, `plan_stockham_fft` uses radix-8 (e.g. 512=8³). Shader had no radix-8 codelet. Added `radix8()`, `radix_butterfly_8()`, and radix-8 pass loop in `perform_fft`. n=512..4096 now pass.

9. **FFTPushConstants push constant overflow**: Struct was 132 bytes (33×uint32); Vulkan limit is 128 bytes. `real` field at offset 128 was always out-of-bounds → `is_rfft` always false, RFFT loaded complex64 from float32 input. Fixed: removed unused `rader[9]` from struct and GLSL block → struct drops to 96 bytes.

10. **RFFT support**: Added binding 2 (`float src_real[]`) to `fft.comp`. When `params.real==1 && params.inv==0`, loads float input as `vec2(x, 0.0)` and truncates output to `n/2+1`. Removed Metal-specific 2-RFFT batch halving from `fft.cpp`.

### Fixed (2026-02-28) — Gather/Scatter ND + Scan GPU dispatch

11. **Gather::eval_gpu — only handled 1D axis=0**: `mx.take(2D_array, idx, axis=1)` threw "Fallback to eval_cpu is unsupported". Fixed: added INDEX_GATHER_GEN (op=3) to `indexing.comp` — general ND gather using `j = tid / slice_total`, `outer_off = within / inner`, `inner_off = within % inner`, `src_pos = outer_off * src_outer_stride + idx[j] * inner + inner_off`. Now all single-axis gather cases are handled on GPU.

12. **GatherAxis/ScatterAxis push constant size mismatch**: Both functions were calling `get_pipeline("indexing", ..., 3, 32)` with a 9- or 8-field struct, but the shader declared 9 fields = 36 bytes. After the push constant was extended to 11 fields (44 bytes), the 32-byte pipeline layout caused vkCmdPushConstants to be out-of-range. Fixed: unified all three indexing ops to use shared `IndexPushConst` (44 bytes, 11 fields) via `indexing_dispatch()` helper.

13. **MoltenVK stale pipeline cache crash**: After changing the indexing push constant layout from 36→44 bytes, the on-disk pipeline cache (`~/.cache/mlx_vulkan_pipeline_cache.bin`) retained the old binary pipeline. MoltenVK's pipeline cache loader crashed with SIGKILL on dlopen. Fixed: added version suffix to cache path (`mlx_vulkan_pipeline_cache_v2.bin`); bump version whenever push constant layouts change.

14. **ScatterAxis negative-index wrap using wrong field**: `INDEX_SCATTER` was wrapping with `int(params.idx_size)` instead of `int(params.src_ax_size)`. Fixed in `indexing.comp`.

15. **Scan::eval_gpu unconditionally threw**: Prefix scan (cumsum/cumprod) always failed with a runtime_error. Implemented full GPU dispatch via a two-level Hillis-Steele scan in `scan.comp`: serial inclusive scan within each thread's chunk (chunk_size = ceil(scan_size/256)), parallel cross-chunk prefix on stotals[], propagate back, exclusive conversion at writeback. Supports scan_size ≤ 1024; Sum/Prod/Max/Min; inclusive and exclusive; reverse.

### Known Remaining Issues

- `unary.comp`: Similar int32 dtype issue may exist for ops like `abs`/`neg` on int32 inputs (not yet tested).
- Hadamard: Not yet implemented (`kernels/hadamard.comp` stub only).
- Scan: scan_size > 1024 throws (multi-pass GPU scan not yet implemented). LogAddExp variant not on GPU.
- Full `test_array.py` suite: Not yet fully run.
- `test_ops.py` suite: Not yet run.

---

## Phase 0: Repository Setup ✅ COMPLETE

- [x] Clone MLX into `/Users/ektasaini/Desktop/mlx-vulkan`
- [x] Install Vulkan toolchain (macOS via Homebrew: vulkan-headers, vulkan-loader, shaderc, spirv-tools, glslang, molten-vk, vulkan-validationlayers)
- [x] Verify Vulkan GPU detected: Apple M1 via MoltenVK
- [x] Read and understand `mlx/backend/gpu/eval.h` — interface contract
- [x] Read and understand `mlx/backend/no_gpu/primitives.cpp` — full list of ~80 ops to implement
- [x] Read and understand `mlx/backend/cuda/device.h` + `allocator.h` — structural template

---

## Phase 1: Build System ✅ COMPLETE

- [x] Add `MLX_BUILD_VULKAN` option to root `CMakeLists.txt`
  - `option(MLX_BUILD_VULKAN "Build Vulkan backend" OFF)`
  - `find_package(Vulkan REQUIRED)` guard block
  - FetchContent fallback for VulkanMemoryAllocator (header-only, include path only)
  - `add_subdirectory(mlx/backend/vulkan)` when `MLX_BUILD_VULKAN=ON`
  - Hooked into same `if(NOT MLX_BUILD_GPU)` guard as CUDA
- [x] Create `mlx/backend/vulkan/` directory
- [x] Create `mlx/backend/vulkan/CMakeLists.txt`
  - `compile_shader()` cmake function using glslc
  - 22 `.comp` shaders compiled to `.spv` — all SPIRV-val validated ✅
  - All `.cpp` sources in `target_sources(mlx PRIVATE ...)`
  - VMA linked as include-only (avoids CMake export set issue)
- [x] CMake configure succeeds with `-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF`
- [x] `cmake --build` succeeds — zero errors, only VMA nullability warnings (harmless)
- [x] **Key fix**: VMA FetchContent target cannot be in MLX export set → use `target_include_directories` with `${vulkanmemoryallocator_SOURCE_DIR}/include` instead of `target_link_libraries`

---

## Phase 2: Device Infrastructure ✅ COMPLETE

### `mlx/backend/vulkan/device.h` + `device.cpp`

- [x] `VulkanDevice` struct with: `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkPipelineCache`
- [x] Instance creation: enable `VK_KHR_get_physical_device_properties2`, validation layers (debug builds)
- [x] Physical device selection: prefer discrete GPU, fall back to integrated
- [x] Logical device + compute queue family discovery
- [x] Pipeline cache: persist to disk (`~/.cache/mlx_vulkan_pipeline_cache.bin`), load on init
- [x] `new_queue(int index)` — creates per-stream `VkCommandPool` + initial `VkCommandBuffer`
- [x] `get_command_buffer(int index)` — returns current recording command buffer for stream
- [x] `end_encoding(int index)` — `vkEndCommandBuffer`
- [x] `commit_command_buffer(int index)` — `vkQueueSubmit` + `vkResetCommandPool` for next
- [x] `command_buffer_needs_commit(int index)` — heuristic (same as Metal: command count threshold)
- [x] `get_pipeline(const std::string& name)` — load SPIR-V from VULKAN_KERNELS_PATH, create+cache `VkPipeline`
- [x] `VulkanDevice& device(mlx::core::Device dev)` — singleton accessor (mirrors `metal::device()`)
- [x] Descriptor pool + `allocate_descriptor_set(VkDescriptorSetLayout)` helper
- [x] `bind_buffer(VkDescriptorSet ds, uint32_t binding, const array& arr)` helper

### `mlx/backend/vulkan/utils.h` + `utils.cpp`

- [x] `div_ceil(uint64_t a, uint64_t b)` — dispatch grid helper
- [x] `insert_buffer_barrier(VkCommandBuffer, const array&)` — pipeline barrier for compute→compute RAW
- [x] `to_vk_format(Dtype)` — mlx dtype → `VkFormat` mapping
- [x] `get_type_string(Dtype)` — for pipeline name keying

---

## Phase 3: Memory Allocator ✅ COMPLETE

### `mlx/backend/vulkan/allocator.h` + `allocator.cpp`

- [x] Include VMA (`vk_mem_alloc.h`) — create `VmaAllocator` on device init
- [x] `VulkanAllocator` class extending `mlx::core::allocator::Allocator`
- [x] `Buffer malloc(size_t size)` — `vmaCreateBuffer` with `VMA_MEMORY_USAGE_AUTO` + `DEVICE_LOCAL`
- [x] `void free(Buffer buffer)` — `vmaDestroyBuffer`
- [x] `size_t size(Buffer buffer) const` — `vmaGetAllocationInfo`
- [x] `void* Buffer::raw_ptr()` — staging buffer mapped pointer for host access
- [x] Staging buffer pool for CPU↔GPU transfers (discrete GPU requires explicit staging)
- [x] `active_memory_`, `peak_memory_`, `memory_limit_` tracking (mirrors MetalAllocator)
- [x] `get_active_memory()`, `get_peak_memory()`, `set_memory_limit()` free functions
- [x] `clear_cache()` — drain VMA pool (mirrors `metal::clear_cache()`)
- [x] `Allocator& allocator()` free function — returns singleton `VulkanAllocator`

---

## Phase 4: Event, Fence, Device Info

### `mlx/backend/vulkan/event.h` + `event.cpp`

- [ ] `VulkanEvent` wrapping `VkSemaphore` (timeline semaphore preferred — `VK_KHR_timeline_semaphore`)
- [ ] `signal(uint64_t value)` — `vkSignalSemaphore`
- [ ] `wait(uint64_t value)` — `vkWaitSemaphores`
- [ ] Integrate with MLX `Event` type (mirrors `metal/event.cpp`)

### `mlx/backend/vulkan/fence.cpp`

- [ ] `VulkanFence` wrapping `VkFence` for CPU-GPU synchronization
- [ ] Used in `gpu::synchronize()` to block CPU until stream completes

### `mlx/backend/vulkan/device_info.cpp`

- [ ] Implement `mlx::core::gpu::device_info()` — return GPU name, VRAM, compute capability
- [ ] Query `VkPhysicalDeviceProperties` + `VkPhysicalDeviceMemoryProperties`
- [ ] Implement `mlx::core::metal::device_info()` stub if required by headers

---

## Phase 5: GPU Eval Dispatch ✅ COMPLETE

### `mlx/backend/vulkan/eval.cpp` — implements `mlx/backend/gpu/eval.h`

- [x] `gpu::new_stream(Stream stream)` — calls `vulkan::device(...).new_queue(stream.index)`
- [x] `gpu::eval(array& arr)` — full dispatch loop (mirrors `metal/eval.cpp`):
  - Get command buffer for stream
  - Call `arr.primitive().eval_gpu(inputs, outputs)`
  - Track input/output buffer lifetimes
  - Check `command_buffer_needs_commit()` → submit if true
  - Register completion handler → `scheduler::notify_task_completion(s)`
- [x] `gpu::finalize(Stream s)` — end encoding + queue submit
- [x] `gpu::synchronize(Stream s)` — end encoding + submit + `vkQueueWaitIdle` (CPU blocks)
- **Note**: eval_gpu() call was accidentally deleted by sed and restored — SIGSEGV was the symptom.

---

## Phase 6: GPU Copy & Slicing 🔄 PARTIAL

### `mlx/backend/vulkan/copy.cpp` — implements `mlx/backend/gpu/copy.h`

- [x] `copy_gpu(src, out, ctype, s)` — dispatch `copy.comp` shader
- [x] `copy_gpu_inplace(in, out, data_shape, i_strides, o_strides, ...)` — strided copy shader
- [x] `fill_gpu(val, out, s)` — dispatch `fill.comp` (scalar broadcast)
- [x] `contiguous_copy_gpu(arr, s)` — returns contiguous buffer copy
- [x] `reshape_gpu(in, out, s)` — transpose+copy via shader
- [x] `flatten_in_eval`, `reshape_in_eval`, `swapaxes_in_eval` helper stubs

### `mlx/backend/vulkan/slicing.cpp` — implements `mlx/backend/gpu/slicing.h`

- [x] `slice_gpu(...)` — dispatch `slicing.comp`
- [ ] `pad_gpu(...)` — dispatch `pad.comp`
- [ ] `concatenate_gpu(...)` — dispatch `concatenate.comp`

---

## Phase 7: GLSL Compute Kernels (AOT SPIR-V) 🔄 PARTIAL

All shaders live in `mlx/backend/vulkan/kernels/`. Each compiled to `.spv` at build time via glslc.
Naming: `<op>_<dtype>.comp` or template + specialization constants for dtype variants.

**Total shaders compiled**: 22 ✅

### Kernel Conventions (apply to all shaders)

- [x] Default workgroup: `layout(local_size_x = 256) in;`
- [x] Bounds check: `if (idx >= size) return;`
- [x] Use push constants for params ≤ 128 bytes; UBO for larger metadata
- [ ] `kernels/bf16.glsl` — bfloat16 pack/unpack/arithmetic helpers
- [ ] `kernels/defines.glsl` — common defines, type aliases, math constants
- [ ] `kernels/utils.glsl` — index flattening, strides, broadcasting helpers

### Tier 1 — Core ✅ COMPLETE

- [x] `kernels/copy.comp` — contiguous, strided, scalar fill, broadcast
- [x] `kernels/unary.comp` — abs, neg, sign, sqrt, rsqrt, cos, sin, exp, log, relu, sigmoid, tanh, ...
  - **Note**: int32 dtype path may need audit (similar to binary.comp issue)
- [x] `kernels/binary.comp` — add, sub, mul, div, pow, min, max, eq, ne, lt, le, gt, ge, logical ops
  - **Fixed**: dtype-aware arithmetic paths added for DTYPE_FLOAT/INT/UINT (Apple Silicon denormal flush bug)
- [x] `kernels/arange.comp` — fill with range

### Tier 2 — Reduction & Matmul ✅ COMPLETE

- [x] `kernels/reduce.comp` — sum, min, max, prod along arbitrary axes (subgroup + workgroup reduction)
- [x] `kernels/arg_reduce.comp` — argmin, argmax
- [x] `kernels/matmul.comp` — tiled GEMM (16×16 or 32×32 tile), handles non-square shapes
- [x] `kernels/binary_two.comp` — two-output binary ops (divmod etc.)
- [x] `kernels/ternary.comp` — select/where (conditional elementwise)

### Tier 3 — Neural Net Essentials 🔄 PARTIAL

- [x] `kernels/softmax.comp` — numerically stable softmax (max-subtract + exp + sum + divide)
- [x] `kernels/logsumexp.comp`
- [x] `kernels/normalization.comp` — layer_norm, rms_norm (mean/variance in subgroup)
- [x] `kernels/rope.comp` — rotary position embeddings
- [x] `kernels/scan.comp` — prefix scan (inclusive/exclusive, add/mul)

### Tier 4 — Indexing & Shape Ops 🔄 PARTIAL

- [x] `kernels/indexing.comp` — gather (read at indices), scatter (write at indices), scatter-add
- [x] `kernels/slicing.comp` — handled naturally via general strided `copy.comp` indexing!
- [x] `kernels/pad.comp` — handled naturally via general strided `copy.comp` indexing!
- [x] `kernels/sort.comp` — bitonic sort (GPU-parallel)

### Tier 5 — Advanced Ops 🔄 PARTIAL

- [x] `kernels/conv.comp` — convolution (im2col approach, dispatch to matmul)
- [x] `kernels/fft.comp` — **COMPLETE** (Stockham Cooley-Tukey radix-2/4/8; RFFT float input via binding 2; 3/3 tests pass)
- [x] `kernels/hadamard.comp` — stub only (throws runtime_error, not yet implemented)
- [ ] `kernels/attention.comp` — scaled dot-product attention (fused QK^T·V)
- [x] `kernels/quantized.comp` — affine quantize/dequantize (int4, int8)
- [x] `kernels/random.comp` — Philox / Threefry PRNG for `mx.random.*`
- [x] `kernels/rbits.comp` — RandomBits Threefry PRNG

---

## Phase 8: Primitives Dispatch 🔄 PARTIAL

### `mlx/backend/vulkan/primitives.cpp`

Implement `eval_gpu()` for every primitive. Pattern per op:

1. Get VkCommandBuffer from stream
2. Get cached VkPipeline by name
3. Allocate + write VkDescriptorSet (bind input/output arrays)
4. Push constants (size, strides, type params)
5. `vkCmdDispatch(cmd, ceil(n/256), 1, 1)`
6. Insert memory barrier

#### Elementwise Unary (dispatch `unary.comp` with op specialization constant)

- [x] Abs, Arccos, Arcsin, Arctan, Ceil, Cos, Cosh, Erf, Erfinv
- [x] Exp, Expm1, Floor, Log, Log1p, Log2, Neg, Round, Rsqrt
- [x] Sigmoid, Sign, Sin, Sinh, Sqrt, Square, StopGradient, Tan, Tanh

#### Elementwise Binary (dispatch `binary.comp`)

- [x] Add, ArcTan2, BitAnd, BitOr, BitXor, Divide
- [x] Equal, FloorDivide, Greater, GreaterEqual, LeftShift
- [x] Less, LessEqual, LogAddExp, Maximum, Minimum
- [x] Multiply, NotEqual, Power, Remainder, RightShift, Subtract

#### Elementwise Ternary

- [x] Select (where)

#### Reduction

- [x] Reduce (sum, min, max, prod, logsum — axis-wise)
- [x] ArgReduce (argmin, argmax)

#### Shape / Memory

- [x] Arange
- [x] AsType (type cast) — verified: int32→float32 ✅
- [x] AsStrided
- [x] Broadcast
- [x] Concatenate
- [x] Copy (contiguous copy)
- [x] Flatten (via reshape)
- [ ] NumberOfElements
- [x] Pad
- [x] Reshape (via copy_gpu) — verified: [4]→[2,2] ✅
- [x] Slice, SliceUpdate
- [ ] Split
- [x] Squeeze, Expand
- [x] Transpose
- [ ] Unflatten
- [ ] View

#### Linear Algebra

- [x] AddMM (A + alpha \* B @ C)
- [ ] BlockMaskedMM
- [ ] GatherMM, GatherQMM
- [ ] SegmentedMM
- [x] Matmul — verified ✅
- [ ] QuantizedMatmul
- [ ] QQMatmul
- [ ] QRF, SVD, Inverse, Cholesky, Eig, Eigh, LUF (Advanced Linear Algebra - CPU Fallbacks pending)

#### Neural Net Ops

- [x] Conv1D, Conv2D, Conv3D (ConvolutionVjp)
- [x] FFT, RFFT — **COMPLETE** (Stockham radix-2/4/8 GPU dispatch; 3/3 tests passing)
- [x] IFFT, IRFFT — dispatch implemented; inverse path via `params.inv=1`
- [x] Hadamard — stub throws runtime_error (Phase G needed)
- [x] LayerNorm, RMSNorm (GPU dispatch via `normalization.comp`)
- [x] LogSumExp
- [x] Rope (GPU dispatch via `rope.comp`)
- [x] ScaledDotProductAttention
- [x] Softmax — verified via smoke test ✅
- [x] Scan (prefix ops — GPU dispatch via scan.comp; Hillis-Steele 2-level; scan_size ≤ 1024; 5/5 tests pass)

#### Indexing

- [x] Gather (unsupported bounds native halt via exceptions)
- [x] GatherAxis, ScatterAxis (GPU dispatch via `indexing.comp`)
- [x] Scatter (unsupported bounds native halt via exceptions)

#### Sort

- [x] ArgSort (unsupported bounds native halt via exceptions)
- [x] Sort (GPU dispatch via `sort.comp`, bitonic ≤256)
- [x] Partition, ArgPartition (unsupported bounds native halt via exceptions)

#### Random

- [ ] BernoulliWithCDF
- [x] RandomBits (Threefry PRNG via `rbits.comp`)

#### Quantization

- [ ] AffineQuantize, DequantizedMatmul

#### Misc

- [ ] Compiled (fused kernel — stub, complex)
- [ ] CustomVJP, CustomTransforms (CPU fallback OK for now)
- [ ] Depends (sync primitive)
- [ ] Load (mmap)
- [ ] Jit (stub)

---

## Phase 9: Integration & Testing 🔄 PARTIAL

### Build Validation ✅ COMPLETE

- [x] `cmake -B build_vulkan -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON` — succeeds
- [x] `cmake --build build_vulkan -j4` — zero errors (only VMA nullability warnings)
- [x] All 22 `.comp` shaders compile to `.spv` without errors
- [x] All 22 `.spv` pass `spirv-val` validation

### Smoke Tests ✅ PASSING

- [x] Vulkan device detected: Apple M1 (via MoltenVK)
- [x] VK_EXT_external_memory_host available (zero-copy path enabled)
- [x] `add` (float32): `mx.ones(4) + mx.full(4, 3)` → 4.0 ✅
- [x] `multiply` (float32): `mx.full(4,2) * mx.full(4,3)` → 6.0 ✅
- [x] `arange`: `mx.arange(0,4,1)[2]` → 2.0 ✅
- [x] `sum` reduction: `mx.sum(mx.ones(4))` → 4.0 ✅
- [x] `matmul`: `mx.matmul(ones(2,3), ones(3,2))[0,0]` → 3.0 ✅
- [x] `exp` (unary): `mx.exp(mx.zeros(4))[0]` → 1.0 ✅
- [x] `maximum` (binary): `mx.maximum(ones*1, ones*2)[0]` → 2.0 ✅
- [x] `int32 add/sub/mul/div`: `[1,2,3]+[4,5,6]=[5,7,9]` ✅
- [x] `int32 eq/lt/gt comparisons` ✅
- [x] `float32 add/mul` ✅
- [x] `astype int32→float32`: `[1,2,3]→[1.0,2.0,3.0]` ✅
- [x] `bool equality (a==a)`: `[True,True,True]` ✅
- [x] `reshape [4]→[2,2]` ✅
- [x] `softmax` (from previous session) ✅

### Python REPL Tests — In Progress 🔄

- [x] `int32` arithmetic correctness (fixed binary.comp dtype bug)
- [ ] `test_array.py` full suite — in progress
- [ ] `test_ops.py` suite — not yet run
- [ ] `test_random.py` — not yet run
- [ ] `unary.comp` int32 paths (abs, neg on int32) — needs audit

### Numerical Equivalence (vs CPU)

- [ ] Write `tests/vulkan_equivalence.py` — compare GPU vs CPU output for all primitives
- [ ] Tolerance: `atol=1e-4` for float32, `atol=1e-2` for float16/bfloat16
- [ ] Matmul equivalence for sizes: 4×4, 128×128, 512×512, 1024×1024
- [ ] Reduction equivalence along all axes

### MLX Test Suite

- [ ] `python -m pytest tests/ -x -v` — all existing tests pass on Vulkan backend
- [ ] `python -m pytest tests/test_ops.py` — op-level coverage
- [ ] `python -m pytest tests/test_random.py` — RNG reproducibility

### Performance Baselines

- [ ] Run `benchmarks/` matmul benchmark — record GFLOPS for comparison
- [ ] Compare against CPU backend throughput

---

## Phase 10: Continuous Integration

- [ ] Add `.github/workflows/vulkan.yml` — build + smoke test on Ubuntu runner with GPU
- [ ] Add `MLX_BUILD_VULKAN` to CI matrix
- [ ] Add SPIR-V shader validation step: `spirv-val kernels/*.spv`

---

## Reference Links

- MLX CUDA backend (structure reference): `mlx/backend/cuda/`
- MLX gpu/ interface contract: `mlx/backend/gpu/eval.h`
- [Vulkan-Samples by LunarG](https://github.com/KhronosGroup/Vulkan-Samples) — SPIR-V loading, init boilerplate
- [Shaderc](https://github.com/google/shaderc) — GLSL→SPIR-V, AOT + optional JIT
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) — SPIR-V reflection for descriptor layout
- [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) — device memory allocator
- [Vulkan Compute for AI](https://github.com/PacktPublishing/Vulkan-Compute) — compute shader patterns
- GitHub issue: https://github.com/ml-explore/mlx/issues/1751

---

## Key Technical Decisions

| Decision        | Choice                          | Rationale                                                                         |
| --------------- | ------------------------------- | --------------------------------------------------------------------------------- |
| Shader language | GLSL 4.60 compute               | Standard, mature tooling, all Vulkan drivers                                      |
| Compilation     | AOT via glslc at build time     | No runtime compiler dep, faster startup                                           |
| Memory          | VMA (VulkanMemoryAllocator)     | Handles suballocation, mirrors MetalAllocator pooling                             |
| bfloat16        | uint16 storage + manual ops     | No native Vulkan bf16 — same approach as metal bf16.h                             |
| Kernel variants | VkSpecializationInfo per dtype  | Avoids JIT string templates, compile-time branching                               |
| macOS           | MoltenVK (active dev target)    | Linux remains primary; macOS M1 used for iteration                                |
| Discrete GPU    | Staging buffers for host access | No unified memory — explicit CPU↔GPU transfer path                                |
| int32 on GPU    | Dtype-aware GLSL paths          | Apple Silicon flushes denormals-to-zero; int bits must not pass through float ops |

---

## Next Steps (Priority Order)

### Immediate: Unblock test_array.py and test_ops.py

1. Audit `unary.comp` for int32 dtype correctness (same class of bug as binary.comp fix).
2. Implement missing shape/memory primitives: `Concatenate`, `Split`, `Pad`, `NumberOfElements`.
3. Run `test_array.py` suite to completion; log all failures.
4. Run `test_ops.py` suite; identify remaining CPU fallback ops causing failures.

### Phase G: FFT / Hadamard (Spectral Ops) — FFT COMPLETE ✅

- [x] `FFT`, `RFFT` implemented in `kernels/fft.comp` — Stockham Cooley-Tukey radix-2/4/8. 3/3 tests passing.
- [x] `IFFT`, `IRFFT` dispatch paths wired (inverse twiddles via `params.inv=1`).
- [ ] `Hadamard` in `kernels/hadamard.comp` — still stub, needs recursive Walsh-Hadamard transform.
- [ ] Rader / Bluestein paths in `fft.cpp` — dispatch wired but `fft.comp` only handles Stockham; non-power-of-2 sizes currently fall through to error.

### Option 1: Quantization & Advanced LLM Support (Recommended next milestone)

Essential for running large models efficiently natively on the Vulkan device.

- **Goals**: Implement `AffineQuantize`, `DequantizedMatmul`, `QuantizedMatmul`.
- **Secondary**: `BlockMaskedMM`, `GatherMM`, `GatherQMM`.

### Option 2: Multi-Axis Indexing & Fused Kernels

The 1D logic is solid (`GatherAxis`, `ScatterAxis`), but multi-axis generalizations still drop to CPU.

- **Goals**: Move general multi-axis `Gather` and `Scatter` fully to GPU shaders.
- **Fusions**: Introduce the `Compiled` primitive to accelerate chained elementwise blocks.

### Option 3: Full Validation Pipeline & CI Integration

- Create automated numeric equivalence tests validating `float32`, `float16`, `bfloat16` accuracy.
- Setup CI integration with regression binding to existing test suite.
- Add `tests/vulkan_equivalence.py`.
