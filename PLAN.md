# MLX Vulkan Backend — Task Tracker

## Context

Implement a Vulkan compute backend for MLX (ml-explore/mlx) to enable the framework on Linux
with any Vulkan-capable GPU (AMD, NVIDIA, Intel). Mirrors the existing CUDA backend structure.
Target: Linux-first. macOS via MoltenVK deferred. Full primitive coverage. AOT SPIR-V kernels.

**Key contract**: `mlx/backend/gpu/eval.h` — 4 functions all GPU backends must implement:

- `gpu::new_stream(Stream)`, `gpu::eval(array&)`, `gpu::finalize(Stream)`, `gpu::synchronize(Stream)`

**Reference backends**: `mlx/backend/cuda/` (structure), `mlx/backend/metal/` (kernel patterns)

---

## Phase 0: Repository Setup ✅ Prerequisite

- [ ] Clone MLX into `/Users/ektasaini/Desktop/mlx-vulkan`
  ```bash
  git clone https://github.com/ml-explore/mlx.git /Users/ektasaini/Desktop/mlx-vulkan
  ```
- [ ] Install Vulkan toolchain on Linux target machine
  ```bash
  sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers \
    glslc glslang-tools spirv-tools libshaderc-dev libvulkan-memory-allocator-dev
  ```
- [ ] Install macOS dev toolchain (for iteration)
  ```bash
  brew install vulkan-headers vulkan-loader vulkan-tools shaderc spirv-tools \
    glslang molten-vk vulkan-validationlayers
  ```
- [ ] Verify Vulkan GPU detected: `vulkaninfo --summary`
- [ ] Read and understand `mlx/backend/gpu/eval.h` — this is the interface contract
- [ ] Read and understand `mlx/backend/no_gpu/primitives.cpp` — full list of ~80 ops to implement
- [ ] Read and understand `mlx/backend/cuda/device.h` + `allocator.h` — structural template

---

## Phase 1: Build System

- [ ] Add `MLX_BUILD_VULKAN` option to root `CMakeLists.txt`
  - Add `option(MLX_BUILD_VULKAN "Build Vulkan backend" OFF)`
  - Add `find_package(Vulkan REQUIRED)` guard block
  - Add `find_package(VulkanMemoryAllocator REQUIRED)` or FetchContent fallback
  - Add `add_subdirectory(mlx/backend/vulkan)` when `MLX_BUILD_VULKAN=ON`
  - Hook into the same `if(NOT MLX_BUILD_GPU)` guard as CUDA
- [ ] Create `mlx/backend/vulkan/` directory
- [ ] Create `mlx/backend/vulkan/CMakeLists.txt`
  - Define `compile_shader(SHADER_FILE)` cmake function that runs `glslc`
  - List all `.comp` shader files as compile targets → `.spv` outputs
  - `add_custom_target(vulkan_shaders DEPENDS ${SPIRV_OUTPUTS})`
  - List all `.cpp` sources via `target_sources(mlx PRIVATE ...)`
  - `target_link_libraries(mlx PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)`
  - `target_compile_definitions(mlx PRIVATE VULKAN_KERNELS_PATH="...")`
- [ ] Verify CMake configure succeeds with `-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF`
- [ ] Verify glslc compiles a minimal test shader during build

---

## Phase 2: Device Infrastructure

### `mlx/backend/vulkan/device.h` + `device.cpp`

- [ ] `VulkanDevice` struct with: `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkPipelineCache`
- [ ] Instance creation: enable `VK_KHR_get_physical_device_properties2`, validation layers (debug builds)
- [ ] Physical device selection: prefer discrete GPU, fall back to integrated
- [ ] Logical device + compute queue family discovery
- [ ] Pipeline cache: persist to disk (`~/.cache/mlx_vulkan_pipeline_cache.bin`), load on init
- [ ] `new_queue(int index)` — creates per-stream `VkCommandPool` + initial `VkCommandBuffer`
- [ ] `get_command_buffer(int index)` — returns current recording command buffer for stream
- [ ] `end_encoding(int index)` — `vkEndCommandBuffer`
- [ ] `commit_command_buffer(int index)` — `vkQueueSubmit` + `vkResetCommandPool` for next
- [ ] `command_buffer_needs_commit(int index)` — heuristic (same as Metal: command count threshold)
- [ ] `get_pipeline(const std::string& name)` — load SPIR-V from VULKAN_KERNELS_PATH, create+cache `VkPipeline`
- [ ] `VulkanDevice& device(mlx::core::Device dev)` — singleton accessor (mirrors `metal::device()`)
- [ ] Descriptor pool + `allocate_descriptor_set(VkDescriptorSetLayout)` helper
- [ ] `bind_buffer(VkDescriptorSet ds, uint32_t binding, const array& arr)` helper

### `mlx/backend/vulkan/utils.h` + `utils.cpp`

- [ ] `div_ceil(uint64_t a, uint64_t b)` — dispatch grid helper
- [ ] `insert_buffer_barrier(VkCommandBuffer, const array&)` — pipeline barrier for compute→compute RAW
- [ ] `to_vk_format(Dtype)` — mlx dtype → `VkFormat` mapping
- [ ] `get_type_string(Dtype)` — for pipeline name keying

---

## Phase 3: Memory Allocator ✅

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

## Phase 5: GPU Eval Dispatch

### `mlx/backend/vulkan/eval.cpp` — implements `mlx/backend/gpu/eval.h`

- [ ] `gpu::new_stream(Stream stream)` — calls `vulkan::device(...).new_queue(stream.index)`
- [ ] `gpu::eval(array& arr)` — full dispatch loop (mirrors `metal/eval.cpp`):
  - Get command buffer for stream
  - Call `arr.primitive().eval_gpu(inputs, outputs)`
  - Track input/output buffer lifetimes
  - Check `command_buffer_needs_commit()` → submit if true
  - Register completion handler → `scheduler::notify_task_completion(s)`
- [ ] `gpu::finalize(Stream s)` — end encoding + queue submit
- [ ] `gpu::synchronize(Stream s)` — end encoding + submit + `vkQueueWaitIdle` (CPU blocks)

---

## Phase 6: GPU Copy & Slicing

### `mlx/backend/vulkan/copy.cpp` — implements `mlx/backend/gpu/copy.h`

- [ ] `copy_gpu(src, out, ctype, s)` — dispatch `copy.comp` shader
- [ ] `copy_gpu_inplace(in, out, data_shape, i_strides, o_strides, ...)` — strided copy shader
- [ ] `fill_gpu(val, out, s)` — dispatch `fill.comp` (scalar broadcast)
- [ ] `contiguous_copy_gpu(arr, s)` — returns contiguous buffer copy
- [ ] `reshape_gpu(in, out, s)` — transpose+copy via shader
- [ ] `flatten_in_eval`, `reshape_in_eval`, `swapaxes_in_eval` helper stubs

### `mlx/backend/vulkan/slicing.cpp` — implements `mlx/backend/gpu/slicing.h`

- [ ] `slice_gpu(...)` — dispatch `slicing.comp`
- [ ] `pad_gpu(...)` — dispatch `pad.comp`
- [ ] `concatenate_gpu(...)` — dispatch `concatenate.comp`

---

## Phase 7: GLSL Compute Kernels (AOT SPIR-V)

All shaders live in `mlx/backend/vulkan/kernels/`. Each compiled to `.spv` at build time via glslc.
Naming: `<op>_<dtype>.comp` or template + specialization constants for dtype variants.

### Kernel Conventions (apply to all shaders)

- [ ] Define `bf16_t` as `uint16_t` + manual pack/unpack helpers (`bf16.glsl` include)
- [ ] Use `GL_EXT_shader_explicit_arithmetic_types` for float16 (`f16vec4` etc.)
- [ ] Default workgroup: `layout(local_size_x = 256) in;`
- [ ] Bounds check: `if (idx >= size) return;`
- [ ] Use push constants for params ≤ 128 bytes; UBO for larger metadata

### Utility Headers

- [ ] `kernels/bf16.glsl` — bfloat16 pack/unpack/arithmetic helpers
- [ ] `kernels/defines.glsl` — common defines, type aliases, math constants
- [ ] `kernels/utils.glsl` — index flattening, strides, broadcasting helpers

### Tier 1 — Core (Unblock everything else)

- [ ] `kernels/copy.comp` — contiguous, strided, scalar fill, broadcast
- [ ] `kernels/unary.comp` — abs, neg, sign, sqrt, rsqrt, cos, sin, exp, log, relu, sigmoid, tanh, ...
- [ ] `kernels/binary.comp` — add, sub, mul, div, pow, min, max, eq, ne, lt, le, gt, ge, logical ops
- [ ] `kernels/arange.comp` — fill with range

### Tier 2 — Reduction & Matmul (Critical for ML)

- [ ] `kernels/reduce.comp` — sum, min, max, prod along arbitrary axes (subgroup + workgroup reduction)
- [ ] `kernels/arg_reduce.comp` — argmin, argmax
- [ ] `kernels/matmul.comp` — tiled GEMM (16×16 or 32×32 tile), handles non-square shapes
- [ ] `kernels/binary_two.comp` — two-output binary ops (divmod etc.)
- [ ] `kernels/ternary.comp` — select/where (conditional elementwise)

### Tier 3 — Neural Net Essentials

- [ ] `kernels/softmax.comp` — numerically stable softmax (max-subtract + exp + sum + divide)
- [ ] `kernels/logsumexp.comp`
- [ ] `kernels/normalization.comp` — layer_norm, rms_norm (mean/variance in subgroup)
- [ ] `kernels/rope.comp` — rotary position embeddings
- [ ] `kernels/scan.comp` — prefix scan (inclusive/exclusive, add/mul)

### Tier 4 — Indexing & Shape Ops

- [ ] `kernels/indexing.comp` — gather (read at indices), scatter (write at indices), scatter-add
- [ ] `kernels/slicing.comp` — strided slice read/write
- [ ] `kernels/pad.comp` — zero/constant padding
- [ ] `kernels/sort.comp` — bitonic sort (GPU-parallel)

### Tier 5 — Advanced Ops

- [ ] `kernels/conv.comp` — convolution (im2col approach, dispatch to matmul)
- [ ] `kernels/fft.comp` — Cooley-Tukey FFT (radix-2, radix-4)
- [ ] `kernels/hadamard.comp` — Hadamard transform
- [ ] `kernels/attention.comp` — scaled dot-product attention (fused QK^T·V)
- [ ] `kernels/quantized.comp` — affine quantize/dequantize (int4, int8)
- [ ] `kernels/random.comp` — Philox / Threefry PRNG for `mx.random.*`

---

## Phase 8: Primitives Dispatch

### `mlx/backend/vulkan/primitives.cpp`

Implement `eval_gpu()` for every primitive. Pattern per op:

1. Get VkCommandBuffer from stream
2. Get cached VkPipeline by name
3. Allocate + write VkDescriptorSet (bind input/output arrays)
4. Push constants (size, strides, type params)
5. `vkCmdDispatch(cmd, ceil(n/256), 1, 1)`
6. Insert memory barrier

#### Elementwise Unary (dispatch `unary.comp` with op specialization constant)

- [ ] Abs, Arccos, Arcsin, Arctan, Ceil, Cos, Cosh, Erf, Erfinv
- [ ] Exp, Expm1, Floor, Log, Log1p, Log2, Neg, Round, Rsqrt
- [ ] Sigmoid, Sign, Sin, Sinh, Sqrt, Square, StopGradient, Tan, Tanh

#### Elementwise Binary (dispatch `binary.comp`)

- [ ] Add, ArcTan2, BitAnd, BitOr, BitXor, Divide
- [ ] Equal, FloorDivide, Greater, GreaterEqual, LeftShift
- [ ] Less, LessEqual, LogAddExp, Maximum, Minimum
- [ ] Multiply, NotEqual, Power, Remainder, RightShift, Subtract

#### Elementwise Ternary

- [ ] Select (where)

#### Reduction

- [ ] Reduce (sum, min, max, prod, logsum — axis-wise)
- [ ] ArgReduce (argmin, argmax)

#### Shape / Memory

- [ ] Arange
- [ ] AsType (type cast)
- [ ] AsStrided
- [ ] Broadcast
- [ ] Concatenate
- [ ] Copy (contiguous copy)
- [ ] Flatten (via reshape)
- [ ] NumberOfElements
- [ ] Pad
- [ ] Reshape (via copy_gpu)
- [ ] Slice, SliceUpdate
- [ ] Split
- [ ] Squeeze, Expand
- [ ] Transpose

#### Linear Algebra

- [ ] AddMM (A + alpha \* B @ C)
- [ ] BlockMaskedMM
- [ ] GatherMM, GatherQMM
- [ ] Matmul
- [ ] QuantizedMatmul

#### Neural Net Ops

- [ ] Conv1D, Conv2D, Conv3D (ConvolutionVjp)
- [ ] FFT, RFFT, IFFT, IRFFT
- [ ] Hadamard
- [x] LayerNorm, RMSNorm (GPU dispatch via `normalization.comp`)
- [ ] LogSumExp
- [x] Rope (GPU dispatch via `rope.comp`)
- [ ] ScaledDotProductAttention
- [ ] Softmax
- [x] Scan (prefix ops, GPU dispatch via `scan.comp`, ≤512)

#### Indexing

- [ ] Gather (CPU fallback, multi-axis complex)
- [x] GatherAxis, ScatterAxis (GPU dispatch via `indexing.comp`)

#### Sort

- [ ] ArgSort (CPU fallback)
- [x] Sort (GPU dispatch via `sort.comp`, bitonic ≤512)
- [ ] Partition, ArgPartition (CPU fallback)

#### Random

- [ ] BernoulliWithCDF, RandomBits (Philox PRNG)

#### Quantization

- [ ] AffineQuantize, DequantizedMatmul

#### Misc

- [ ] Compiled (fused kernel — stub, complex)
- [ ] CustomVJP, CustomTransforms (CPU fallback OK for now)
- [ ] Depends (sync primitive)
- [ ] Load (mmap)
- [ ] Jit (stub)

---

## Phase 9: Integration & Testing

### Build Validation

- [ ] `cmake -B build -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON`
- [ ] `cmake --build build -j$(nproc)` — zero errors, zero warnings
- [ ] All `.comp` shaders compile to `.spv` without errors

### Smoke Tests

- [ ] `vulkaninfo --summary` — GPU detected
- [ ] Basic array creation + add: `mx.add(mx.ones(4), mx.ones(4))`
- [ ] Matmul: `mx.matmul(mx.ones((4,4)), mx.ones((4,4)))` → all 4s
- [ ] Reduction: `mx.sum(mx.array([1,2,3,4]))` → 10
- [ ] Softmax: `mx.softmax(mx.array([1.0, 2.0, 3.0]))` — sums to 1

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

| Decision        | Choice                          | Rationale                                                 |
| --------------- | ------------------------------- | --------------------------------------------------------- |
| Shader language | GLSL 4.60 compute               | Standard, mature tooling, all Vulkan drivers              |
| Compilation     | AOT via glslc at build time     | No runtime compiler dep, faster startup                   |
| Memory          | VMA (VulkanMemoryAllocator)     | Handles suballocation, mirrors MetalAllocator pooling     |
| bfloat16        | uint16 storage + manual ops     | No native Vulkan bf16 — same approach as metal bf16.h     |
| Kernel variants | VkSpecializationInfo per dtype  | Avoids JIT string templates, compile-time branching       |
| macOS           | Deferred (MoltenVK)             | macOS already has native Metal backend; Linux is priority |
| Discrete GPU    | Staging buffers for host access | No unified memory — explicit CPU↔GPU transfer path        |
