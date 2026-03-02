# Plan: MLX Vulkan Backend

## Context

MLX currently supports Apple Silicon (Metal) and NVIDIA (CUDA). There is no path to run MLX on Linux AMD/Intel/NVIDIA GPUs outside CUDA. This plan implements a Vulkan compute backend (`mlx/backend/vulkan/`) that satisfies the existing `mlx::core::gpu` interface — enabling MLX on any Vulkan-capable GPU on Linux. macOS support via MoltenVK is deferred.

- **Triggered by**: GitHub issue #1751 (ml-explore/mlx) — community request, core team acknowledged feasibility but no bandwidth
- **Target**: Linux-first, full primitive coverage (~80+ ops), AOT SPIR-V kernels (GLSL → compiled .spv at build time via shaderc/glslc)
- **Key insight**: The `mlx/backend/gpu/` abstraction layer defines exactly 4 functions all GPU backends must implement. The CUDA backend (`mlx/backend/cuda/`) is the structural reference.

---

## Architecture

```
mlx/backend/gpu/eval.h          ← CONTRACT: 4 functions Vulkan must implement
mlx/backend/vulkan/             ← NEW: mirrors metal/ and cuda/ structure
    CMakeLists.txt
    device.h / device.cpp       ← VkInstance, VkDevice, VkQueue, CommandPool, PipelineCache
    allocator.h / allocator.cpp ← VMA-based allocator (mirrors MetalAllocator)
    eval.cpp                    ← implements gpu::new_stream/eval/finalize/synchronize
    event.h / event.cpp         ← VkSemaphore wrappers
    fence.cpp                   ← VkFence CPU-GPU sync
    device_info.cpp             ← Reports GPU name/VRAM/capabilities
    primitives.cpp              ← ~80 eval_gpu() dispatch implementations
    copy.cpp                    ← copy_gpu, fill_gpu, reshape_gpu (gpu/copy.h contract)
    slicing.cpp                 ← GPU slicing ops (gpu/slicing.h contract)
    utils.h / utils.cpp         ← shared helpers (type conversion, dispatch utils)
    kernels/                    ← GLSL compute shaders → compiled to .spv at build time
        copy.comp
        unary.comp
        binary.comp
        ternary.comp
        reduce.comp
        matmul.comp
        softmax.comp
        sort.comp
        scan.comp
        arange.comp
        conv.comp
        fft.comp
        indexing.comp           ← gather + scatter
        rope.comp
        normalization.comp      ← layer_norm, rms_norm
        quantized.comp
        logsumexp.comp
        arg_reduce.comp
        binary_two.comp
        random.comp
```

**Root CMakeLists.txt modification** — add:

```cmake
option(MLX_BUILD_VULKAN "Build Vulkan backend" OFF)
if(MLX_BUILD_VULKAN)
  find_package(Vulkan REQUIRED)
  find_package(VulkanMemoryAllocator REQUIRED)  # via vcpkg or FetchContent
  add_subdirectory(mlx/backend/vulkan)
endif()
```

---

## The gpu/ Contract (4 functions in `namespace mlx::core::gpu`)

```cpp
void new_stream(Stream stream);   // Create VkCommandPool for stream
void eval(array& arr);            // Record+submit compute dispatch for one primitive
void finalize(Stream s);          // vkQueueSubmit + begin new command buffer
void synchronize(Stream s);       // vkQueueWaitIdle (blocks CPU)
```

---

## Implementation Phases

### Phase 0: Repository Setup & Toolchain (prerequisite)

1. Clone MLX into working directory:

   ```bash
   git clone https://github.com/ml-explore/mlx.git /Users/ektasaini/Desktop/mlx-vulkan
   cd /Users/ektasaini/Desktop/mlx-vulkan
   ```

2. Install Vulkan toolchain on the build/target Linux machine:

   ```bash
   # Linux (Ubuntu/Debian)
   sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers \
     glslc glslang-tools spirv-tools libshaderc-dev

   # macOS dev environment (for iteration)
   brew install vulkan-headers vulkan-loader vulkan-tools shaderc spirv-tools \
     glslang molten-vk vulkan-validationlayers
   ```

3. Verify Vulkan GPU visibility:
   ```bash
   export VK_ICD_FILENAMES=/path/to/vendor_icd.json   # Linux: auto
   vulkaninfo --summary
   ```

---

### Phase 1: Build System Scaffolding

**File: `mlx/backend/vulkan/CMakeLists.txt`**

```cmake
# AOT SPIR-V compilation: for each .comp file, run glslc at configure time
function(compile_shader SHADER_FILE)
  get_filename_component(NAME ${SHADER_FILE} NAME_WE)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels/${NAME}.spv
    COMMAND glslc --target-env=vulkan1.2
                  -fshader-stage=compute
                  -o ${CMAKE_CURRENT_BINARY_DIR}/kernels/${NAME}.spv
                  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${SHADER_FILE}
    DEPENDS kernels/${SHADER_FILE})
  list(APPEND SPIRV_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/kernels/${NAME}.spv)
  set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} PARENT_SCOPE)
endfunction()

compile_shader(copy.comp)
compile_shader(unary.comp)
compile_shader(binary.comp)
# ... all shaders ...

add_custom_target(vulkan_shaders DEPENDS ${SPIRV_OUTPUTS})

target_sources(mlx PRIVATE
  allocator.cpp copy.cpp device.cpp device_info.cpp
  eval.cpp event.cpp fence.cpp primitives.cpp slicing.cpp utils.cpp)

target_link_libraries(mlx PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)
add_dependencies(mlx vulkan_shaders)
target_compile_definitions(mlx PRIVATE
  VULKAN_KERNELS_PATH="${CMAKE_CURRENT_BINARY_DIR}/kernels/")
```

---

### Phase 2: Device & Memory Infrastructure

**`device.h` key types:**

```cpp
namespace mlx::core::vulkan {
  struct VulkanDevice {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family;
    VkPipelineCache pipeline_cache;

    void new_queue(int index);                              // creates per-stream CommandPool
    VkCommandBuffer get_command_buffer(int index);
    VkPipeline get_pipeline(const std::string& name);      // cached pipeline lookup
    void submit(int stream_index);
    void wait_idle(int stream_index);
  };

  VulkanDevice& device(mlx::core::Device dev);
}
```

**`allocator.h` key types (mirrors MetalAllocator):**

```cpp
class VulkanAllocator : public Allocator {
  // Uses VMA (VulkanMemoryAllocator) for device-local memory
  // Tracks active_memory_, peak_memory_, memory_limit_
  // BufferCache (LRU) to reuse VkBuffer+VmaAllocation pairs
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;
};
```

Key difference from Metal: Vulkan does NOT have unified memory on discrete GPUs.
Must handle: `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for compute, `HOST_VISIBLE | HOST_COHERENT` for staging.
Use VMA's `VMA_MEMORY_USAGE_AUTO` with `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` flags.

---

### Phase 3: Eval Dispatch (eval.cpp)

Pattern mirroring `metal/eval.cpp`:

```cpp
void eval(array& arr) {
  auto s = arr.primitive().stream();
  auto& d = vulkan::device(s.device);
  auto cmd = d.get_command_buffer(s.index);

  vkBeginCommandBuffer(cmd, &begin_info);
  arr.primitive().eval_gpu(arr.inputs(), arr.outputs());

  if (d.should_submit(s.index)) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(d.compute_queue, 1, &submit, fence);
    scheduler::notify_new_task(s);
    // fence completion handler → scheduler::notify_task_completion(s)
    d.get_command_buffer(s.index);  // begin next
  }
}
```

---

### Phase 4: GLSL Kernel Conventions

Each `.comp` shader follows this template:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input/output buffers via descriptor sets
layout(set = 0, binding = 0) readonly buffer InBuf  { float data[]; } in_buf;
layout(set = 0, binding = 1) writeonly buffer OutBuf { float data[]; } out_buf;

// Constants via push constants (replaces Metal set_bytes)
layout(push_constant) uniform Params {
  uint size;
  // op-specific params
} params;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= params.size) return;
  out_buf.data[idx] = /* operation */ in_buf.data[idx];
}
```

**Critical Vulkan compute conventions:**

- Workgroup size: 256 threads default (tunable per kernel)
- Dispatch: `vkCmdDispatch(cmd, ceil(n/256), 1, 1)`
- Type support: use `GL_EXT_shader_explicit_arithmetic_types` for float16/bfloat16
- bfloat16: Vulkan has no native bf16 — use uint16 storage + manual conversion (same as Metal bf16.h)
- Push constants max 128 bytes — for large metadata use UBO (uniform buffer)
- Specialization constants for compile-time kernel variants (dtype, reduction type, etc.)

**Priority kernel implementation order:**

1. `copy.comp` — most used, needed by almost every op
2. `unary.comp` — relu, exp, log, etc. (simple, validates pipeline)
3. `binary.comp` — add, mul, etc.
4. `reduce.comp` — sum, max (needed for softmax, loss)
5. `matmul.comp` — tiled GEMM (most performance-critical)
6. `softmax.comp`
7. `arange.comp`
8. Remaining ops in alphabetical order

---

### Phase 5: Primitives Dispatch (primitives.cpp)

Pattern for each primitive (mirrors `metal/primitives.cpp`):

```cpp
void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = vulkan::device(s.device);
  auto cmd = d.get_command_buffer(s.index);

  auto pipeline = d.get_pipeline("binary_add_float");  // cached VkPipeline

  // Bind descriptor set (input/output buffers)
  VkDescriptorSet ds = d.allocate_descriptor_set(pipeline_layout);
  bind_buffer(ds, 0, inputs[0]);
  bind_buffer(ds, 1, inputs[1]);
  bind_buffer(ds, 2, out);

  // Push constants
  BinaryParams params{(uint32_t)out.size()};
  vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, div_ceil(out.size(), 256), 1, 1);

  // Memory barrier before next op reads this output
  insert_buffer_barrier(cmd, out);
}
```

**Primitive coverage required** (full list from `no_gpu/primitives.cpp`):

- Elementwise (unary): Abs, Cos, Erf, Exp, Log, Neg, Relu, Sigmoid, Sign, Sin, Sqrt, ...
- Elementwise (binary): Add, Sub, Mul, Div, Maximum, Minimum, Power, ...
- Reduction: ArgReduce, Reduce (sum/max/min/prod over axes)
- Matrix: AddMM, Matmul, Gemv
- Indexing: Gather, Scatter, ScatterAxis, ...
- Shape: Arange, AsType, Copy, Pad, Slice, Transpose, ...
- Normalization: LayerNorm, RMSNorm
- Attention: ScaledDotProductAttention
- Misc: FFT, Scan, Sort, Hadamard, Quantized ops, ...

---

## Critical Files to Modify

| File                               | Change                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `CMakeLists.txt`                   | Add `MLX_BUILD_VULKAN` option + `find_package(Vulkan)` + `add_subdirectory` |
| `mlx/backend/gpu/eval.h`           | No change — this is the interface contract                                  |
| All files in `mlx/backend/vulkan/` | New (see above)                                                             |

## Reference Implementations

| Reference                                                                | What to use it for                                                                  |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `mlx/backend/cuda/` (local)                                              | File-for-file structural template (device.h, allocator.h, eval.cpp, primitives.cpp) |
| `mlx/backend/gpu/eval.h` (local)                                         | The exact 4-function interface to implement                                         |
| [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)         | SPIR-V loading patterns, Vulkan initialization boilerplate                          |
| [Shaderc](https://github.com/google/shaderc)                             | glslc AOT SPIR-V compilation in CMake + optional runtime compilation                |
| [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)               | Reflection for descriptor set layout auto-generation                                |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Device memory allocation (replaces MTL::Heap + BufferCache)                         |

---

## Verification

```bash
# 1. Build with Vulkan backend
cmake -B build -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON
cmake --build build -j$(nproc)

# 2. Verify GPU detection
vulkaninfo --summary

# 3. Smoke test — simple forward pass
python -c "
import mlx.core as mx
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
print(mx.add(a, b))  # expected: [5, 7, 9]
print(mx.matmul(mx.ones((4,4)), mx.ones((4,4))))  # expected: 4s
"

# 4. Run MLX test suite
python -m pytest tests/ -x -v --tb=short

# 5. Numerical equivalence check (vs CPU)
python -c "
import mlx.core as mx
import numpy as np
a = mx.random.normal((64, 64))
cpu_result = mx.array(np.array(a))
gpu_result = mx.matmul(a, a.T, stream=mx.gpu)
mx.eval(gpu_result)
assert np.allclose(np.array(cpu_result @ cpu_result.T), np.array(gpu_result), atol=1e-4)
print('Numerical equivalence: PASS')
"
```

---

## Key Technical Decisions

1. **No MoltenVK on macOS for now** — development targeting Linux native Vulkan. MoltenVK can be added later with a `VK_ICD_FILENAMES` env var for macOS CI.

2. **VMA over raw VkDeviceMemory** — avoids suballocation complexity. Mirrors how Metal's `BufferCache` handles memory pooling.

3. **AOT SPIR-V, not JIT** — eliminates runtime shaderc dependency, faster startup, easier distribution. JIT via shaderc C API can be added as `MLX_VULKAN_JIT` option later.

4. **bfloat16 via uint16 storage** — same approach as `mlx/backend/metal/kernels/bf16.h`. Use `GL_EXT_shader_explicit_arithmetic_types_float16` for fp16; bf16 needs manual pack/unpack.

5. **Specialization constants for kernel variants** — instead of runtime template strings (Metal JIT), use `VkSpecializationInfo` to parameterize kernels by dtype and reduction type at pipeline creation time.

---

## External Resources

### Awesome GPU Engineering
**URL**: https://github.com/goabiaryan/awesome-gpu-engineering

A curated list of GPU engineering resources. Vulkan coverage is limited — one entry under "GPU Programming Frameworks" and a brief mention of **Kompute** (higher-level Vulkan compute abstraction). No dedicated Vulkan compute tutorials, SPIR-V guides, or ML kernel resources.

**Relevant to this project**:

| Topic | Coverage | Notes |
|---|---|---|
| Vulkan compute | ⚠️ Minimal | Single entry, no tutorials |
| Kompute (Vulkan abstraction) | ✅ Listed | Could be useful reference for descriptor management patterns |
| CUDA/ROCm kernels | ✅ Extensive | Port patterns applicable to GLSL equivalents |
| GPU memory management | ✅ General | Concepts transfer to VMA usage |

**Verdict**: Useful as a general GPU engineering reading list. Not Vulkan-specific enough to directly inform this backend implementation. Bookmark for general GPU compute background.

### Other Key External References (already in PLAN.md)

| Resource | Purpose |
|---|---|
| [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) | SPIR-V loading, Vulkan init boilerplate |
| [Shaderc](https://github.com/google/shaderc) | AOT SPIR-V compilation |
| [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Device memory allocator |
| [ARM AI-ML Emulation Layer](https://github.com/arm/ai-ml-emulation-layer-for-vulkan) | Assessed; VK extension layer only — not a linkable library |
| [MLX issue #1751](https://github.com/ml-explore/mlx/issues/1751) | Original community request |
