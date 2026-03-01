# MLX Vulkan Backend — Project Setup & Execution Guide

## Repository Structure

```
mlx-vulkan/
├── mlx-src/                    # Main source tree (forked from ml-explore/mlx)
│   ├── mlx/backend/vulkan/     # ← All Vulkan backend code lives here
│   │   ├── device.cpp/h        # VkInstance, VkDevice, CommandEncoder, pipeline cache
│   │   ├── allocator.cpp/h     # VMA-backed memory allocator
│   │   ├── primitives.cpp      # eval_gpu() for all ~80 ops
│   │   ├── eval.cpp            # gpu::new_stream, eval, finalize, synchronize
│   │   ├── event.cpp/h         # Timeline semaphore Event
│   │   ├── fence.cpp           # CPU-GPU sync Fence
│   │   ├── device_info.cpp     # gpu::device_info, gpu::is_available
│   │   ├── fft.cpp             # FFT dispatch logic
│   │   ├── copy.cpp            # copy_gpu, fill_gpu, reshape_gpu
│   │   ├── slicing.cpp         # slice_gpu
│   │   └── kernels/            # GLSL compute shaders (*.comp → *.spv at build time)
│   ├── setup.py                # Build entry point (CMakeExtension via setuptools)
│   └── python/mlx/             # Python package output dir
│       └── core.cpython-311-darwin.so  ← The importable extension
├── tests/vulkan/               # Custom stage test suite (test_stageNN_*.py)
├── PLAN.md                     # Phase tracker and task checklist
├── TIMELINE.md                 # Chronological change log (update after every major commit)
├── CLAUDE.md                   # Protocol rules for Claude (TIMELINE.md protocol, pipeline cache rule)
└── PROJECT.md                  # This file
```

---

## Build System

### How the Build Works

The project uses `setup.py` (CMakeExtension via setuptools). CMake arguments are passed via
the `CMAKE_ARGS` environment variable. **There is no standalone `cmake -B build_vulkan` workflow** —
the PLAN.md docs referencing `build_vulkan` are outdated.

The real CMake build directory is:
```
mlx-src/build/temp.macosx-11.0-arm64-cpython-311/mlx.core/
```

### Build Command (Full Rebuild)

```bash
cd /Users/ektasaini/Desktop/mlx-vulkan/mlx-src
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON" \
  python setup.py build_ext --inplace
```

This:
1. Runs CMake configure inside `build/temp.macosx-11.0-arm64-cpython-311/mlx.core/`
2. Compiles all `.cpp` sources and all `.comp` shaders → `.spv`
3. Copies output `.so` to `python/mlx/core.cpython-311-darwin.so` automatically

### Incremental C++ Rebuild (after `.cpp` changes only — fastest)

```bash
BUILDDIR=/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build/temp.macosx-11.0-arm64-cpython-311/mlx.core

cmake --build "$BUILDDIR" --target install -j$(sysctl -n hw.ncpu)

# Copy the freshly linked .so to the importable location
cp /Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build/lib.macosx-11.0-arm64-cpython-311/mlx/core.cpython-311-darwin.so \
   /Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python/mlx/core.cpython-311-darwin.so
```

> ⚠️ `setup.py build_ext --inplace` does **not** reliably detect changed `.cpp` files on subsequent runs.
> Always use the direct `cmake --build` command above for incremental rebuilds instead.

### Shader-Only Rebuild (after `.comp` changes)

```bash
# Recompile a single shader
glslc --target-env=vulkan1.2 \
  mlx/backend/vulkan/kernels/FOO.comp \
  -o build/temp.macosx-11.0-arm64-cpython-311/mlx.core/mlx/backend/vulkan/kernels/FOO.spv

# Or just do a full rebuild (rebuilds changed shaders automatically)
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON" \
  python setup.py build_ext --inplace
```

> ⚠️ **Critical gotcha**: `setup.py build_ext` does **NOT** automatically detect `.comp` shader changes.
> The build system only tracks `.cpp` file changes. If you edit a shader but not its corresponding `.cpp`,
> you must trigger a rebuild manually with `glslc` or by touching the `.cpp` file.

### Python Extension Location

```
# The importable .so:
python/mlx/core.cpython-311-darwin.so

# Set PYTHONPATH to use it:
export PYTHONPATH=/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python
```

---

## Running Tests

### Custom Vulkan Stage Tests

```bash
cd /Users/ektasaini/Desktop/mlx-vulkan/mlx-src

# Run a single stage
PYTHONPATH=python timeout 30 python3 ../tests/vulkan/test_stageNN_name.py

# Run all stages (summary view)
for f in ../tests/vulkan/test_stage*.py; do
  name=$(basename $f)
  result=$(PYTHONPATH=python timeout 30 python3 $f 2>&1 | grep "Results:")
  echo "$name: $result"
done
```

### MLX Python Test Suite (upstream tests)

```bash
cd /Users/ektasaini/Desktop/mlx-vulkan/mlx-src
PYTHONPATH=python python -m pytest python/tests/test_ops.py -k "unary" -x -v
PYTHONPATH=python python -m pytest python/tests/test_array.py -x -v
```

### Quick Smoke Test

```bash
cd /Users/ektasaini/Desktop/mlx-vulkan/mlx-src
PYTHONPATH=python python3 -c "
import mlx.core as mx
print('GPU:', mx.default_device())
a = mx.ones((4,)) + mx.full((4,), 3.0)
mx.eval(a)
print('add:', a.tolist())  # should be [4.0, 4.0, 4.0, 4.0]
info = mx.metal.device_info()
print('device:', info.get('device_name'))
print('arch:  ', info.get('architecture'))
"
```

---

## Critical Rules & Gotchas

### Pipeline Cache Versioning

**Always bump `kPipelineCacheVersion`** in `mlx-src/mlx/backend/vulkan/device.cpp`
whenever **any shader push constant layout changes**. Stale cache causes MoltenVK to
`SIGKILL` the process at import time.

```cpp
// device.cpp
static constexpr uint32_t kPipelineCacheVersion = 9; // bump on every push-constant change
```

Cache lives at: `~/.cache/mlx_vulkan_pipeline_cache_vN.bin`

### Indexing Push Constants

ALL indexing ops (`Gather`, `GatherAxis`, `ScatterAxis`) **must** use `kIndexPushSize = 44`
(11-field `IndexPushConst`). Never call `get_pipeline("indexing", ...)` with a different size.

### Apple Silicon (MoltenVK) Specifics

- **Denormal flush-to-zero**: Apple Silicon GPUs flush denormal floats to zero.
  Integer values reinterpreted as float bits (`uintBitsToFloat`) get silently zeroed.
  All shaders must use dtype-aware paths — never pass integer data through float ops.
- **MoltenVK stale pipeline cache**: changing push constant sizes without bumping the
  cache version causes SIGKILL at `dlopen` time (not at pipeline creation).
- **Two dispatches in one eval_gpu deadlock**: Dispatching two compute shaders within
  a single `eval_gpu` call (e.g. copy + binary) causes a GPU hang on MoltenVK.
  Always use single-dispatch solutions (e.g. stride-based in-shader broadcasting).
- **Vendor ID**: Apple Silicon via MoltenVK reports `vendorID = 0x106B`.

### Two-Pass Debug Workflow

When a shader change doesn't take effect:
1. Check if the old `.spv` is stale (check modification time vs `.comp`)
2. Run `glslc` manually on the `.comp` file
3. Check pipeline cache version — bump if push constant layout changed

---

## Completed Stages (as of 2026-03-01)

| Stage | Test File | Status |
|-------|-----------|--------|
| 13 | test_stage13_indexing.py | ✅ 7/7 |
| 14 | test_stage14_sort.py | ✅ 6/6 |
| 15 | test_stage15_scan.py | ✅ 5/5 |
| 16 | test_stage16_nn_extended.py | ✅ 8/8 |
| 17 | test_stage17_fft.py | ✅ 3/3 |
| 17b | test_stage17_addmm_conv_rbits.py | ✅ |
| 18 | test_stage18_concat.py | ✅ 3/3 |
| 19 | test_stage19_quantized.py | — |
| 20 | test_stage20_linalg.py | ✅ 4/4 |
| 21 | test_stage21_advanced_mm.py | ✅ 7/7 |
| 22 | test_stage22_sync.py | ✅ Phase 4 |

---

## Key File Locations

| What | Path |
|------|------|
| Vulkan backend source | `mlx-src/mlx/backend/vulkan/` |
| GLSL shaders | `mlx-src/mlx/backend/vulkan/kernels/*.comp` |
| Compiled SPV shaders | `mlx-src/build/temp.macosx-11.0-arm64-cpython-311/mlx.core/mlx/backend/vulkan/kernels/*.spv` |
| Python extension | `mlx-src/python/mlx/core.cpython-311-darwin.so` |
| Pipeline cache | `~/.cache/mlx_vulkan_pipeline_cache_vN.bin` |
| Stage tests | `tests/vulkan/test_stageNN_*.py` |
| Upstream Python tests | `mlx-src/python/tests/` |
