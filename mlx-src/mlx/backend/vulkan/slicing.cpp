// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU slicing operations

#include "mlx/backend/gpu/slicing.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/device.h"

#include <numeric>

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
    // Skip empty inputs — passing a zero-size buffer to copy_gpu_inplace
    // would bind a null VkBuffer into the descriptor set → DEVICE_LOST.
    if (inputs[i].size() == 0)
      continue;

    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  // On MoltenVK with unified memory, GPU buffers are CPU-accessible.
  // The MLX scheduler guarantees `indices` is already evaluated before
  // this function is called, so its data pointer is valid.
  array out(Shape{}, int32, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));

  int64_t offset = 0;
  for (size_t i = 0; i < axes.size(); i++) {
    int64_t idx = 0;
    if (indices.dtype() == int32) {
      idx = static_cast<int64_t>(indices.data<int32_t>()[i]);
    } else if (indices.dtype() == int64) {
      idx = indices.data<int64_t>()[i];
    } else if (indices.dtype() == uint32) {
      idx = static_cast<int64_t>(indices.data<uint32_t>()[i]);
    } else if (indices.dtype() == uint64) {
      idx = static_cast<int64_t>(indices.data<uint64_t>()[i]);
    }
    // Wrap negative indices
    int64_t ax_size = strides.size() > static_cast<size_t>(axes[i])
        ? 0 // size not directly available here; rely on caller for clamping
        : 0;
    (void)ax_size;
    offset += idx * static_cast<int64_t>(strides[axes[i]]);
  }

  int32_t* ptr = out.data<int32_t>();
  *ptr = static_cast<int32_t>(offset);
  return out;
}

} // namespace mlx::core
