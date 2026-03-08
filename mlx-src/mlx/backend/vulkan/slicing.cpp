// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU slicing operations

#include "mlx/backend/gpu/slicing.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
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
  // Materialize the small index vector on the CPU so this helper remains valid
  // on discrete GPUs where primary Vulkan allocations may not be host-visible.
  array out(Shape{}, int32, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));

  std::vector<int64_t> host_indices(axes.size(), 0);
  switch (indices.dtype()) {
    case int32: {
      std::vector<int32_t> tmp(axes.size(), 0);
      vulkan::copy_to_host(indices, tmp.data(), tmp.size() * sizeof(int32_t), s);
      for (size_t i = 0; i < tmp.size(); ++i)
        host_indices[i] = tmp[i];
      break;
    }
    case int64:
      vulkan::copy_to_host(
          indices, host_indices.data(), host_indices.size() * sizeof(int64_t), s);
      break;
    case uint32: {
      std::vector<uint32_t> tmp(axes.size(), 0);
      vulkan::copy_to_host(indices, tmp.data(), tmp.size() * sizeof(uint32_t), s);
      for (size_t i = 0; i < tmp.size(); ++i)
        host_indices[i] = static_cast<int64_t>(tmp[i]);
      break;
    }
    case uint64: {
      std::vector<uint64_t> tmp(axes.size(), 0);
      vulkan::copy_to_host(indices, tmp.data(), tmp.size() * sizeof(uint64_t), s);
      for (size_t i = 0; i < tmp.size(); ++i)
        host_indices[i] = static_cast<int64_t>(tmp[i]);
      break;
    }
    default:
      throw std::runtime_error(
          "[vulkan::compute_dynamic_offset] unsupported index dtype");
  }

  int64_t offset = 0;
  for (size_t i = 0; i < axes.size(); i++) {
    int64_t idx = host_indices[i];
    // Wrap negative indices
    int64_t ax_size = strides.size() > static_cast<size_t>(axes[i])
        ? 0 // size not directly available here; rely on caller for clamping
        : 0;
    (void)ax_size;
    offset += idx * static_cast<int64_t>(strides[axes[i]]);
  }

  int32_t offset32 = static_cast<int32_t>(offset);
  vulkan::copy_from_host(out, &offset32, sizeof(offset32), s);
  return out;
}

} // namespace mlx::core
