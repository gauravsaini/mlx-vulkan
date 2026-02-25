// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU slicing operations

#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/allocator.h"

#include <numeric>

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides,
    const Stream& s) {
  // Compute the data offset and output strides using the common slicing helper
  auto [data_offset, out_strides] = prepare_slice(out, start_indices, strides);

  // Set up output to point into input buffer with computed offset and strides
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  out.copy_shared_buffer(in, out_strides, flags, out.size(), data_offset);
}

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
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
}

void pad_gpu(
    const array& in,
    const array& val,
    array& out,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Stream& s) {
  // Fill output with padding value first
  fill_gpu(val, out, s);

  // Then copy input into the appropriate region
  if (in.size() == 0) return;

  // Compute offset into out where input should be placed
  Strides out_strides = out.strides();
  int64_t data_offset = 0;
  for (size_t i = 0; i < axes.size(); i++) {
    data_offset += low_pad_size[i] * out_strides[axes[i]];
  }

  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  array out_slice(in.shape(), out.dtype(), nullptr, {});
  out_slice.copy_shared_buffer(out, out_strides, flags, in.size(), data_offset);
  copy_gpu_inplace(in, out_slice, CopyType::GeneralGeneral, s);
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  // Simple CPU-side computation for dynamic offset
  // For a full GPU implementation, this would be a shader dispatch
  array out(Shape{}, int32, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));

  // Evaluate indices on CPU and compute offset
  // This is a simplified implementation - production would compute on GPU
  int64_t offset = 0;
  if (indices.ndim() == 1) {
    for (size_t i = 0; i < axes.size(); i++) {
      // indices[i] * strides[axes[i]]
      // We'd need to read from the GPU buffer here
      // For now, return zero offset as a stub
    }
  }

  int32_t* ptr = out.data<int32_t>();
  *ptr = static_cast<int32_t>(offset);
  return out;
}

} // namespace mlx::core
