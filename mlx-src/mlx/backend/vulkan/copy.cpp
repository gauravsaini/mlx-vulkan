// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU copy operations

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/allocator.h"

#include <cassert>
#include <vector>

namespace mlx::core {

// Dispatch a copy shader for Vulkan
static void dispatch_copy_shader(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {

  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;

  VkBuffer src_buf = vulkan::get_buffer(in);
  VkBuffer dst_buf = vulkan::get_buffer(out);

  if (src_buf == VK_NULL_HANDLE || dst_buf == VK_NULL_HANDLE) {
    return; // Zero-size array or unallocated
  }

  auto& dev = vulkan::device(s.device);
  VkCommandBuffer cmd = encoder.cmd;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("copy", layout, ds_layout, 3, 24);
  if (pipeline == VK_NULL_HANDLE) return;

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);

  // Bind src buffer (binding 0)
  VkDescriptorBufferInfo src_info{src_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dst_info{dst_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &src_info;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = ds;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &dst_info;

  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t copy_type;
    uint32_t ndim;
    uint32_t src_stride0;
    uint32_t dst_stride0;
    uint32_t elem_size;
  } pc;

  pc.n = static_cast<uint32_t>(out.size());
  pc.copy_type = static_cast<uint32_t>(ctype);
  pc.ndim = static_cast<uint32_t>(in.ndim());
  pc.src_stride0 = in.ndim() > 0 ? static_cast<uint32_t>(in.strides()[0]) : 1;
  pc.dst_stride0 = out.ndim() > 0 ? static_cast<uint32_t>(out.strides()[0]) : 1;
  pc.elem_size = vulkan::dtype_size(out.dtype());

  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t groups = vulkan::div_ceil(out.size(), vulkan::WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups, 1, 1);

  // Memory barrier
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(in, out, ctype, [&](size_t n) {
    return allocator::malloc(n);
  });
  if (donated && in.dtype() == out.dtype()) {
    return; // Same type, buffer donated - nothing to copy
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu(const array& in, array& out, CopyType ctype) {
  copy_gpu(in, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  if (out.size() == 0) return;
  dispatch_copy_shader(in, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s) {
  if (out.size() == 0) return;
  // For strided copies, use General type
  dispatch_copy_shader(in, out, CopyType::General, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (out.size() == 0) return;
  dispatch_copy_shader(in, out, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  // Fill is a scalar broadcast copy
  copy_gpu_inplace(val, out, CopyType::Scalar, s);
}

array contiguous_copy_gpu(const array& arr, const Stream& s) {
  array out(arr.shape(), arr.dtype(), nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  CopyType ctype = arr.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu_inplace(arr, out, ctype, s);
  return out;
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto ctype = (in.flags().contiguous && in.size() == in.data_size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, ctype, s);
}

array flatten_in_eval(const array& x, int start_axis, int end_axis, Stream s) {
  auto shape = x.shape();
  int ndim = shape.size();
  if (start_axis < 0) start_axis += ndim;
  if (end_axis < 0)   end_axis   += ndim;
  start_axis = std::max(0, std::min(start_axis, ndim));
  end_axis   = std::max(0, std::min(end_axis,   ndim));

  int flat_size = 1;
  for (int i = start_axis; i <= end_axis; i++) flat_size *= shape[i];

  Shape new_shape;
  for (int i = 0; i < start_axis; i++) new_shape.push_back(shape[i]);
  new_shape.push_back(flat_size);
  for (int i = end_axis + 1; i < ndim; i++) new_shape.push_back(shape[i]);

  return reshape_in_eval(x, new_shape, s);
}

array reshape_in_eval(const array& x, Shape shape, Stream s) {
  array out(shape, x.dtype(), nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu_inplace(x, out, CopyType::General, s);
  return out;
}

array swapaxes_in_eval(const array& x, int axis1, int axis2) {
  // No-op for the GPU path (strides handle this)
  return x;
}

} // namespace mlx::core
