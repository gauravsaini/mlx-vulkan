// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU copy operations

#include "mlx/backend/gpu/copy.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <vector>

namespace mlx::core {

static void dispatch_copy_shader(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
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
  // copy.comp now has 3 bindings and a 40-byte push constant (added
  // src_dtype/dst_dtype)
  VkPipeline pipeline = dev.get_pipeline("copy", layout, ds_layout, 3, 40);
  if (pipeline == VK_NULL_HANDLE)
    return;

  // Set up meta buffer for multidimensional strides if required
  VkBuffer meta_buf = VK_NULL_HANDLE;
  std::vector<int32_t> meta_data;
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    int ndim = data_shape.size();
    meta_data.resize(ndim * 3);
    for (int i = 0; i < ndim; ++i) {
      meta_data[i] = data_shape[i];
      meta_data[ndim + i] = i_strides[i];
      meta_data[2 * ndim + i] = o_strides[i];
    }
    size_t meta_bytes = meta_data.size() * sizeof(int32_t);
    // Allocate host-visible VMA staging buffer for the strides
    auto* staging = vulkan::allocator().alloc_staging(meta_bytes);
    std::memcpy(staging->mapped_ptr, meta_data.data(), meta_bytes);
    meta_buf = staging->buffer;

    // Dispose staging buffer intelligently when the execution encodes
    // completely
    vulkan::device(s.device).add_completed_handler(
        s, [staging]() { vulkan::allocator().free_staging(staging); });
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(s, ds_layout);

  // Bind src buffer (binding 0), dst buffer (binding 1), meta buffer (binding
  // 2)
  VkDescriptorBufferInfo src_info{src_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dst_info{dst_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo meta_info{
      meta_buf != VK_NULL_HANDLE ? meta_buf : src_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[3]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &src_info;

  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &dst_info;

  writes[2] = writes[0];
  writes[2].dstBinding = 2;
  writes[2].pBufferInfo = &meta_info;

  vkUpdateDescriptorSets(dev.vk_device(), 3, writes, 0, nullptr);

  // Dtype enum values matching copy.comp DTYPE_* constants
  // Mirrors mlx/dtype.h Dtype ordering
  auto dtype_to_enum = [](Dtype dt) -> uint32_t {
    switch (dt) {
      case bool_:
        return 0;
      case uint8:
        return 1;
      case uint16:
        return 2;
      case uint32:
        return 3;
      case uint64:
        return 4;
      case int8:
        return 5;
      case int16:
        return 6;
      case int32:
        return 7;
      case int64:
        return 8;
      case float16:
        return 9;
      case bfloat16:
        return 10;
      case float32:
        return 11;
      case float64:
        return 12;
      case complex64:
        return 13;
      default:
        return 11; // fallback float32
    }
  };

  struct PushConst {
    uint32_t n;
    uint32_t copy_type;
    uint32_t ndim;
    uint32_t src_stride0;
    uint32_t dst_stride0;
    uint32_t elem_size;
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t src_dtype;
    uint32_t dst_dtype;
  } pc;

  size_t n = 1;
  for (auto d : data_shape)
    n *= d;

  pc.n = static_cast<uint32_t>(n);
  pc.copy_type = static_cast<uint32_t>(ctype);
  pc.ndim = static_cast<uint32_t>(data_shape.size());
  pc.src_stride0 = i_strides.empty() ? 1 : static_cast<uint32_t>(i_strides[0]);
  pc.dst_stride0 = o_strides.empty() ? 1 : static_cast<uint32_t>(o_strides[0]);
  pc.elem_size = vulkan::dtype_size(in.dtype()); // source element size

  // Vulkan buffers do not natively encode the array offset, so we must add it
  // here! array::offset() is strictly evaluated in bytes, while copy.comp
  // offsets are in elements.
  uint32_t in_base_elem_offset = in.offset() / in.itemsize();
  uint32_t out_base_elem_offset = out.offset() / out.itemsize();

  pc.src_offset = static_cast<uint32_t>(i_offset + in_base_elem_offset);
  pc.dst_offset = static_cast<uint32_t>(o_offset + out_base_elem_offset);

  pc.src_dtype = dtype_to_enum(in.dtype());
  pc.dst_dtype = dtype_to_enum(out.dtype());

  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t groups = vulkan::div_ceil(n, vulkan::WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups, 1, 1);

  // Memory barrier
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(
      in, out, ctype, [&](size_t n) { return allocator::malloc(n); });
  if (donated && in.dtype() == out.dtype()) {
    return; // Same type, buffer donated - nothing to copy
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
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
  if (out.size() == 0)
    return;
  dispatch_copy_shader(
      in, out, data_shape, i_strides, o_strides, i_offset, o_offset, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  // Scalar broadcast: data_shape must be out.shape() so n = out.size() threads
  // are dispatched. The 4-arg overload uses in.shape() which is {} for a
  // scalar, causing only 1 thread to run and leaving remaining elements at
  // zero.
  copy_gpu_inplace(
      val, out, out.shape(), {}, out.strides(), 0, 0, CopyType::Scalar, s);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

} // namespace mlx::core
