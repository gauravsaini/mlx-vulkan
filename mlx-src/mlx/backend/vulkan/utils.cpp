// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/utils.h"

#include <stdexcept>
#include <string>

namespace mlx::core::vulkan {

VkFormat to_vk_format(Dtype dtype) {
  switch (dtype) {
    case float32: return VK_FORMAT_R32_SFLOAT;
    case float16: return VK_FORMAT_R16_SFLOAT;
    case bfloat16: return VK_FORMAT_R16_UINT; // stored as uint16
    case int32:   return VK_FORMAT_R32_SINT;
    case uint32:  return VK_FORMAT_R32_UINT;
    case int16:   return VK_FORMAT_R16_SINT;
    case uint16:  return VK_FORMAT_R16_UINT;
    case int8:    return VK_FORMAT_R8_SINT;
    case uint8:   return VK_FORMAT_R8_UINT;
    case bool_:   return VK_FORMAT_R8_UINT;
    case int64:   return VK_FORMAT_R64_SINT;
    case uint64:  return VK_FORMAT_R64_UINT;
    case complex64: return VK_FORMAT_R32G32_SFLOAT;
    default:      return VK_FORMAT_UNDEFINED;
  }
}

uint32_t dtype_size(Dtype dtype) {
  switch (dtype) {
    case float32:   return 4;
    case float16:   return 2;
    case bfloat16:  return 2;
    case int32:     return 4;
    case uint32:    return 4;
    case int16:     return 2;
    case uint16:    return 2;
    case int8:      return 1;
    case uint8:     return 1;
    case bool_:     return 1;
    case int64:     return 8;
    case uint64:    return 8;
    case complex64: return 8;
    default:        return 4;
  }
}

const char* dtype_name(Dtype dtype) {
  switch (dtype) {
    case float32:   return "float32";
    case float16:   return "float16";
    case bfloat16:  return "bfloat16";
    case int32:     return "int32";
    case uint32:    return "uint32";
    case int16:     return "int16";
    case uint16:    return "uint16";
    case int8:      return "int8";
    case uint8:     return "uint8";
    case bool_:     return "bool";
    case int64:     return "int64";
    case uint64:    return "uint64";
    case complex64: return "complex64";
    default:        return "unknown";
  }
}

void insert_buffer_barrier(VkCommandBuffer cmd, VkBuffer buffer) {
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer;
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void insert_buffer_barrier(VkCommandBuffer cmd, const array& arr) {
  if (arr.data<void>() == nullptr) return;
  // We use the raw pointer as a placeholder - actual VkBuffer obtained via device
  // This is called from primitives which have access to device
}

void vk_check(VkResult result, const char* msg) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        std::string("Vulkan error: ") + msg +
        " (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
  }
}

} // namespace mlx::core::vulkan
