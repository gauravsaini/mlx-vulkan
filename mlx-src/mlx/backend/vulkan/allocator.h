// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - VMA-based GPU memory allocator

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <mutex>
#include <unordered_map>

#include "mlx/allocator.h"

namespace mlx::core {
class array;
namespace vulkan {

// Internal buffer tracking struct
struct VulkanBuffer {
  VkBuffer buffer{VK_NULL_HANDLE};
  VmaAllocation allocation{VK_NULL_HANDLE};
  size_t size{0};
  void* mapped_ptr{nullptr}; // non-null only for staging/host-visible buffers
};

class VulkanAllocator : public allocator::Allocator {
 public:
  allocator::Buffer malloc(size_t size) override;
  void free(allocator::Buffer buffer) override;
  size_t size(allocator::Buffer buffer) const override;
  allocator::Buffer make_buffer(void* ptr, size_t size) override;
  void release(allocator::Buffer buffer) override;

  // Host-visible staging buffer (for CPU<->GPU transfers)
  VulkanBuffer* alloc_staging(size_t size);
  void free_staging(VulkanBuffer* buf);

  // Memory statistics
  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit() const;
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  void clear_cache();

  VkBuffer vk_buffer(allocator::Buffer buf) const;

 private:
  VulkanAllocator();
  friend VulkanAllocator& allocator();

  mutable std::mutex mutex_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t memory_limit_{1ULL << 34}; // 16 GB default
};

VulkanAllocator& allocator();

// Get VkBuffer from an mlx array
VkBuffer get_buffer(const array& arr);

} // namespace vulkan
} // namespace mlx::core
