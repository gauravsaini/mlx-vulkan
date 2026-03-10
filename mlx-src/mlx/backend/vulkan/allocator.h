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
struct Stream;
namespace vulkan {

// Internal buffer tracking struct
struct VulkanBuffer {
  static constexpr uint64_t kMagic = 0x4d4c58564b425546ULL; // "MLXVKBUF"
  uint64_t magic{kMagic};
  VkBuffer buffer{VK_NULL_HANDLE};
  VmaAllocation allocation{VK_NULL_HANDLE};
  size_t size{0};
  void* mapped_ptr{nullptr}; // VMA-mapped pointer for host-visible allocations
  void* cpu_readback_ptr{nullptr}; // heap snapshot for non-host-visible readback
  bool owns_vma_mapping{false};
  bool owns_allocation{true};
  bool submitted_to_gpu{false};
  VkMemoryPropertyFlags memory_properties{0};
};

class VulkanAllocator : public allocator::Allocator {
 public:
  allocator::Buffer malloc(size_t size) override;
  void free(allocator::Buffer buffer) override;
  size_t size(allocator::Buffer buffer) const override;
  allocator::Buffer make_buffer(void* ptr, size_t size) override;
  void release(allocator::Buffer buffer) override;
  void copy_from_host(
      allocator::Buffer buffer,
      const void* src,
      size_t size,
      size_t offset = 0) override;
  void copy_to_host(
      allocator::Buffer buffer,
      void* dst,
      size_t size,
      size_t offset = 0) override;

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

// Explicit host transfer helpers for Vulkan-backed arrays.
void copy_from_host(
    const array& arr,
    const void* src,
    size_t bytes,
    const Stream& s,
    size_t byte_offset = 0);
void copy_to_host(
    const array& arr,
    void* dst,
    size_t bytes,
    const Stream& s,
    size_t byte_offset = 0);

} // namespace vulkan
} // namespace mlx::core
