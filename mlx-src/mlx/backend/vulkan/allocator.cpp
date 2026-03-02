// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - VMA-based GPU memory allocator
//
// Pattern mirrors mlx/backend/cuda/allocator.cpp:
//   Buffer::ptr()    → opaque pointer to VulkanBuffer
//   Buffer::raw_ptr() → CPU-mapped pointer (staging path for discrete GPUs)
//   allocator::allocator() → returns the singleton VulkanAllocator

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"

#include <cassert>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace vulkan {

// ─────────────────────────────────────────────────────────────────────────────
// VulkanAllocator constructor
// ─────────────────────────────────────────────────────────────────────────────

VulkanAllocator::VulkanAllocator() {
  // Query total device-local memory for the default memory limit
  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(dev.vk_physical_device(), &mem_props);

  size_t total_device_mem = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total_device_mem += mem_props.memoryHeaps[i].size;
    }
  }
  // Default limit: 95% of device-local memory, capped at 16GB
  memory_limit_ = std::min(
      static_cast<size_t>(total_device_mem * 0.95), size_t{1ULL << 34});
}

// ─────────────────────────────────────────────────────────────────────────────
// malloc: Create a device-local VkBuffer via VMA
// ─────────────────────────────────────────────────────────────────────────────

allocator::Buffer VulkanAllocator::malloc(size_t size) {
  if (size == 0) {
    return allocator::Buffer{
        new VulkanBuffer{VK_NULL_HANDLE, VK_NULL_HANDLE, 0, nullptr}};
  }

  // Pad to 4 bytes so that shaders can read/write 32-bit uints safely
  size_t alloc_size = (size + 3) & ~3;

  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});

  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = alloc_size;
  buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  // Prefer device-local; for integrated GPUs VMA will pick host-visible
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT;
  // Ensure that memory maps are automatically flushed to GPU without manual
  // vmaFlushAllocation
  alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBuffer vk_buffer;
  VmaAllocation allocation;
  VkResult res = vmaCreateBuffer(
      dev.vma_allocator(),
      &buf_info,
      &alloc_info,
      &vk_buffer,
      &allocation,
      nullptr);

  if (res != VK_SUCCESS) {
    std::ostringstream msg;
    msg << "[vulkan::malloc] Unable to allocate " << size << " bytes.";
    throw std::runtime_error(msg.str());
  }

  // Check if the allocation ended up host-visible (integrated GPU)
  VmaAllocationInfo vma_info{};
  vmaGetAllocationInfo(dev.vma_allocator(), allocation, &vma_info);
  void* mapped = nullptr;
  VkMemoryPropertyFlags mem_flags;
  vmaGetAllocationMemoryProperties(dev.vma_allocator(), allocation, &mem_flags);
  if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    vmaMapMemory(dev.vma_allocator(), allocation, &mapped);
  }

  auto* vk_buf = new VulkanBuffer{vk_buffer, allocation, size, mapped};

  {
    std::lock_guard<std::mutex> lk(mutex_);
    active_memory_ += size;
    peak_memory_ = std::max(active_memory_, peak_memory_);
  }

  return allocator::Buffer{vk_buf};
}

// ─────────────────────────────────────────────────────────────────────────────
// free: Destroy a VkBuffer and its VMA allocation
// ─────────────────────────────────────────────────────────────────────────────

void VulkanAllocator::free(allocator::Buffer buffer) {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (!buf)
    return;
  if (buf->size == 0) {
    delete buf;
    return;
  }

  auto stream = Stream{0, mlx::core::Device{mlx::core::Device::gpu, 0}};
  auto& dev = device(stream.device);
  auto& encoder = get_command_encoder(stream);

  // We must defer destruction until the GPU has finished executing all commands
  // that might be referencing this buffer. Apple Metal achieves this implicitly
  // via pooled reference counting; for Vulkan, we push it to the completion
  // handler.
  vulkan::device(stream.device)
      .add_completed_handler(stream, [buf, dev = &dev]() {
        if (buf->mapped_ptr) {
          vmaUnmapMemory(dev->vma_allocator(), buf->allocation);
        }
        vmaDestroyBuffer(dev->vma_allocator(), buf->buffer, buf->allocation);

        {
          std::lock_guard<std::mutex> lk(allocator().mutex_);
          allocator().active_memory_ -= buf->size;
        }
        delete buf;
      });
}

// ─────────────────────────────────────────────────────────────────────────────
// make_buffer: Create a non-owning Buffer wrapping an existing pointer
// ─────────────────────────────────────────────────────────────────────────────

allocator::Buffer VulkanAllocator::make_buffer(void* ptr, size_t size) {
  // `ptr` is expected to be a `VulkanBuffer*` from a previous `malloc` call.
  // We create a new `VulkanBuffer` wrapper that shares the underlying Vulkan
  // resources but does NOT own them.
  auto* src_buf = static_cast<VulkanBuffer*>(ptr);
  if (!src_buf) {
    return allocator::Buffer{nullptr};
  }

  // Create a new wrapper pointing to the same Vulkan resources.
  // Crucially, when `release` is called on this new wrapper, we will ONLY
  // delete the wrapper (`new_buf`), not the underlying Vulkan memory.
  auto* new_buf = new VulkanBuffer{
      src_buf->buffer, src_buf->allocation, size, src_buf->mapped_ptr};

  return allocator::Buffer{new_buf};
}

// ─────────────────────────────────────────────────────────────────────────────
// release: Destroy the wrapper but NOT the underlying VMA allocation
// ─────────────────────────────────────────────────────────────────────────────

void VulkanAllocator::release(allocator::Buffer buffer) {
  // This is called for buffers created via `make_buffer`.
  // We only delete the `VulkanBuffer` tracking struct. We DO NOT call
  // `vmaDestroyBuffer` because this wrapper doesn't own the memory.
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (buf) {
    delete buf;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// size: Query size of an allocation
// ─────────────────────────────────────────────────────────────────────────────

size_t VulkanAllocator::size(allocator::Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (!buf)
    return 0;
  return buf->size;
}

// ─────────────────────────────────────────────────────────────────────────────
// Staging buffers (host-visible, for CPU↔GPU transfers on discrete GPUs)
// ─────────────────────────────────────────────────────────────────────────────

VulkanBuffer* VulkanAllocator::alloc_staging(size_t size) {
  if (size == 0)
    return nullptr;

  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});

  size_t alloc_size = (size + 3) & ~3;

  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = alloc_size;
  buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // needed for shader binding
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer vk_buffer;
  VmaAllocation allocation;
  VmaAllocationInfo vma_info{};
  VkResult res = vmaCreateBuffer(
      dev.vma_allocator(),
      &buf_info,
      &alloc_info,
      &vk_buffer,
      &allocation,
      &vma_info);

  if (res != VK_SUCCESS) {
    throw std::runtime_error(
        "[vulkan::alloc_staging] Unable to allocate staging buffer");
  }

  return new VulkanBuffer{vk_buffer, allocation, size, vma_info.pMappedData};
}

void VulkanAllocator::free_staging(VulkanBuffer* buf) {
  if (!buf)
    return;

  auto stream = Stream{0, mlx::core::Device{mlx::core::Device::gpu, 0}};
  auto& dev = device(stream.device);
  auto& encoder = dev.get_command_encoder(stream);

  vulkan::device(stream.device).add_completed_handler(stream, [buf]() {
    auto& d = device(mlx::core::Device{mlx::core::Device::gpu, 0});
    vmaDestroyBuffer(d.vma_allocator(), buf->buffer, buf->allocation);
    delete buf;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// VkBuffer helper: extract VkBuffer from an mlx array's data pointer
// ─────────────────────────────────────────────────────────────────────────────

VkBuffer VulkanAllocator::vk_buffer(allocator::Buffer buf) const {
  auto* vb = static_cast<VulkanBuffer*>(buf.ptr());
  if (!vb)
    return VK_NULL_HANDLE;
  return vb->buffer;
}

// Free function: get VkBuffer from array
VkBuffer get_buffer(const array& arr) {
  auto* vk_buf =
      static_cast<VulkanBuffer*>(const_cast<void*>(arr.buffer().ptr()));
  if (!vk_buf)
    return VK_NULL_HANDLE;
  return vk_buf->buffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory statistics
// ─────────────────────────────────────────────────────────────────────────────

size_t VulkanAllocator::get_active_memory() const {
  return active_memory_;
}

size_t VulkanAllocator::get_peak_memory() const {
  return peak_memory_;
}

void VulkanAllocator::reset_peak_memory() {
  std::lock_guard<std::mutex> lk(mutex_);
  peak_memory_ = active_memory_;
}

size_t VulkanAllocator::get_memory_limit() const {
  return memory_limit_;
}

size_t VulkanAllocator::set_memory_limit(size_t limit) {
  std::lock_guard<std::mutex> lk(mutex_);
  size_t old = memory_limit_;
  memory_limit_ = limit;
  return old;
}

size_t VulkanAllocator::get_cache_memory() const {
  // No buffer cache in initial implementation
  return 0;
}

void VulkanAllocator::clear_cache() {
  // No buffer cache to clear in initial implementation
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton accessor for VulkanAllocator
// ─────────────────────────────────────────────────────────────────────────────

VulkanAllocator& allocator() {
  static auto* alloc = []() {
    // Ensure scheduler is created before allocator (mirrors CUDA pattern)
    scheduler::scheduler();
    return new VulkanAllocator();
  }();
  return *alloc;
}

} // namespace vulkan

// ─────────────────────────────────────────────────────────────────────────────
// Global allocator:: namespace — bridge to the Vulkan implementation
// ─────────────────────────────────────────────────────────────────────────────

namespace allocator {

Allocator& allocator() {
  return vulkan::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_)
    return nullptr;
  auto* vk_buf = static_cast<vulkan::VulkanBuffer*>(ptr_);

  // Zero-sized buffers have no backing memory
  if (vk_buf->size == 0) {
    return nullptr;
  }

  if (vk_buf->mapped_ptr) {
    return vk_buf->mapped_ptr;
  }

  // For device-local without host-visible memory (discrete GPU),
  // we need a staging copy. Map via VMA with a forced mapping.
  auto& dev = vulkan::device(mlx::core::Device{mlx::core::Device::gpu, 0});
  void* mapped = nullptr;
  VkResult res = vmaMapMemory(dev.vma_allocator(), vk_buf->allocation, &mapped);
  if (res == VK_SUCCESS) {
    vk_buf->mapped_ptr = mapped;
    return mapped;
  }

  // Fallback: allocate staging, copy, and return
  // This is the slow path for discrete GPUs
  auto* staging = vulkan::allocator().alloc_staging(vk_buf->size);
  if (!staging || !staging->mapped_ptr) {
    if (staging)
      vulkan::allocator().free_staging(staging);
    return nullptr;
  }

  // Record a copy command
  auto& enc = dev.get_command_encoder(
      Stream{0, mlx::core::Device{mlx::core::Device::gpu, 0}});
  VkBufferCopy region{0, 0, vk_buf->size};
  vkCmdCopyBuffer(enc.cmd, vk_buf->buffer, staging->buffer, 1, &region);
  enc.op_count++;

  // Synchronize to ensure copy completes
  dev.synchronize(Stream{0, mlx::core::Device{mlx::core::Device::gpu, 0}});

  // Copy data out of staging to a persistent CPU allocation
  void* cpu_ptr = std::malloc(vk_buf->size);
  std::memcpy(cpu_ptr, staging->mapped_ptr, vk_buf->size);
  vulkan::allocator().free_staging(staging);

  // Note: this leaks cpu_ptr — in production, the caller should ensure
  // proper lifecycle. For correctness, store it back.
  vk_buf->mapped_ptr = cpu_ptr;
  return cpu_ptr;
}

} // namespace allocator

// ─────────────────────────────────────────────────────────────────────────────
// Global free functions required by mlx/memory.h
// ─────────────────────────────────────────────────────────────────────────────

size_t get_active_memory() {
  return vulkan::allocator().get_active_memory();
}

size_t get_peak_memory() {
  return vulkan::allocator().get_peak_memory();
}

void reset_peak_memory() {
  vulkan::allocator().reset_peak_memory();
}

size_t set_memory_limit(size_t limit) {
  return vulkan::allocator().set_memory_limit(limit);
}

size_t get_memory_limit() {
  return vulkan::allocator().get_memory_limit();
}

size_t get_cache_memory() {
  return vulkan::allocator().get_cache_memory();
}

size_t set_cache_limit(size_t limit) {
  // No buffer cache in initial Vulkan implementation
  (void)limit;
  return 0;
}

void clear_cache() {
  vulkan::allocator().clear_cache();
}

size_t set_wired_limit(size_t) {
  // Not applicable for Vulkan
  return 0;
}

} // namespace mlx::core
