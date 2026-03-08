// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - VMA-based GPU memory allocator
//
// Pattern mirrors mlx/backend/cuda/allocator.cpp:
//   Buffer::ptr()    → opaque pointer to VulkanBuffer
//   Buffer::raw_ptr() → CPU-mapped pointer (staging path for discrete GPUs)
//   allocator::allocator() → returns the singleton VulkanAllocator

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace vulkan {

namespace {

VulkanBuffer* unwrap_buffer(const allocator::Buffer& buffer) {
  return static_cast<VulkanBuffer*>(const_cast<void*>(buffer.ptr()));
}

void copy_from_host_buffer(
    const allocator::Buffer& buffer,
    size_t base_offset,
    const void* src,
    size_t bytes,
    const Stream& s) {
  auto* vk_buf = unwrap_buffer(buffer);
  if (!vk_buf || bytes == 0) {
    return;
  }

  auto& dev = device(s.device);
  if (vk_buf->mapped_ptr) {
    std::memcpy(static_cast<char*>(vk_buf->mapped_ptr) + base_offset, src, bytes);
    if (!(vk_buf->memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      vmaFlushAllocation(
          dev.vma_allocator(), vk_buf->allocation, base_offset, bytes);
    }
    return;
  }

  auto* staging = allocator().alloc_staging(bytes);
  if (!staging || !staging->mapped_ptr) {
    throw std::runtime_error("[vulkan::copy_from_host] staging allocation failed");
  }
  std::memcpy(staging->mapped_ptr, src, bytes);

  auto& enc = get_command_encoder(s);
  VkBufferCopy region{
      0, static_cast<VkDeviceSize>(base_offset), static_cast<VkDeviceSize>(bytes)};
  vkCmdCopyBuffer(enc.cmd, staging->buffer, vk_buf->buffer, 1, &region);
  enc.op_count++;

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(
      enc.cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  device(s.device).add_completed_handler(
      s, [staging]() { allocator().free_staging(staging); });
}

void copy_to_host_buffer(
    const allocator::Buffer& buffer,
    size_t base_offset,
    void* dst,
    size_t bytes,
    const Stream& s) {
  auto* vk_buf = unwrap_buffer(buffer);
  if (!vk_buf || bytes == 0) {
    return;
  }

  auto& dev = device(s.device);
  if (vk_buf->mapped_ptr) {
    if (!(vk_buf->memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      vmaInvalidateAllocation(
          dev.vma_allocator(), vk_buf->allocation, base_offset, bytes);
    }
    std::memcpy(
        dst, static_cast<const char*>(vk_buf->mapped_ptr) + base_offset, bytes);
    return;
  }

  auto* staging = allocator().alloc_staging(bytes);
  if (!staging || !staging->mapped_ptr) {
    throw std::runtime_error("[vulkan::copy_to_host] staging allocation failed");
  }

  dev.synchronize(s);

  auto& enc = get_command_encoder(s);
  VkBufferCopy region{
      static_cast<VkDeviceSize>(base_offset), 0, static_cast<VkDeviceSize>(bytes)};
  vkCmdCopyBuffer(enc.cmd, vk_buf->buffer, staging->buffer, 1, &region);
  enc.op_count++;
  dev.synchronize(s);

  std::memcpy(dst, staging->mapped_ptr, bytes);
  vmaDestroyBuffer(dev.vma_allocator(), staging->buffer, staging->allocation);
  delete staging;
}

} // namespace

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
    return allocator::Buffer{new VulkanBuffer{
        VK_NULL_HANDLE, VK_NULL_HANDLE, 0, nullptr, nullptr, false, true, 0}};
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
  // Prefer device-local allocations and rely on explicit staging when direct
  // host access is unavailable. Integrated GPUs may still choose host-visible
  // memory, but discrete bring-up must not depend on it.
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT;

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

  auto* vk_buf = new VulkanBuffer{
      vk_buffer,
      allocation,
      size,
      mapped,
      nullptr,
      mapped != nullptr,
      true,
      mem_flags};

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
        if (buf->owns_vma_mapping && buf->mapped_ptr) {
          vmaUnmapMemory(dev->vma_allocator(), buf->allocation);
        }
        if (buf->cpu_readback_ptr) {
          std::free(buf->cpu_readback_ptr);
        }
        if (buf->owns_allocation) {
          vmaDestroyBuffer(dev->vma_allocator(), buf->buffer, buf->allocation);
        }

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
  (void)ptr;
  (void)size;
  // Vulkan does not yet implement host-memory import / true no-copy wrapping
  // for arbitrary CPU pointers. Force the generic array constructor to take
  // the safe copy path instead of misinterpreting a host pointer as a
  // VulkanBuffer handle.
  return allocator::Buffer{nullptr};
}

// ─────────────────────────────────────────────────────────────────────────────
// release: Destroy the wrapper but NOT the underlying VMA allocation
// ─────────────────────────────────────────────────────────────────────────────

void VulkanAllocator::release(allocator::Buffer buffer) {
  (void)buffer;
  // No-op until host-memory import / no-copy wrapping is implemented.
}

void VulkanAllocator::copy_from_host(
    allocator::Buffer buffer,
    const void* src,
    size_t bytes,
    size_t offset) {
  auto s = default_stream(mlx::core::Device::gpu);
  copy_from_host_buffer(buffer, offset, src, bytes, s);
  device(s.device).synchronize(s);
}

void VulkanAllocator::copy_to_host(
    allocator::Buffer buffer,
    void* dst,
    size_t bytes,
    size_t offset) {
  auto s = default_stream(mlx::core::Device::gpu);
  device(s.device).synchronize(s);
  copy_to_host_buffer(buffer, offset, dst, bytes, s);
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

  return new VulkanBuffer{
      vk_buffer,
      allocation,
      size,
      vma_info.pMappedData,
      nullptr,
      vma_info.pMappedData != nullptr,
      true,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
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

void copy_from_host(
    const array& arr,
    const void* src,
    size_t bytes,
    const Stream& s,
    size_t byte_offset) {
  copy_from_host_buffer(
      arr.buffer(), static_cast<size_t>(arr.offset()) + byte_offset, src, bytes, s);
}

void copy_to_host(
    const array& arr,
    void* dst,
    size_t bytes,
    const Stream& s,
    size_t byte_offset) {
  copy_to_host_buffer(
      arr.buffer(), static_cast<size_t>(arr.offset()) + byte_offset, dst, bytes, s);
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
// CPU fallback allocator for hosts where Vulkan is unavailable
//
// The Vulkan build still needs plain host allocations for CPU arrays and for
// environments where vkCreateInstance fails. Keep this local so the Vulkan
// backend can degrade cleanly without linking the no_gpu backend.
// ─────────────────────────────────────────────────────────────────────────────

namespace allocator {

#ifdef __APPLE__
#include "mlx/backend/no_gpu/apple_memory.h"
#elif defined(__linux__)
#include "mlx/backend/no_gpu/linux_memory.h"
#else
namespace {
size_t get_host_memory_size() {
  return 0;
}
} // namespace
#endif

namespace {

size_t cpu_memory_size() {
#if defined(__APPLE__) || defined(__linux__)
  return get_memory_size();
#else
  return get_host_memory_size();
#endif
}

struct CpuAllocHeader {
  static constexpr uint64_t kMagic = 0x4d4c5843564b414cULL; // "MLXCVKAL"
  uint64_t magic{kMagic};
  size_t size{0};
};

constexpr uint64_t kCpuAllocFooterMagic = 0x464f4f5445524d58ULL; // "FOOTERMX"

CpuAllocHeader* cpu_header(void* ptr) {
  return static_cast<CpuAllocHeader*>(ptr);
}

const CpuAllocHeader* cpu_header(const void* ptr) {
  return static_cast<const CpuAllocHeader*>(ptr);
}

class CommonAllocator : public Allocator {
 public:
  Buffer malloc(size_t size) override {
    void* ptr =
        std::malloc(size + sizeof(CpuAllocHeader) + sizeof(kCpuAllocFooterMagic));
    if (ptr != nullptr) {
      auto* header = cpu_header(ptr);
      header->magic = CpuAllocHeader::kMagic;
      header->size = size;
      auto* footer = reinterpret_cast<uint64_t*>(
          static_cast<char*>(ptr) + sizeof(CpuAllocHeader) + size);
      *footer = kCpuAllocFooterMagic;
    }
    std::unique_lock lk(mutex_);
    active_memory_ += size;
    peak_memory_ = std::max(active_memory_, peak_memory_);
    return Buffer{ptr};
  }

  void free(Buffer buffer) override {
    auto sz = size(buffer);
    std::unique_lock lk(mutex_);
    active_memory_ -= sz;
    retired_.push_back(buffer.ptr());
    retired_bytes_ += sz;
    while (retired_bytes_ > retired_limit_ && !retired_.empty()) {
      auto* ptr = retired_.front();
      retired_bytes_ -= cpu_header(ptr)->size;
      std::free(ptr);
      retired_.pop_front();
    }
  }

  size_t size(Buffer buffer) const override {
    if (buffer.ptr() == nullptr) {
      return 0;
    }
    auto* header = cpu_header(buffer.ptr());
    if (header->magic != CpuAllocHeader::kMagic) {
      std::fprintf(
          stderr,
          "[CommonAllocator] Corrupted CPU allocation header at %p\n",
          buffer.ptr());
      std::abort();
    }
    auto* footer = reinterpret_cast<const uint64_t*>(
        static_cast<const char*>(buffer.ptr()) + sizeof(CpuAllocHeader) +
        header->size);
    if (*footer != kCpuAllocFooterMagic) {
      std::fprintf(
          stderr,
          "[CommonAllocator] Corrupted CPU allocation footer at %p (size=%zu)\n",
          buffer.ptr(),
          header->size);
      std::abort();
    }
    return header->size;
  }

  size_t get_active_memory() const {
    return active_memory_;
  }

  size_t get_peak_memory() const {
    return peak_memory_;
  }

  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  }

  size_t get_memory_limit() const {
    return memory_limit_;
  }

  size_t set_memory_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(memory_limit_, limit);
    return limit;
  }

 private:
  CommonAllocator() : memory_limit_(0.8 * cpu_memory_size()) {
    if (memory_limit_ == 0) {
      memory_limit_ = 1UL << 33;
    }
  }

  friend CommonAllocator& common_allocator();

  size_t memory_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t retired_bytes_{0};
  const size_t retired_limit_{64ULL << 20};
  std::deque<void*> retired_;
  mutable std::mutex mutex_;
};

CommonAllocator& common_allocator() {
  static CommonAllocator allocator_;
  return allocator_;
}

bool use_cpu_allocator() {
  return !gpu::is_available();
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Global allocator:: namespace — bridge to the active implementation
// ─────────────────────────────────────────────────────────────────────────────

Allocator& allocator() {
  if (use_cpu_allocator()) {
    return common_allocator();
  }
  return vulkan::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_)
    return nullptr;

  if (use_cpu_allocator()) {
    return cpu_header(ptr_) + 1;
  }

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
    vk_buf->owns_vma_mapping = true;
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

  // Copy data out of staging into a reusable CPU snapshot buffer.
  if (!vk_buf->cpu_readback_ptr) {
    vk_buf->cpu_readback_ptr = std::malloc(vk_buf->size);
  }
  std::memcpy(vk_buf->cpu_readback_ptr, staging->mapped_ptr, vk_buf->size);
  vulkan::allocator().free_staging(staging);
  return vk_buf->cpu_readback_ptr;
}

} // namespace allocator

// ─────────────────────────────────────────────────────────────────────────────
// Global free functions required by mlx/memory.h
// ─────────────────────────────────────────────────────────────────────────────

size_t get_active_memory() {
  if (allocator::use_cpu_allocator()) {
    return allocator::common_allocator().get_active_memory();
  }
  return vulkan::allocator().get_active_memory();
}

size_t get_peak_memory() {
  if (allocator::use_cpu_allocator()) {
    return allocator::common_allocator().get_peak_memory();
  }
  return vulkan::allocator().get_peak_memory();
}

void reset_peak_memory() {
  if (allocator::use_cpu_allocator()) {
    allocator::common_allocator().reset_peak_memory();
    return;
  }
  vulkan::allocator().reset_peak_memory();
}

size_t set_memory_limit(size_t limit) {
  if (allocator::use_cpu_allocator()) {
    return allocator::common_allocator().set_memory_limit(limit);
  }
  return vulkan::allocator().set_memory_limit(limit);
}

size_t get_memory_limit() {
  if (allocator::use_cpu_allocator()) {
    return allocator::common_allocator().get_memory_limit();
  }
  return vulkan::allocator().get_memory_limit();
}

size_t get_cache_memory() {
  if (allocator::use_cpu_allocator()) {
    return 0;
  }
  return vulkan::allocator().get_cache_memory();
}

size_t set_cache_limit(size_t limit) {
  if (allocator::use_cpu_allocator()) {
    (void)limit;
    return 0;
  }
  // No buffer cache in initial Vulkan implementation
  (void)limit;
  return 0;
}

void clear_cache() {
  if (allocator::use_cpu_allocator()) {
    return;
  }
  vulkan::allocator().clear_cache();
}

size_t set_wired_limit(size_t) {
  // Not applicable for Vulkan
  return 0;
}

} // namespace mlx::core
