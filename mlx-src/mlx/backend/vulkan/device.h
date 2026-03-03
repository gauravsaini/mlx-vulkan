// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Device management

#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

struct CommandEncoder {
  VkCommandPool pool{VK_NULL_HANDLE};
  VkDescriptorPool desc_pool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  std::vector<std::function<void()>> completion_handlers;
  std::vector<std::shared_ptr<array::Data>> temporaries;
  std::vector<std::pair<VkSemaphore, uint64_t>> wait_semaphores;
  std::vector<std::pair<VkSemaphore, uint64_t>> signal_semaphores;
  int op_count{0};
  bool recording{false};
  bool first_commit{true};
};

// Pipeline cache entry
struct PipelineCacheEntry {
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorSetLayout ds_layout{VK_NULL_HANDLE};
};

class Device {
 public:
  explicit Device();
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Stream management
  CommandEncoder& get_command_encoder(Stream s);
  void commit(Stream s);
  void synchronize(Stream s);

  // Thread-safe encoder states
  void add_temporary(Stream s, const array& arr);
  void add_completed_handler(Stream s, std::function<void()> handler);
  void add_wait_semaphore(Stream s, VkSemaphore sem, uint64_t val);
  void add_signal_semaphore(Stream s, VkSemaphore sem, uint64_t val);
  bool needs_commit(Stream s);

  // Pipeline cache
  VkPipeline get_pipeline(
      const std::string& name,
      VkPipelineLayout& layout_out,
      VkDescriptorSetLayout& ds_layout_out,
      uint32_t num_bindings,
      uint32_t push_constant_size = 0,
      const VkSpecializationInfo* specialization_info = nullptr);

  // Memory management
  VmaAllocator vma_allocator() const {
    return vma_allocator_;
  }
  VkDevice vk_device() const {
    return device_;
  }
  VkPhysicalDevice vk_physical_device() const {
    return physical_device_;
  }
  VkQueue compute_queue() const {
    return compute_queue_;
  }
  uint32_t queue_family() const {
    return compute_queue_family_;
  }

  // Subgroup / workgroup size queries
  uint32_t subgroup_size() const {
    return subgroup_size_;
  }
  uint32_t preferred_workgroup_size() const {
    return preferred_workgroup_size_;
  }

  // Descriptor set allocation (one-shot, freed after each commit)
  VkDescriptorSet alloc_descriptor_set(Stream s, VkDescriptorSetLayout layout);

 private:
  void init_instance();
  void select_physical_device();
  void query_subgroup_size();
  void create_logical_device();
  void create_vma();
  void create_pipeline_cache();
  void save_pipeline_cache();

  VkInstance instance_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_{0};

  VmaAllocator vma_allocator_{VK_NULL_HANDLE};
  VkPipelineCache pipeline_cache_{VK_NULL_HANDLE};

  VkBuffer dummy_buffer_{VK_NULL_HANDLE};
  VmaAllocation dummy_alloc_{VK_NULL_HANDLE};

  // Subgroup / preferred workgroup sizes (queried at device init)
  uint32_t subgroup_size_{32};
  uint32_t preferred_workgroup_size_{256};

  std::mutex mutex_;
  std::unordered_map<int, CommandEncoder> encoders_;
  std::unordered_map<std::string, PipelineCacheEntry> pipeline_map_;

#ifdef MLX_VULKAN_VALIDATION
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
#endif
};

// Singleton device accessor
Device& device(mlx::core::Device d);
CommandEncoder& get_command_encoder(Stream s);

} // namespace mlx::core::vulkan
