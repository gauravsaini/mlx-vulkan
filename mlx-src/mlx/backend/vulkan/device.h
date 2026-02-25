// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Device management

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

// A wrapper around a VkCommandBuffer that tracks state for a single stream
struct CommandEncoder {
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkCommandPool pool{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  std::vector<std::function<void()>> completion_handlers;
  std::vector<std::shared_ptr<array::Data>> temporaries;
  int op_count{0};
  bool recording{false};

  void add_temporary(const array& arr) {
    temporaries.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> handler) {
    completion_handlers.push_back(std::move(handler));
  }

  bool needs_commit() const {
    return op_count > 0 && op_count % 64 == 0;
  }
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

  // Pipeline cache
  VkPipeline get_pipeline(
      const std::string& name,
      VkPipelineLayout& layout_out,
      VkDescriptorSetLayout& ds_layout_out,
      uint32_t num_bindings,
      uint32_t push_constant_size = 0);

  // Memory management
  VmaAllocator vma_allocator() const { return vma_allocator_; }
  VkDevice vk_device() const { return device_; }
  VkPhysicalDevice vk_physical_device() const { return physical_device_; }
  VkQueue compute_queue() const { return compute_queue_; }
  uint32_t queue_family() const { return compute_queue_family_; }

  // Descriptor set allocation (one-shot, freed after each commit)
  VkDescriptorSet alloc_descriptor_set(VkDescriptorSetLayout layout);

 private:
  void init_instance();
  void select_physical_device();
  void create_logical_device();
  void create_vma();
  void create_pipeline_cache();
  void save_pipeline_cache();
  void create_descriptor_pool();

  VkInstance instance_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_{0};

  VmaAllocator vma_allocator_{VK_NULL_HANDLE};
  VkPipelineCache pipeline_cache_{VK_NULL_HANDLE};

  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};

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
