// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Timeline semaphore events

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace mlx::core::vulkan {

class VulkanEvent {
 public:
  VulkanEvent();
  ~VulkanEvent();

  VulkanEvent(const VulkanEvent&) = delete;
  VulkanEvent& operator=(const VulkanEvent&) = delete;

  void signal(uint64_t value);
  void wait(uint64_t value);

  VkSemaphore semaphore() const { return semaphore_; }

 private:
  VkSemaphore semaphore_{VK_NULL_HANDLE};
};

} // namespace mlx::core::vulkan
