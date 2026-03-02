// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Fence implementation (CPU-GPU cross stream)

#include "mlx/fence.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/scheduler.h"

#include <stdexcept>

namespace mlx::core {

struct FenceImpl {
  uint32_t count{0};
  VkSemaphore sem{VK_NULL_HANDLE};

  FenceImpl() {
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &timeline_info;

    if (vkCreateSemaphore(
            vulkan::device(Device::gpu).vk_device(),
            &createInfo,
            nullptr,
            &sem) != VK_SUCCESS) {
      throw std::runtime_error("[Fence] Failed to create timeline semaphore");
    }
  }

  ~FenceImpl() {
    if (sem != VK_NULL_HANDLE) {
      vkDestroySemaphore(vulkan::device(Device::gpu).vk_device(), sem, nullptr);
    }
  }
};

Fence::Fence(Stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (stream.device == Device::gpu) {
    // MoltenVK does not support GPU-side timeline semaphore waits in
    // vkQueueSubmit. Drain the compute queue directly — this ensures any
    // previously-submitted work (and its completion handlers that signal the
    // fence semaphore) has run. We use vkQueueWaitIdle directly rather than
    // synchronize(stream) because synchronize() commits the current in-flight
    // command buffer which would corrupt the recording state for subsequent
    // ops.
    auto& dev = vulkan::device(stream.device);
    vkQueueWaitIdle(dev.compute_queue());
  } else if (stream.device == Device::cpu) {
    uint64_t wait_count = f.count;
    scheduler::enqueue(stream, [fence_ = fence_, wait_count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      VkSemaphoreWaitInfo wait_info{};
      wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      wait_info.semaphoreCount = 1;
      wait_info.pSemaphores = &f.sem;
      wait_info.pValues = &wait_count;
      if (vkWaitSemaphores(
              vulkan::device(Device::gpu).vk_device(),
              &wait_info,
              UINT64_MAX) != VK_SUCCESS) {
        throw std::runtime_error("[Fence::wait] CPU Wait failed");
      }
    });
  }
}

void Fence::update(Stream stream, const array&, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;
  uint64_t signal_count = f.count;

  if (stream.device == Device::gpu) {
    // Signal the timeline semaphore via completion handler after GPU work
    // finishes. MoltenVK does not support GPU-side timeline semaphore signals
    // in vkQueueSubmit.
    vulkan::device(stream.device)
        .add_completed_handler(
            stream, [fence_ = fence_, signal_count]() mutable {
              auto& f = *static_cast<FenceImpl*>(fence_.get());
              VkSemaphoreSignalInfo signal_info{};
              signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
              signal_info.semaphore = f.sem;
              signal_info.value = signal_count;
              vkSignalSemaphore(
                  vulkan::device(Device::gpu).vk_device(), &signal_info);
            });
  } else if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, signal_count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      VkSemaphoreSignalInfo signal_info{};
      signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
      signal_info.semaphore = f.sem;
      signal_info.value = signal_count;
      if (vkSignalSemaphore(
              vulkan::device(Device::gpu).vk_device(), &signal_info) !=
          VK_SUCCESS) {
        throw std::runtime_error("[Fence::update] CPU Signal failed");
      }
    });
  }
}

} // namespace mlx::core
