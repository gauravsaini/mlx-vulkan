// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Fence implementation (CPU-GPU cross stream)

#include "mlx/fence.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/scheduler.h"

#include <stdexcept>

namespace mlx::core {

struct FenceImpl {
  Stream stream;
  uint32_t count{0};
  VkSemaphore sem{VK_NULL_HANDLE};

  FenceImpl(Stream s) : stream(s) {
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

Fence::Fence(Stream stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{stream}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (f.stream == stream) {
    return;
  }

  if (f.stream.device == Device::gpu) {
    vulkan::device(f.stream.device).commit(f.stream);
  }

  if (stream.device == Device::gpu) {
    // MoltenVK does not support GPU-side timeline semaphore waits in
    // vkQueueSubmit, so we cannot encode a semaphore-wait into the command
    // buffer. Instead, block the calling (main) thread until the CPU stream
    // signals the timeline semaphore. The signal is enqueued to the CPU
    // stream thread pool by Fence::update *after* the CPU primitive
    // (e.g. hadamard butterfly) is dispatched, so waiting here guarantees
    // that all CPU-stream work preceding this fence has completed before
    // any subsequent GPU operation (e.g. subtraction) reads those buffers.
    //
    // Note: the old implementation only called vkQueueWaitIdle which drains
    // the GPU queue but does NOT wait for the CPU stream thread -- causing a
    // race when GPU ops immediately follow CPU ops on a different stream.
    uint64_t wait_count = f.count;
    if (wait_count > 0) {
      VkSemaphoreWaitInfo wait_info{};
      wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      wait_info.semaphoreCount = 1;
      wait_info.pSemaphores = &f.sem;
      wait_info.pValues = &wait_count;
      if (vkWaitSemaphores(
              vulkan::device(Device::gpu).vk_device(),
              &wait_info,
              UINT64_MAX) != VK_SUCCESS) {
        throw std::runtime_error("[Fence::wait] vkWaitSemaphores failed");
      }
    }
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
