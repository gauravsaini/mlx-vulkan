// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Event (timeline semaphore) implementation

#include "mlx/event.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/event.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace mlx::core {

namespace {

struct CpuEventState {
  uint64_t signal_value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

struct VulkanEventState {
  vulkan::VulkanEvent vk_event;
  uint64_t signal_value{0};
};

bool use_cpu_event(const Stream& stream) {
  return stream.device == Device::cpu || !gpu::is_available();
}

} // anonymous namespace

Event::Event(Stream stream) : stream_(stream) {
  if (use_cpu_event(stream)) {
    event_ = std::shared_ptr<void>(
        new CpuEventState{},
        [](void* ptr) { delete static_cast<CpuEventState*>(ptr); });
  } else {
    event_ = std::shared_ptr<void>(
        new VulkanEventState{},
        [](void* ptr) { delete static_cast<VulkanEventState*>(ptr); });
  }
}

void Event::wait() {
  if (!valid())
    return;
  if (use_cpu_event(stream_)) {
    auto* state = static_cast<CpuEventState*>(event_.get());
    std::unique_lock<std::mutex> lk(state->mtx);
    if (state->signal_value >= value_) {
      return;
    }
    state->cv.wait(lk, [state, value = value_] {
      return state->signal_value >= value;
    });
  } else {
    auto* state = static_cast<VulkanEventState*>(event_.get());
    state->vk_event.wait(state->signal_value);
  }
}

void Event::wait(Stream stream) {
  if (!valid())
    return;
  if (use_cpu_event(stream_)) {
    if (stream.device == Device::cpu) {
      scheduler::enqueue(stream, [*this]() mutable { wait(); });
    } else {
      wait();
    }
  } else if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    // MoltenVK does not support GPU-side timeline semaphore waits in
    // vkQueueSubmit (results in VK_ERROR_DEVICE_LOST). Use CPU-blocking: drain
    // the signal stream first so the completion handler has fired, then block
    // CPU until signaled.
    vulkan::device(stream.device).synchronize(stream_);
    wait();
  }
}

void Event::signal(Stream stream) {
  if (!valid())
    return;
  if (use_cpu_event(stream_)) {
    auto notify = [event_ = event_, val = value_]() mutable {
      auto* state = static_cast<CpuEventState*>(event_.get());
      {
        std::lock_guard<std::mutex> lk(state->mtx);
        state->signal_value = val;
      }
      state->cv.notify_all();
    };
    if (stream.device == Device::gpu && gpu::is_available()) {
      vulkan::device(stream.device)
          .add_completed_handler(stream, std::move(notify));
    } else {
      scheduler::enqueue(stream, std::move(notify));
    }
  } else {
    auto* state = static_cast<VulkanEventState*>(event_.get());
    state->signal_value = value_;

    if (stream.device == Device::gpu) {
      // Signal the timeline semaphore via completion handler after GPU work
      // finishes. add_completed_handler fires from the background thread after
      // vkWaitForFences.
      vulkan::device(stream.device)
          .add_completed_handler(
              stream, [event_ = event_, val = value_]() mutable {
                auto* s = static_cast<VulkanEventState*>(event_.get());
                s->vk_event.signal(val);
              });
    } else {
      scheduler::enqueue(stream, [event_ = event_, val = value_]() mutable {
        auto* s = static_cast<VulkanEventState*>(event_.get());
        s->vk_event.signal(val);
      });
    }
  }
}

bool Event::is_signaled() const {
  if (!valid())
    return false;
  if (use_cpu_event(stream_)) {
    auto* state = static_cast<CpuEventState*>(event_.get());
    std::lock_guard<std::mutex> lk(state->mtx);
    return state->signal_value >= value_;
  } else {
    auto* state = static_cast<VulkanEventState*>(event_.get());
    auto& dev = vulkan::device(mlx::core::Device{mlx::core::Device::gpu, 0});
    uint64_t current_value = 0;
    VkResult res = vkGetSemaphoreCounterValue(
        dev.vk_device(), state->vk_event.semaphore(), &current_value);
    if (res != VK_SUCCESS) {
      return false;
    }
    return current_value >= state->signal_value;
  }
}

} // namespace mlx::core

// VulkanEvent implementation
namespace mlx::core::vulkan {

VulkanEvent::VulkanEvent() {
  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});

  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = 0;

  VkSemaphoreCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &type_info;

  VkResult res =
      vkCreateSemaphore(dev.vk_device(), &create_info, nullptr, &semaphore_);
  if (res != VK_SUCCESS) {
    throw std::runtime_error("Vulkan: failed to create timeline semaphore");
  }
}

VulkanEvent::~VulkanEvent() {
  if (semaphore_ != VK_NULL_HANDLE) {
    auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});
    vkDestroySemaphore(dev.vk_device(), semaphore_, nullptr);
  }
}

void VulkanEvent::signal(uint64_t value) {
  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});
  VkSemaphoreSignalInfo signal_info{};
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.semaphore = semaphore_;
  signal_info.value = value;
  vkSignalSemaphore(dev.vk_device(), &signal_info);
}

void VulkanEvent::wait(uint64_t value) {
  auto& dev = device(mlx::core::Device{mlx::core::Device::gpu, 0});
  VkSemaphoreWaitInfo wait_info{};
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.semaphoreCount = 1;
  wait_info.pSemaphores = &semaphore_;
  wait_info.pValues = &value;
  vkWaitSemaphores(dev.vk_device(), &wait_info, UINT64_MAX);
}

} // namespace mlx::core::vulkan
