// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Event (timeline semaphore) implementation

#include "mlx/event.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/event.h"
#include "mlx/scheduler.h"

#include <memory>
#include <stdexcept>

namespace mlx::core {

namespace {

struct VulkanEventState {
  vulkan::VulkanEvent vk_event;
  uint64_t signal_value{0};
};

} // anonymous namespace

Event::Event(Stream stream) : stream_(stream) {
  auto* state = new VulkanEventState{};
  event_ = std::shared_ptr<void>(
      state, [](void* ptr) { delete static_cast<VulkanEventState*>(ptr); });
}

void Event::wait() {
  if (!valid()) { fprintf(stderr, "[EVENT] wait: not valid\n"); fflush(stderr); return; }
  auto* state = static_cast<VulkanEventState*>(event_.get());
  fprintf(stderr, "[EVENT] wait: waiting for signal_value=%lu\n", (unsigned long)state->signal_value); fflush(stderr);
  state->vk_event.wait(state->signal_value);
  fprintf(stderr, "[EVENT] wait: done\n"); fflush(stderr);
}

void Event::wait(Stream stream) {
  if (!valid()) return;
  auto* state = static_cast<VulkanEventState*>(event_.get());
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    vulkan::device(stream.device).add_wait_semaphore(stream, state->vk_event.semaphore(), state->signal_value);
  }
}

void Event::signal(Stream stream) {
  if (!valid()) return;
  auto* state = static_cast<VulkanEventState*>(event_.get());
  state->signal_value = value_;

  if (stream.device == Device::gpu) {
    vulkan::device(stream.device).add_signal_semaphore(stream, state->vk_event.semaphore(), value_);
  } else {
    scheduler::enqueue(stream, [event_ = event_, val = value_]() mutable {
      auto* s = static_cast<VulkanEventState*>(event_.get());
      s->vk_event.signal(val);
    });
  }
}

bool Event::is_signaled() const {
  if (!valid()) return false;
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

  VkResult res = vkCreateSemaphore(dev.vk_device(), &create_info, nullptr, &semaphore_);
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
