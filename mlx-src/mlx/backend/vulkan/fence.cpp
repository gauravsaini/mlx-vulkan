// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Fence implementation (CPU-GPU sync)

#include "mlx/fence.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

// For Vulkan, we use a simple CPU-side counter + condition variable for
// cross-stream sync, plus a GPU-side timeline semaphore for GPU-GPU sync.
// The simple implementation defers to CPU sync (like no_gpu) with GPU drain.
struct FenceImpl {
  uint32_t count{0};
  uint32_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

Fence::Fence(Stream s) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  if (stream.device == Device::gpu) {
    // Drain GPU stream first, then CPU waits
    vulkan::device(stream.device).synchronize(stream);
  }
  if (stream.device == Device::cpu) {
    uint32_t wait_count = f.count;
    scheduler::enqueue(stream, [wait_count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      if (f.value >= wait_count) return;
      f.cv.wait(lk, [&f, wait_count] { return f.value >= wait_count; });
    });
  }
}

void Fence::update(Stream stream, const array&, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;
  if (stream.device == Device::gpu) {
    uint32_t signal_count = f.count;
    // Schedule CPU signal after GPU drains
    vulkan::get_command_encoder(stream).add_completed_handler(
        [signal_count, fence_ = fence_]() mutable {
          auto& f = *static_cast<FenceImpl*>(fence_.get());
          std::unique_lock<std::mutex> lk(f.mtx);
          f.value = signal_count;
          f.cv.notify_all();
        });
  } else if (stream.device == Device::cpu) {
    uint32_t signal_count = f.count;
    scheduler::enqueue(stream, [signal_count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      f.value = signal_count;
      f.cv.notify_all();
    });
  }
}

} // namespace mlx::core
