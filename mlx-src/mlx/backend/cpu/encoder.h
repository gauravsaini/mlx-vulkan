// Copyright © 2025 Apple Inc.

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/allocator.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

// Number of dispatches per scheduler task
constexpr int DISPATCHES_PER_TASK = 10;

struct MLX_API CommandEncoder {
  CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  CommandEncoder(CommandEncoder&&) = delete;
  CommandEncoder& operator=(CommandEncoder&&) = delete;

  void set_input_array(const array& a) {
    retain(a);
  }
  void set_output_array(array& a) {
    retain(a);
    if (stream_.device.type == Device::DeviceType::gpu) {
      output_arrays_.push_back(array::unsafe_weak_copy(a));
    }
  }

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array arr) {
    retain(arr);
  }

  void add_temporaries(std::vector<array> arrays) {
    for (const auto& arr : arrays) {
      retain(arr);
    }
  }

  std::vector<array>& temporaries() {
    return temporaries_;
  }

  std::vector<std::shared_ptr<array::Data>>& data_references() {
    return data_references_;
  }

  template <class F, class... Args>
  void dispatch(F&& f, Args&&... args) {
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    // When called from a GPU-stream eval_gpu (unified memory CPU fallback),
    // execute the task synchronously — the GPU scheduler cannot manage CPU
    // tasks enqueued to a different stream, and on MoltenVK the data pointer
    // is already CPU-accessible (no staging needed).
    if (stream_.device.type == Device::DeviceType::gpu) {
      task();
      flush_gpu_outputs();
      temporaries_.clear();
      data_references_.clear();
      output_arrays_.clear();
      return;
    }

    num_ops_ = (num_ops_ + 1) % DISPATCHES_PER_TASK;
    if (num_ops_ == 0) {
      scheduler::notify_new_task(stream_);
      auto task_wrap = [s = stream_, task = std::move(task)]() mutable {
        task();
        scheduler::notify_task_completion(s);
      };
      scheduler::enqueue(stream_, std::move(task_wrap));
    } else {
      scheduler::enqueue(stream_, std::move(task));
    }
  }

 private:
  void flush_gpu_outputs() {
    std::unordered_set<const void*> flushed_buffers;
    for (auto& out : output_arrays_) {
      if (!out.data_shared_ptr()) {
        continue;
      }
      const void* buffer_ptr = out.buffer().ptr();
      if (!buffer_ptr || !flushed_buffers.insert(buffer_ptr).second) {
        continue;
      }
      auto bytes = out.buffer_size();
      if (bytes == 0) {
        continue;
      }
      auto* host_ptr = out.buffer().raw_ptr();
      if (!host_ptr) {
        continue;
      }
      allocator::copy_from_host(out.buffer(), host_ptr, bytes);
    }
  }

  void retain(const array& arr) {
    if (auto data = arr.data_shared_ptr()) {
      data_references_.push_back(std::move(data));
    } else {
      temporaries_.push_back(arr);
    }
  }

  Stream stream_;
  std::vector<array> temporaries_;
  std::vector<std::shared_ptr<array::Data>> data_references_;
  std::vector<array> output_arrays_;
  int num_ops_{0};
};

MLX_API CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::cpu
