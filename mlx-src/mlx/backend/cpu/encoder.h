// Copyright © 2025 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"
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

  void set_input_array(const array& a) {}
  void set_output_array(array& a) {}

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array arr) {
    temporaries_.push_back(std::move(arr));
  }

  void add_temporaries(std::vector<array> arrays) {
    temporaries_.insert(
        temporaries_.end(),
        std::make_move_iterator(arrays.begin()),
        std::make_move_iterator(arrays.end()));
  }

  std::vector<array>& temporaries() {
    return temporaries_;
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
  Stream stream_;
  std::vector<array> temporaries_;
  int num_ops_{0};
};

MLX_API CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::cpu
