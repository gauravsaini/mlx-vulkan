// Copyright © 2025 Apple Inc.

#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/allocator.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

// Number of dispatches per scheduler task
constexpr int DISPATCHES_PER_TASK = 10;

inline bool fail_on_gpu_cpu_fallback() {
  const char* env = std::getenv("MLX_VULKAN_FAIL_ON_CPU_FALLBACK");
  return env && env[0] != '\0' && env[0] != '0';
}

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
    output_arrays_.push_back(array::unsafe_weak_copy(a));
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
    // or while a Vulkan CPU-eval temporarily forces CPU allocations, execute
    // synchronously. The worker thread cannot inherit the allocator override,
    // and these paths are correctness-first rather than throughput-critical.
    if (
        stream_.device.type == Device::DeviceType::gpu ||
        allocator::cpu_allocator_override_enabled()) {
      if (
          stream_.device.type == Device::DeviceType::gpu &&
          fail_on_gpu_cpu_fallback()) {
        throw std::runtime_error(
            "CPU fallback executed on a GPU stream while "
            "MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1.");
      }
      task();
      flush_output_arrays(output_arrays_);
      temporaries_.clear();
      data_references_.clear();
      output_arrays_.clear();
      return;
    }

    auto temporaries = std::move(temporaries_);
    auto data_references = std::move(data_references_);
    auto output_arrays = std::move(output_arrays_);

    num_ops_ = (num_ops_ + 1) % DISPATCHES_PER_TASK;
    if (num_ops_ == 0) {
      scheduler::notify_new_task(stream_);
      auto task_wrap = [
                           s = stream_,
                           task = std::move(task),
                           temporaries = std::move(temporaries),
                           data_references = std::move(data_references),
                           output_arrays = std::move(output_arrays)]() mutable {
        task();
        flush_output_arrays(output_arrays);
        scheduler::notify_task_completion(s);
      };
      scheduler::enqueue(stream_, std::move(task_wrap));
    } else {
      auto task_wrap = [
                           task = std::move(task),
                           temporaries = std::move(temporaries),
                           data_references = std::move(data_references),
                           output_arrays = std::move(output_arrays)]() mutable {
        task();
        flush_output_arrays(output_arrays);
      };
      scheduler::enqueue(stream_, std::move(task_wrap));
    }
  }

 private:
  static void flush_output_arrays(std::vector<array>& output_arrays) {
    std::unordered_set<const void*> flushed_buffers;
    for (auto& out : output_arrays) {
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
