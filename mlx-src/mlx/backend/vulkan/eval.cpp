// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU eval interface implementation

#include <unordered_set>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream s) {
  // Initialize Vulkan device and create per-stream command pool
  auto& d = vulkan::device(s.device);
  (void)d.get_command_encoder(s);
}

void eval(array& arr) {
  auto outputs = arr.outputs();

  // Dispatch the primitive's GPU implementation
  arr.primitive().eval_gpu(arr.inputs(), outputs);

  auto& stream = arr.primitive().stream();
  auto& encoder = vulkan::get_command_encoder(stream);

  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (vulkan::device(stream.device).needs_commit(stream)) {
    scheduler::notify_new_task(stream);
    vulkan::device(stream.device).add_completed_handler(stream, 
        [stream, buffers = std::move(buffers)]() {
          scheduler::notify_task_completion(stream);
        });
    vulkan::device(stream.device).commit(stream);
  } else {
    vulkan::device(stream.device).add_completed_handler(stream, [buffers = std::move(buffers)]() {});
  }
}

void finalize(Stream s) {
  vulkan::device(s.device).commit(s);
}

void synchronize(Stream s) {
  vulkan::device(s.device).synchronize(s);
}

} // namespace mlx::core::gpu
