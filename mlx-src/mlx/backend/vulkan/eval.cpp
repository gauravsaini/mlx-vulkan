// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - GPU eval interface implementation

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
  {
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  auto& stream = arr.primitive().stream();
  auto& encoder = vulkan::get_command_encoder(stream);

  // Keep input buffers alive until GPU finishes
  for (auto& in : arr.inputs()) {
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (encoder.needs_commit()) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    // Note: actual commit deferred until finalize() to batch operations
    encoder.op_count = 0; // reset counter after notify
  }
}

void finalize(Stream s) {
  vulkan::device(s.device).commit(s);
}

void synchronize(Stream s) {
  vulkan::device(s.device).synchronize(s);
}

} // namespace mlx::core::gpu
