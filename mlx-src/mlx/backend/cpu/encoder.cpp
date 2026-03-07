// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/encoder.h"

namespace mlx::core::cpu {

CommandEncoder& get_command_encoder(Stream stream) {
  static std::unordered_map<int, CommandEncoder> encoder_map;
  // Key combines device type (0=cpu, 1=gpu) and stream index so that
  // GPU-stream fallback callers get a separate encoder from CPU-stream callers
  // (same stream index can exist for both CPU and GPU streams).
  int key = (stream.device.type == Device::DeviceType::gpu ? 100000 : 0) + stream.index;
  auto it = encoder_map.find(key);
  if (it == encoder_map.end()) {
    it = encoder_map.emplace(key, stream).first;
  }
  return it->second;
}

} // namespace mlx::core::cpu
