// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Utility helpers

#pragma once

#include <vulkan/vulkan.h>
#include "mlx/array.h"
#include "mlx/dtype.h"

namespace mlx::core::vulkan {

// Integer ceil division
inline uint32_t div_ceil(uint64_t a, uint64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

// Default workgroup size for 1D compute dispatches.
// Kept as a compile-time constant for existing dispatch math;
// use preferred_workgroup_size() for device-adaptive sizing.
constexpr uint32_t WORKGROUP_SIZE = 256;

// Returns the device-queried preferred workgroup size (a multiple of the
// hardware subgroup width, capped at 256).  Forward-declared here;
// defined in utils.cpp to avoid a circular dependency on device.h.
uint32_t preferred_workgroup_size();

// Map MLX Dtype to Vulkan format (for descriptor set hints)
VkFormat to_vk_format(Dtype dtype);

// Type size in bytes
uint32_t dtype_size(Dtype dtype);

// String name for dtype (for pipeline cache keying)
const char* dtype_name(Dtype dtype);

// Insert a pipeline barrier ensuring prior compute writes are visible to subsequent compute reads
void insert_buffer_barrier(VkCommandBuffer cmd, VkBuffer buffer);

// Insert barrier for all output buffers of an array
void insert_buffer_barrier(VkCommandBuffer cmd, const array& arr);

// Check VkResult and throw on error
void vk_check(VkResult result, const char* msg);

} // namespace mlx::core::vulkan
