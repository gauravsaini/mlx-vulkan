// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Device info implementation

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core::gpu {

bool is_available() {
  return true;
}

int device_count() {
  // For now return 1 (the selected device)
  // Could be extended to enumerate all Vulkan physical devices
  return 1;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  static thread_local std::unordered_map<std::string, std::variant<std::string, size_t>> info;

  if (device_index != 0) {
    static auto empty = std::unordered_map<std::string, std::variant<std::string, size_t>>();
    return empty;
  }

  auto& dev = vulkan::device(mlx::core::Device{mlx::core::Device::gpu, 0});

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(dev.vk_physical_device(), &props);

  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(dev.vk_physical_device(), &mem_props);

  info["device_name"] = std::string(props.deviceName);

  // Architecture string based on vendor
  std::string arch = "vulkan_";
  switch (props.vendorID) {
    case 0x1002: arch += "amd"; break;
    case 0x10DE: arch += "nvidia"; break;
    case 0x8086: arch += "intel"; break;
    case 0x13B5: arch += "arm"; break;
    case 0x5143: arch += "qualcomm"; break;
    default:     arch += "unknown"; break;
  }
  info["architecture"] = arch;

  // API version
  char api_ver[32];
  snprintf(api_ver, sizeof(api_ver), "%d.%d.%d",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
  info["vulkan_api_version"] = std::string(api_ver);

  // Total device-local memory
  size_t total_device_mem = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total_device_mem += mem_props.memoryHeaps[i].size;
    }
  }
  info["memory_size"] = total_device_mem;
  info["total_memory"] = total_device_mem;

  return info;
}

} // namespace mlx::core::gpu
