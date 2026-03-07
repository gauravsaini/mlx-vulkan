// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Device implementation
//
// Initializes VkInstance, selects physical device, creates logical device,
// VMA allocator, pipeline cache, and manages per-stream CommandEncoders.

#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Define VMA implementation in this translation unit
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace mlx::core::vulkan {

namespace {

#ifdef MLX_VULKAN_VALIDATION
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
  }
  return VK_FALSE;
}
#endif

// Read SPIR-V binary from file
std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Vulkan: cannot open SPIR-V file: " + path);
  }
  size_t size = file.tellg();
  file.seekg(0);
  std::vector<uint32_t> code(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(code.data()), size);
  return code;
}

} // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
// Device construction
// ────────────────────────────────────────────────────────────────────────────

Device::Device() {
  init_instance();
  select_physical_device();
  query_subgroup_size();
  create_logical_device();
  create_vma();
  create_pipeline_cache();

  // Create a 16-byte dummy storage buffer for pipeline warmup
  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = 16;
  buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  vk_check(
      vmaCreateBuffer(
          vma_allocator_,
          &buf_info,
          &alloc_info,
          &dummy_buffer_,
          &dummy_alloc_,
          nullptr),
      "vmaCreateBuffer (dummy)");
}

Device::~Device() {
  vkDeviceWaitIdle(device_);

  if (dummy_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(vma_allocator_, dummy_buffer_, dummy_alloc_);
  }

  // Save pipeline cache
  save_pipeline_cache();

  // Destroy pipelines
  for (auto& [name, entry] : pipeline_map_) {
    if (entry.pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device_, entry.pipeline, nullptr);
    if (entry.layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device_, entry.layout, nullptr);
    if (entry.ds_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device_, entry.ds_layout, nullptr);
  }

  // Destroy encoders
  for (auto& [idx, enc] : encoders_) {
    for (auto& handler : enc.completion_handlers) {
      handler();
    }
    enc.completion_handlers.clear();

    if (enc.fence != VK_NULL_HANDLE)
      vkDestroyFence(device_, enc.fence, nullptr);
    if (enc.pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(device_, enc.pool, nullptr);
    if (enc.desc_pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(device_, enc.desc_pool, nullptr);
  }

  // Join all background commit threads
  for (auto& t : commit_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }

  if (pipeline_cache_ != VK_NULL_HANDLE)
    vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
  if (vma_allocator_ != VK_NULL_HANDLE)
    vmaDestroyAllocator(vma_allocator_);

#ifdef MLX_VULKAN_VALIDATION
  if (debug_messenger_ != VK_NULL_HANDLE) {
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance_, "vkDestroyDebugUtilsMessengerEXT");
    if (fn)
      fn(instance_, debug_messenger_, nullptr);
  }
#endif

  if (device_ != VK_NULL_HANDLE)
    vkDestroyDevice(device_, nullptr);
  if (instance_ != VK_NULL_HANDLE)
    vkDestroyInstance(instance_, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// Instance creation
// ────────────────────────────────────────────────────────────────────────────

void Device::init_instance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "MLX";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "MLX Vulkan Backend";
  app_info.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char*> extensions = {
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
#if defined(__APPLE__)
      "VK_KHR_portability_enumeration",
#endif
  };

  std::vector<const char*> layers;

#ifdef MLX_VULKAN_VALIDATION
  layers.push_back("VK_LAYER_KHRONOS_validation");
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

  VkInstanceCreateInfo inst_info{};
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
#if defined(__APPLE__)
  inst_info.flags |=
      0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#endif
  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  inst_info.ppEnabledLayerNames = layers.data();
  inst_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  inst_info.ppEnabledExtensionNames = extensions.data();

  vk_check(
      vkCreateInstance(&inst_info, nullptr, &instance_), "vkCreateInstance");

#ifdef MLX_VULKAN_VALIDATION
  VkDebugUtilsMessengerCreateInfoEXT dbg{};
  dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  dbg.pfnUserCallback = debug_callback;

  auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance_, "vkCreateDebugUtilsMessengerEXT");
  if (fn)
    fn(instance_, &dbg, nullptr, &debug_messenger_);
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Physical device selection: prefer discrete, fall back to integrated
// ────────────────────────────────────────────────────────────────────────────

void Device::select_physical_device() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (count == 0) {
    throw std::runtime_error("Vulkan: no physical devices found");
  }
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance_, &count, devices.data());

  // Check env override
  const char* env_idx = std::getenv("MLX_VULKAN_DEVICE");
  if (env_idx) {
    int idx = std::atoi(env_idx);
    if (idx >= 0 && idx < static_cast<int>(count)) {
      physical_device_ = devices[idx];
      VkPhysicalDeviceProperties p{};
      vkGetPhysicalDeviceProperties(physical_device_, &p);
      fprintf(stderr, "[MLX Vulkan] Using device %d: %s\n", idx, p.deviceName);
      return;
    }
  }

  // Score and pick best device
  VkPhysicalDevice best = VK_NULL_HANDLE;
  int best_score = -1;

  for (auto& dev : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);

    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score += 1000;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score += 100;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
      score += 50;
    score += static_cast<int>(props.limits.maxComputeSharedMemorySize / 1024);

    if (score > best_score) {
      best_score = score;
      best = dev;
    }
  }

  physical_device_ = best;

  VkPhysicalDeviceProperties chosen{};
  vkGetPhysicalDeviceProperties(physical_device_, &chosen);
  fprintf(stderr, "[MLX Vulkan] Selected device: %s\n", chosen.deviceName);
}

// ────────────────────────────────────────────────────────────────────────────
// Subgroup size query (called right after physical device selection)
// ────────────────────────────────────────────────────────────────────────────

void Device::query_subgroup_size() {
  // Use VkPhysicalDeviceSubgroupProperties (core Vulkan 1.1+) via
  // vkGetPhysicalDeviceProperties2 to retrieve the hardware subgroup width.
  // On AMD RDNA this is 64, Intel Arc is 32, Apple M-series via MoltenVK is 32.
  VkPhysicalDeviceSubgroupProperties subgroup_props{};
  subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroup_props;

  vkGetPhysicalDeviceProperties2(physical_device_, &props2);

  subgroup_size_ = subgroup_props.subgroupSize;
  if (subgroup_size_ == 0) {
    // vkGetPhysicalDeviceProperties2 returned zero — safe fallback
    subgroup_size_ = 32;
  }

  // Round 256 up to the nearest multiple of subgroup_size_, cap at 256.
  // This gives a workgroup size perfectly aligned for 16x16 native matmul tiles
  // across all GPU backends.
  preferred_workgroup_size_ = std::min(
      256u, ((256u + subgroup_size_ - 1u) / subgroup_size_) * subgroup_size_);
}

// ────────────────────────────────────────────────────────────────────────────
// Logical device + compute queue
// ────────────────────────────────────────────────────────────────────────────

void Device::create_logical_device() {
  // Find compute queue family
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qf_props(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &qf_count, qf_props.data());

  compute_queue_family_ = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_queue_family_ = i;
      break;
    }
  }
  if (compute_queue_family_ == UINT32_MAX) {
    throw std::runtime_error("Vulkan: no compute queue family found");
  }

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = compute_queue_family_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  // Required device extensions
  std::vector<const char*> dev_extensions = {
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
  };

  // Optional: check for external memory host (zero-copy for APUs)
  uint32_t ext_count = 0;
  vkEnumerateDeviceExtensionProperties(
      physical_device_, nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> avail_exts(ext_count);
  vkEnumerateDeviceExtensionProperties(
      physical_device_, nullptr, &ext_count, avail_exts.data());

  auto has_ext = [&](const char* name) {
    for (auto& e : avail_exts) {
      if (strcmp(e.extensionName, name) == 0)
        return true;
    }
    return false;
  };

  if (has_ext(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    dev_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    dev_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    fprintf(
        stderr,
        "[MLX Vulkan] Zero-copy (VK_EXT_external_memory_host) available\n");
  }

  // Check for hardware tensor cores (Cooperative Matrix)
  if (has_ext("VK_KHR_cooperative_matrix")) {
    dev_extensions.push_back("VK_KHR_cooperative_matrix");
    has_cooperative_matrix_ = true;
    fprintf(
        stderr, "[MLX Vulkan] Cooperative Matrix (Tensor Cores) available\n");
  }

  // Enable timeline semaphore feature
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{};
  timeline_feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timeline_feat.timelineSemaphore = VK_TRUE;

  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &timeline_feat;
  // Basic float64 for double precision (optional)
  features2.features.shaderFloat64 = VK_FALSE;
  features2.features.shaderInt64 = VK_TRUE;

  VkDeviceCreateInfo dev_info{};
  dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dev_info.pNext = &features2;
  dev_info.queueCreateInfoCount = 1;
  dev_info.pQueueCreateInfos = &queue_info;
  dev_info.enabledExtensionCount = static_cast<uint32_t>(dev_extensions.size());
  dev_info.ppEnabledExtensionNames = dev_extensions.data();

  vk_check(
      vkCreateDevice(physical_device_, &dev_info, nullptr, &device_),
      "vkCreateDevice");

  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

  // MoltenVK WORKAROUND: The very first command buffer submitted to a queue on
  // Apple Silicon may silently abort/drop during async shader translations.
  // We submit and wait on an empty command buffer immediately to absorb the
  // fault.
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex = compute_queue_family_;
  VkCommandPool pool;
  vkCreateCommandPool(device_, &pool_info, nullptr, &pool);

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device_, &alloc_info, &cmd);

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(compute_queue_, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(compute_queue_);

  vkDestroyCommandPool(device_, pool, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// VMA allocator
// ────────────────────────────────────────────────────────────────────────────

void Device::create_vma() {
  VmaAllocatorCreateInfo vma_info{};
  vma_info.physicalDevice = physical_device_;
  vma_info.device = device_;
  vma_info.instance = instance_;
  vma_info.vulkanApiVersion = VK_API_VERSION_1_2;

  vk_check(
      vmaCreateAllocator(&vma_info, &vma_allocator_), "vmaCreateAllocator");
}

// ────────────────────────────────────────────────────────────────────────────
// Pipeline cache (persisted to disk across runs)
// ────────────────────────────────────────────────────────────────────────────

// Bump this version whenever shader push-constant layouts change.
// Prevents MoltenVK from loading stale binary cache blobs.
// History of recent bumps:
// v21 (2026-03-05): scan.comp added 4 parameter fields for multi-pass scans
// v20 (2026-03-05): gather_mm.comp moved shape/stride arrays to SSBO (PushConst 200B -> 56B)
// v19 (2026-03-05): unary.comp added UNARY_ISINF, UNARY_ISNAN, UNARY_ISNEGINF
// v18 (2026-03-05): ternary.comp extended push constants for 16-bit bool Select
// v22 (2026-03-07): hadamard push constant 12→16 bytes (added h_step field)
// v17 (2026-03-04): radix_sort.comp added for >128 element arrays
// v16 (2026-03-03): arange/binary_two/binary dtype fields extended (v10->v12/14/16)
static constexpr int kPipelineCacheVersion = 24;

static std::string pipeline_cache_path() {
  const char* home = std::getenv("HOME");
  if (!home)
    home = "/tmp";
  return std::string(home) + "/.cache/mlx_vulkan_pipeline_cache_v" +
      std::to_string(kPipelineCacheVersion) + ".bin";
}

void Device::create_pipeline_cache() {
  // Try to load existing cache
  std::vector<char> cache_data;
  std::ifstream f(pipeline_cache_path(), std::ios::binary | std::ios::ate);
  if (f) {
    size_t size = f.tellg();
    f.seekg(0);
    cache_data.resize(size);
    f.read(cache_data.data(), size);
  }

  VkPipelineCacheCreateInfo cache_info{};
  cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  cache_info.initialDataSize = cache_data.size();
  cache_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

  vk_check(
      vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_),
      "vkCreatePipelineCache");
}

void Device::save_pipeline_cache() {
  if (pipeline_cache_ == VK_NULL_HANDLE)
    return;
  size_t size = 0;
  vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr);
  if (size == 0)
    return;
  std::vector<char> data(size);
  vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data());

  // Ensure cache dir exists
  std::string path = pipeline_cache_path();
  size_t slash = path.rfind('/');
  if (slash != std::string::npos) {
    std::string dir = path.substr(0, slash);
    // mkdir -p
    std::string cmd = "mkdir -p " + dir;
    (void)std::system(cmd.c_str());
  }

  std::ofstream f(path, std::ios::binary);
  if (f)
    f.write(data.data(), data.size());
}

// ────────────────────────────────────────────────────────────────────────────
// Command encoder management (per stream)
// ────────────────────────────────────────────────────────────────────────────

CommandEncoder& Device::get_command_encoder(Stream s) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    return it->second;
  }

  // Create new encoder for this stream index
  CommandEncoder& enc = encoders_[s.index];

  // Command pool
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = compute_queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vk_check(
      vkCreateCommandPool(device_, &pool_info, nullptr, &enc.pool),
      "vkCreateCommandPool");

  VkDescriptorPoolSize dpool_size{};
  dpool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dpool_size.descriptorCount = 10000;

  VkDescriptorPoolCreateInfo desc_pool_info{};
  desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  desc_pool_info.maxSets = 2000;
  desc_pool_info.poolSizeCount = 1;
  desc_pool_info.pPoolSizes = &dpool_size;
  vk_check(
      vkCreateDescriptorPool(device_, &desc_pool_info, nullptr, &enc.desc_pool),
      "vkCreateDescriptorPool");

  // Allocate initial command buffer
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = enc.pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  vk_check(
      vkAllocateCommandBuffers(device_, &alloc_info, &enc.cmd),
      "vkAllocateCommandBuffers");

  // Fence for CPU-GPU sync — must be UNSIGNALED so vkQueueSubmit is valid
  // and vkWaitForFences actually blocks until GPU completes.
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = 0;
  vk_check(
      vkCreateFence(device_, &fence_info, nullptr, &enc.fence),
      "vkCreateFence");

  // Begin recording immediately
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(enc.cmd, &begin);
  enc.recording = true;

  return enc;
}

void Device::commit(Stream s) {
  std::unique_lock<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end())
    return;

  CommandEncoder& enc = it->second;
  bool has_sems =
      !enc.wait_semaphores.empty() || !enc.signal_semaphores.empty();
  bool has_handlers = !enc.completion_handlers.empty();
  bool has_temps = !enc.temporaries.empty();

  if (!enc.recording ||
      (enc.op_count == 0 && !has_sems && !has_handlers && !has_temps)) {
    return;
  }
  vkEndCommandBuffer(enc.cmd);

  std::vector<VkSemaphore> wait_sems;
  std::vector<uint64_t> wait_vals;
  std::vector<VkPipelineStageFlags> wait_stages;
  for (auto& p : enc.wait_semaphores) {
    wait_sems.push_back(p.first);
    wait_vals.push_back(p.second);
    wait_stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }
  std::vector<VkSemaphore> signal_sems;
  std::vector<uint64_t> signal_vals;
  for (auto& p : enc.signal_semaphores) {
    signal_sems.push_back(p.first);
    signal_vals.push_back(p.second);
  }

  enc.wait_semaphores.clear();
  enc.signal_semaphores.clear();

  VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.waitSemaphoreValueCount = wait_vals.size();
  timeline_info.pWaitSemaphoreValues = wait_vals.data();
  timeline_info.signalSemaphoreValueCount = signal_vals.size();
  timeline_info.pSignalSemaphoreValues = signal_vals.data();

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = &timeline_info;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &enc.cmd;
  submit.waitSemaphoreCount = wait_sems.size();
  submit.pWaitSemaphores = wait_sems.data();
  submit.pWaitDstStageMask = wait_stages.data();
  submit.signalSemaphoreCount = signal_sems.size();
  submit.pSignalSemaphores = signal_sems.data();

  vk_check(
      vkQueueSubmit(compute_queue_, 1, &submit, enc.fence), "vkQueueSubmit");

  // Run completion handlers async!
  auto handlers = std::move(enc.completion_handlers);
  enc.completion_handlers.clear();
  auto temporaries = std::move(enc.temporaries);
  enc.temporaries.clear();
  enc.op_count = 0;

  struct CommitState {
    VkDevice dev;
    VkCommandPool old_pool;
    VkDescriptorPool old_desc_pool;
    VkFence old_fence;
    std::vector<std::function<void()>> handlers;
    std::vector<std::shared_ptr<array::Data>> temporaries;
  };
  auto* state = new CommitState{
      device_,
      enc.pool,
      enc.desc_pool,
      enc.fence,
      std::move(handlers),
      std::move(temporaries)};

  bool is_first_commit = enc.first_commit;
  if (enc.first_commit) {
    enc.first_commit = false;
  }

  // Create BRAND NEW command pool and fence for the encoder
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex = compute_queue_family_;
  vk_check(
      vkCreateCommandPool(device_, &pool_info, nullptr, &enc.pool),
      "vkCreateCommandPool");

  VkDescriptorPoolSize dpool_size{};
  dpool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dpool_size.descriptorCount = 10000;

  VkDescriptorPoolCreateInfo desc_info{};
  desc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  desc_info.maxSets = 2000;
  desc_info.poolSizeCount = 1;
  desc_info.pPoolSizes = &dpool_size;
  vk_check(
      vkCreateDescriptorPool(device_, &desc_info, nullptr, &enc.desc_pool),
      "vkCreateDescriptorPool");

  VkCommandBufferAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc.commandPool = enc.pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  vk_check(
      vkAllocateCommandBuffers(device_, &alloc, &enc.cmd),
      "vkAllocCommandBuffers");

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vk_check(
      vkCreateFence(device_, &fence_info, nullptr, &enc.fence),
      "vkCreateFence");

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(enc.cmd, &begin);
  enc.recording = true;

  // UNLOCK MUTEX before waiting/destroying to prevent deadlocks!
  lk.unlock();

  if (is_first_commit) {
    // --- MOLTENVK FIRST-COMMIT WARMUP ---
    // Wait synchronously for the first execution to finish.
    vkWaitForFences(state->dev, 1, &state->old_fence, VK_TRUE, UINT64_MAX);

    state->temporaries.clear();
    for (auto& h : state->handlers) {
      if (h)
        h();
    }

    vkDestroyCommandPool(state->dev, state->old_pool, nullptr);
    vkDestroyDescriptorPool(state->dev, state->old_desc_pool, nullptr);
    vkDestroyFence(state->dev, state->old_fence, nullptr);
    delete state;
  } else {
    // Clean up finished threads
    commit_threads_.erase(
        std::remove_if(
            commit_threads_.begin(),
            commit_threads_.end(),
            [](const std::thread& t) {
              // We can't actually do non-blocking check on std::thread easily without a future.
              // Just keeping them in the vector for now until destruction.
              // Actually, maybe a cleaner approach is possible, but we don't have many commits that
              // are totally detached like this, or we do.
              return false;
            }),
        commit_threads_.end());

    commit_threads_.emplace_back(
        [](CommitState* s) {
          vkWaitForFences(s->dev, 1, &s->old_fence, VK_TRUE, UINT64_MAX);

          try {
            s->temporaries.clear();
          } catch (const std::exception& e) {
            fprintf(
                stderr, "[BACKGROUND] temporaries exception: %s\n", e.what());
            fflush(stderr);
          }

          for (auto& h : s->handlers) {
            if (h) {
              try {
                h();
              } catch (const std::exception& e) {
                fprintf(stderr, "[BACKGROUND] Exception: %s\n", e.what());
                fflush(stderr);
              }
            }
          }

          vkDestroyCommandPool(s->dev, s->old_pool, nullptr);
          vkDestroyDescriptorPool(s->dev, s->old_desc_pool, nullptr);
          vkDestroyFence(s->dev, s->old_fence, nullptr);

          delete s;
        },
        state);
  }
}

void Device::synchronize(Stream s) {
  // Commit any pending work for the stream first
  commit(s);

  // Physically block the CPU thread until the Vulkan compute queue is fully
  // idle. This is critical for CPU fallbacks (e.g. Gather::eval_gpu) which need
  // to safely read GPU memory that was written by previously submitted command
  // buffers.
  std::lock_guard<std::mutex> lk(mutex_);
  vkQueueWaitIdle(compute_queue_);
}

// ────────────────────────────────────────────────────────────────────────────
// Pipeline management: load SPIR-V and create cached compute pipelines
// ────────────────────────────────────────────────────────────────────────────

VkPipeline Device::get_pipeline(
    const std::string& name,
    VkPipelineLayout& layout_out,
    VkDescriptorSetLayout& ds_layout_out,
    uint32_t num_bindings,
    uint32_t push_constant_size,
    const VkSpecializationInfo* specialization_info) {
  std::lock_guard<std::mutex> lk(mutex_);

  auto it = pipeline_map_.find(name);
  if (it != pipeline_map_.end()) {
    layout_out = it->second.layout;
    ds_layout_out = it->second.ds_layout;
    return it->second.pipeline;
  }

  // Build descriptor set layout (all storage buffers)
  std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
  for (uint32_t i = 0; i < num_bindings; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo ds_info{};
  ds_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ds_info.bindingCount = num_bindings;
  ds_info.pBindings = bindings.data();

  VkDescriptorSetLayout ds_layout;
  vk_check(
      vkCreateDescriptorSetLayout(device_, &ds_info, nullptr, &ds_layout),
      "vkCreateDescriptorSetLayout");

  // Build pipeline layout with push constants
  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc_range.offset = 0;
  pc_range.size = push_constant_size > 0 ? push_constant_size : 128;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &ds_layout;
  layout_info.pushConstantRangeCount = push_constant_size > 0 ? 1 : 0;
  layout_info.pPushConstantRanges =
      push_constant_size > 0 ? &pc_range : nullptr;

  VkPipelineLayout layout;
  vk_check(
      vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout),
      "vkCreatePipelineLayout");

  // Load SPIR-V
  const char* env_path = std::getenv("MLX_VULKAN_PATH");
  std::string base_path = env_path ? std::string(env_path) + "/" : std::string(VULKAN_KERNELS_PATH);
  std::string spv_path = base_path + name + ".spv";
  std::vector<uint32_t> code;
  try {
    code = read_spirv(spv_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX Vulkan] WARNING: %s\n", e.what());
    // Return null pipeline — callers should fall back to CPU
    pipeline_map_[name] = {VK_NULL_HANDLE, layout, ds_layout};
    layout_out = layout;
    ds_layout_out = ds_layout;
    return VK_NULL_HANDLE;
  }

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = code.size() * sizeof(uint32_t);
  shader_info.pCode = code.data();

  VkShaderModule shader_module;
  vk_check(
      vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module),
      "vkCreateShaderModule");

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.layout = layout;
  pipeline_info.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  const VkSpecializationInfo* final_spec = specialization_info;
  if (!final_spec) {
    final_spec = get_default_specialization_info(*this);
  }

  pipeline_info.stage.module = shader_module;
  pipeline_info.stage.pName = "main";
  pipeline_info.stage.pSpecializationInfo = final_spec;

  VkPipeline pipeline;
  VkResult res = vkCreateComputePipelines(
      device_, pipeline_cache_, 1, &pipeline_info, nullptr, &pipeline);

  vkDestroyShaderModule(device_, shader_module, nullptr);

  if (res != VK_SUCCESS) {
    fprintf(
        stderr,
        "[MLX Vulkan] WARNING: failed to create pipeline '%s'\n",
        name.c_str());
    pipeline = VK_NULL_HANDLE;
  } else {
    // MOLTENVK BUG WORKAROUND: The very first VkCommandBuffer that utilizes a
    // newly compiled MTLComputePipelineState will be silently dropped by the
    // Apple Silicon driver. To absorb this fault, we immediately allocate a
    // transient command buffer, bind the new pipeline, dispatch a single
    // workgroup, and submit it.
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = compute_queue_family_;
    VkCommandPool pool;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &pool) ==
        VK_SUCCESS) {
      VkCommandBufferAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      alloc_info.commandPool = pool;
      alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_info.commandBufferCount = 1;
      VkCommandBuffer cmd;
      if (vkAllocateCommandBuffers(device_, &alloc_info, &cmd) == VK_SUCCESS) {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = num_bindings > 0 ? num_bindings : 1;
        VkDescriptorPoolCreateInfo desc_pool_info{};
        desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        desc_pool_info.poolSizeCount = 1;
        desc_pool_info.pPoolSizes = &pool_size;
        desc_pool_info.maxSets = 1;
        VkDescriptorPool desc_pool;
        if (vkCreateDescriptorPool(
                device_, &desc_pool_info, nullptr, &desc_pool) == VK_SUCCESS) {
          VkDescriptorSetAllocateInfo ds_alloc{};
          ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
          ds_alloc.descriptorPool = desc_pool;
          ds_alloc.descriptorSetCount = 1;
          ds_alloc.pSetLayouts = &ds_layout;
          VkDescriptorSet ds;
          if (vkAllocateDescriptorSets(device_, &ds_alloc, &ds) == VK_SUCCESS) {
            std::vector<VkDescriptorBufferInfo> buf_infos(num_bindings);
            std::vector<VkWriteDescriptorSet> writes(num_bindings);
            for (uint32_t i = 0; i < num_bindings; i++) {
              buf_infos[i].buffer = dummy_buffer_;
              buf_infos[i].offset = 0;
              buf_infos[i].range = VK_WHOLE_SIZE;
              writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              writes[i].dstSet = ds;
              writes[i].dstBinding = i;
              writes[i].descriptorCount = 1;
              writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
              writes[i].pBufferInfo = &buf_infos[i];
            }
            if (num_bindings > 0) {
              vkUpdateDescriptorSets(
                  device_, num_bindings, writes.data(), 0, nullptr);
            }

            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

            if (push_constant_size > 0) {
              std::vector<uint8_t> zeros(push_constant_size, 0);
              vkCmdPushConstants(
                  cmd,
                  layout,
                  VK_SHADER_STAGE_COMPUTE_BIT,
                  0,
                  push_constant_size,
                  zeros.data());
            }

            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                layout,
                0,
                1,
                &ds,
                0,
                nullptr);
            vkCmdDispatch(cmd, 1, 1, 1);

            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &barrier,
                0,
                nullptr,
                0,
                nullptr);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            vkQueueSubmit(compute_queue_, 1, &submit, VK_NULL_HANDLE);
            // Wait for it to finish so the pipeline is fully initialized in the
            // driver
            vkQueueWaitIdle(compute_queue_);
          }
          vkDestroyDescriptorPool(device_, desc_pool, nullptr);
        }
      }
      vkDestroyCommandPool(device_, pool, nullptr);
    }
  }

  pipeline_map_[name] = {pipeline, layout, ds_layout};
  layout_out = layout;
  ds_layout_out = ds_layout;
  return pipeline;
}

// ────────────────────────────────────────────────────────────────────────────
// Descriptor set allocation
// ────────────────────────────────────────────────────────────────────────────

VkDescriptorSet Device::alloc_descriptor_set(
    Stream s,
    VkDescriptorSetLayout layout) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    throw std::runtime_error(
        "[Device::alloc_descriptor] Invalid stream encoder");
  }

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = it->second.desc_pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &layout;

  VkDescriptorSet ds;
  VkResult res = vkAllocateDescriptorSets(device_, &alloc_info, &ds);
  vk_check(res, "vkAllocateDescriptorSets");
  return ds;
}

// ────────────────────────────────────────────────────────────────────────────
// Singleton accessor
// ────────────────────────────────────────────────────────────────────────────

Device& device(mlx::core::Device /*d*/) {
  static Device instance;
  return instance;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device(s.device).get_command_encoder(s);
}

void Device::add_temporary(Stream s, const array& arr) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    it->second.temporaries.push_back(arr.data_shared_ptr());
  }
}

void Device::add_completed_handler(Stream s, std::function<void()> handler) {
  if (!handler)
    return;
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    it->second.completion_handlers.push_back(std::move(handler));
  }
}

void Device::add_wait_semaphore(Stream s, VkSemaphore sem, uint64_t val) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    it->second.wait_semaphores.push_back({sem, val});
  }
}

void Device::add_signal_semaphore(Stream s, VkSemaphore sem, uint64_t val) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    it->second.signal_semaphores.push_back({sem, val});
  }
}

bool Device::needs_commit(Stream s) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = encoders_.find(s.index);
  if (it != encoders_.end()) {
    return it->second.op_count > 0 && it->second.op_count % 64 == 0;
  }
  return false;
}

} // namespace mlx::core::vulkan
