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
  create_logical_device();
  create_vma();
  create_pipeline_cache();
  create_descriptor_pool();
}

Device::~Device() {
  vkDeviceWaitIdle(device_);

  // Save pipeline cache
  save_pipeline_cache();

  // Destroy pipelines
  for (auto& [name, entry] : pipeline_map_) {
    if (entry.pipeline    != VK_NULL_HANDLE) vkDestroyPipeline(device_, entry.pipeline, nullptr);
    if (entry.layout      != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, entry.layout, nullptr);
    if (entry.ds_layout   != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, entry.ds_layout, nullptr);
  }

  // Destroy encoders
  for (auto& [idx, enc] : encoders_) {
    for (auto& handler : enc.completion_handlers) {
      handler();
    }
    enc.completion_handlers.clear();
    
    if (enc.fence != VK_NULL_HANDLE) vkDestroyFence(device_, enc.fence, nullptr);
    if (enc.pool  != VK_NULL_HANDLE) vkDestroyCommandPool(device_, enc.pool, nullptr);
  }

  if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
  if (pipeline_cache_  != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
  if (vma_allocator_   != VK_NULL_HANDLE) vmaDestroyAllocator(vma_allocator_);

#ifdef MLX_VULKAN_VALIDATION
  if (debug_messenger_ != VK_NULL_HANDLE) {
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
    if (fn) fn(instance_, debug_messenger_, nullptr);
  }
#endif

  if (device_   != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// Instance creation
// ────────────────────────────────────────────────────────────────────────────

void Device::init_instance() {
  VkApplicationInfo app_info{};
  app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName   = "MLX";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName        = "MLX Vulkan Backend";
  app_info.apiVersion         = VK_API_VERSION_1_2;

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
  inst_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
#if defined(__APPLE__)
  inst_info.flags                   |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#endif
  inst_info.pApplicationInfo        = &app_info;
  inst_info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
  inst_info.ppEnabledLayerNames     = layers.data();
  inst_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
  inst_info.ppEnabledExtensionNames = extensions.data();

  vk_check(vkCreateInstance(&inst_info, nullptr, &instance_),
           "vkCreateInstance");

#ifdef MLX_VULKAN_VALIDATION
  VkDebugUtilsMessengerCreateInfoEXT dbg{};
  dbg.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dbg.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  dbg.pfnUserCallback = debug_callback;

  auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
      vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
  if (fn) fn(instance_, &dbg, nullptr, &debug_messenger_);
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
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score += 1000;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    score += 50;
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
// Logical device + compute queue
// ────────────────────────────────────────────────────────────────────────────

void Device::create_logical_device() {
  // Find compute queue family
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qf_props(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

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
  queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = compute_queue_family_;
  queue_info.queueCount       = 1;
  queue_info.pQueuePriorities = &priority;

  // Required device extensions
  std::vector<const char*> dev_extensions = {
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
  };

  // Optional: check for external memory host (zero-copy for APUs)
  uint32_t ext_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> avail_exts(ext_count);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, avail_exts.data());

  auto has_ext = [&](const char* name) {
    for (auto& e : avail_exts) {
      if (strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
  };

  if (has_ext(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    dev_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    dev_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    fprintf(stderr, "[MLX Vulkan] Zero-copy (VK_EXT_external_memory_host) available\n");
  }

  // Enable timeline semaphore feature
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{};
  timeline_feat.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timeline_feat.timelineSemaphore = VK_TRUE;

  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &timeline_feat;
  // Basic float64 for double precision (optional)
  features2.features.shaderFloat64 = VK_FALSE;

  VkDeviceCreateInfo dev_info{};
  dev_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dev_info.pNext                   = &features2;
  dev_info.queueCreateInfoCount    = 1;
  dev_info.pQueueCreateInfos       = &queue_info;
  dev_info.enabledExtensionCount   = static_cast<uint32_t>(dev_extensions.size());
  dev_info.ppEnabledExtensionNames = dev_extensions.data();

  vk_check(vkCreateDevice(physical_device_, &dev_info, nullptr, &device_),
           "vkCreateDevice");

  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

// ────────────────────────────────────────────────────────────────────────────
// VMA allocator
// ────────────────────────────────────────────────────────────────────────────

void Device::create_vma() {
  VmaAllocatorCreateInfo vma_info{};
  vma_info.physicalDevice = physical_device_;
  vma_info.device         = device_;
  vma_info.instance       = instance_;
  vma_info.vulkanApiVersion = VK_API_VERSION_1_2;

  vk_check(vmaCreateAllocator(&vma_info, &vma_allocator_), "vmaCreateAllocator");
}

// ────────────────────────────────────────────────────────────────────────────
// Pipeline cache (persisted to disk across runs)
// ────────────────────────────────────────────────────────────────────────────

// Bump this version whenever shader push-constant layouts change.
// Prevents MoltenVK from loading stale binary cache blobs.
static constexpr int kPipelineCacheVersion = 4; // bumped: binary.comp push constant semantics changed (a_is_scalar→a_size)

static std::string pipeline_cache_path() {
  const char* home = std::getenv("HOME");
  if (!home) home = "/tmp";
  return std::string(home) + "/.cache/mlx_vulkan_pipeline_cache_v"
       + std::to_string(kPipelineCacheVersion) + ".bin";
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
  cache_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  cache_info.initialDataSize = cache_data.size();
  cache_info.pInitialData    = cache_data.empty() ? nullptr : cache_data.data();

  vk_check(vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_),
           "vkCreatePipelineCache");
}

void Device::save_pipeline_cache() {
  if (pipeline_cache_ == VK_NULL_HANDLE) return;
  size_t size = 0;
  vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr);
  if (size == 0) return;
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
  if (f) f.write(data.data(), data.size());
}

// ────────────────────────────────────────────────────────────────────────────
// Descriptor pool
// ────────────────────────────────────────────────────────────────────────────

void Device::create_descriptor_pool() {
  // Pool supports 4096 storage buffer descriptors and 512 sets
  VkDescriptorPoolSize pool_size{};
  pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 4096;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = 512;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes    = &pool_size;

  vk_check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
           "vkCreateDescriptorPool");
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
  pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = compute_queue_family_;
  pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vk_check(vkCreateCommandPool(device_, &pool_info, nullptr, &enc.pool),
           "vkCreateCommandPool");

  // Allocate initial command buffer
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool        = enc.pool;
  alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  vk_check(vkAllocateCommandBuffers(device_, &alloc_info, &enc.cmd),
           "vkAllocateCommandBuffers");

  // Fence for CPU-GPU sync
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled (nothing pending)
  vk_check(vkCreateFence(device_, &fence_info, nullptr, &enc.fence),
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
  if (it == encoders_.end()) return;

  CommandEncoder& enc = it->second;
  if (!enc.recording || enc.op_count == 0) return;

  vkEndCommandBuffer(enc.cmd);

  // Wait for previous submission to finish
  vkWaitForFences(device_, 1, &enc.fence, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &enc.fence);

  VkSubmitInfo submit{};
  submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &enc.cmd;

  vk_check(vkQueueSubmit(compute_queue_, 1, &submit, enc.fence),
           "vkQueueSubmit");

  // Run completion handlers async (after fence signals)
  // Wait immediately then run handlers
  vkWaitForFences(device_, 1, &enc.fence, VK_TRUE, UINT64_MAX);

  auto handlers = std::move(enc.completion_handlers);
  enc.completion_handlers.clear();
  enc.temporaries.clear();
  enc.op_count = 0;

  // Reset and begin new command buffer
  vkResetCommandPool(device_, enc.pool, 0);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(enc.cmd, &begin);
  enc.recording = true;

  // IMPORTANT: Drop lock before executing handlers, as handlers might schedule
  // new MLX tasks which can recursively call Device/Allocator methods and deadlock!
  lk.unlock();

  for (auto& h : handlers) h();
}

void Device::synchronize(Stream s) {
  // Commit any pending work for the stream first
  commit(s);
  
  // Physically block the CPU thread until the Vulkan compute queue is fully idle.
  // This is critical for CPU fallbacks (e.g. Gather::eval_gpu) which need to safely 
  // read GPU memory that was written by previously submitted command buffers.
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
    uint32_t push_constant_size) {

  std::lock_guard<std::mutex> lk(mutex_);

  auto it = pipeline_map_.find(name);
  if (it != pipeline_map_.end()) {
    layout_out    = it->second.layout;
    ds_layout_out = it->second.ds_layout;
    return it->second.pipeline;
  }

  // Build descriptor set layout (all storage buffers)
  std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
  for (uint32_t i = 0; i < num_bindings; i++) {
    bindings[i].binding         = i;
    bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo ds_info{};
  ds_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ds_info.bindingCount = num_bindings;
  ds_info.pBindings    = bindings.data();

  VkDescriptorSetLayout ds_layout;
  vk_check(vkCreateDescriptorSetLayout(device_, &ds_info, nullptr, &ds_layout),
           "vkCreateDescriptorSetLayout");

  // Build pipeline layout with push constants
  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc_range.offset     = 0;
  pc_range.size       = push_constant_size > 0 ? push_constant_size : 128;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount         = 1;
  layout_info.pSetLayouts            = &ds_layout;
  layout_info.pushConstantRangeCount = push_constant_size > 0 ? 1 : 0;
  layout_info.pPushConstantRanges    = push_constant_size > 0 ? &pc_range : nullptr;

  VkPipelineLayout layout;
  vk_check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout),
           "vkCreatePipelineLayout");

  // Load SPIR-V
  std::string spv_path = std::string(VULKAN_KERNELS_PATH) + name + ".spv";
  std::vector<uint32_t> code;
  try {
    code = read_spirv(spv_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX Vulkan] WARNING: %s\n", e.what());
    // Return null pipeline — callers should fall back to CPU
    pipeline_map_[name] = {VK_NULL_HANDLE, layout, ds_layout};
    layout_out    = layout;
    ds_layout_out = ds_layout;
    return VK_NULL_HANDLE;
  }

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = code.size() * sizeof(uint32_t);
  shader_info.pCode    = code.data();

  VkShaderModule shader_module;
  vk_check(vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module),
           "vkCreateShaderModule");

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.layout = layout;
  pipeline_info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = shader_module;
  pipeline_info.stage.pName  = "main";

  VkPipeline pipeline;
  VkResult res = vkCreateComputePipelines(
      device_, pipeline_cache_, 1, &pipeline_info, nullptr, &pipeline);

  vkDestroyShaderModule(device_, shader_module, nullptr);

  if (res != VK_SUCCESS) {
    fprintf(stderr, "[MLX Vulkan] WARNING: failed to create pipeline '%s'\n", name.c_str());
    pipeline = VK_NULL_HANDLE;
  }

  pipeline_map_[name] = {pipeline, layout, ds_layout};
  layout_out    = layout;
  ds_layout_out = ds_layout;
  return pipeline;
}

// ────────────────────────────────────────────────────────────────────────────
// Descriptor set allocation
// ────────────────────────────────────────────────────────────────────────────

VkDescriptorSet Device::alloc_descriptor_set(VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool     = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts        = &layout;

  VkDescriptorSet ds;
  VkResult res = vkAllocateDescriptorSets(device_, &alloc_info, &ds);
  if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL) {
    // Reset pool and retry (lose all existing sets — safe if we commit first)
    vkResetDescriptorPool(device_, descriptor_pool_, 0);
    vk_check(vkAllocateDescriptorSets(device_, &alloc_info, &ds),
             "vkAllocateDescriptorSets (after pool reset)");
  } else {
    vk_check(res, "vkAllocateDescriptorSets");
  }
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

} // namespace mlx::core::vulkan
