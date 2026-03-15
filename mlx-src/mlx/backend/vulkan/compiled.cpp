// Copyright © 2025 Apple Inc.
#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace mlx::core {

namespace {

// In-memory SPIR-V cache to avoid re-invoking glslc for identical kernels
std::mutex spirv_cache_mutex_;
std::unordered_map<std::string, std::vector<uint32_t>> spirv_cache_;

// Helper to compile dynamic GLSL to SPIR-V (with caching)
std::vector<uint32_t> compile_glsl_to_spirv(const std::string& kernel_name, const std::string& glsl_source) {
  {
    std::lock_guard<std::mutex> lk(spirv_cache_mutex_);
    auto it = spirv_cache_.find(kernel_name);
    if (it != spirv_cache_.end()) {
      return it->second;
    }
  }

  std::string tmp_dir = "/tmp/mlx_vulkan_jit/";
  std::filesystem::create_directories(tmp_dir);
  std::string comp_path = tmp_dir + kernel_name + ".comp";
  std::string spv_path = tmp_dir + kernel_name + ".spv";
  
  std::ofstream out(comp_path);
  out << glsl_source;
  out.close();
  
  std::string cmd = "glslc -c " + comp_path + " -o " + spv_path + " -fshader-stage=compute --target-env=vulkan1.2";
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("[Compiled::eval_gpu] Failed to compile dynamic kernel with glslc.");
  }
  
  std::ifstream spv_in(spv_path, std::ios::binary | std::ios::ate);
  if (!spv_in) {
    throw std::runtime_error("[Compiled::eval_gpu] Failed to read compiled SPIR-V.");
  }
  size_t size = spv_in.tellg();
  spv_in.seekg(0);
  std::vector<uint32_t> code(size / sizeof(uint32_t));
  spv_in.read(reinterpret_cast<char*>(code.data()), size);

  {
    std::lock_guard<std::mutex> lk(spirv_cache_mutex_);
    spirv_cache_[kernel_name] = code;
  }

  return code;
}


std::string to_glsl_op(const std::string& pr, const std::vector<array>& inputs, const std::vector<std::string>& input_names) {
  if (inputs.size() == 1) {
    const auto& in0 = input_names[0];
    if (pr == "Abs") return "abs(" + in0 + ")";
    if (pr == "Negative") return "-(" + in0 + ")";
    if (pr == "Sign") return "sign(" + in0 + ")";
    if (pr == "Round") return "round(" + in0 + ")";
    if (pr == "Floor") return "floor(" + in0 + ")";
    if (pr == "Ceil") return "ceil(" + in0 + ")";
    if (pr == "Exp") return "exp(" + in0 + ")";
    if (pr == "Expm1") return "(exp(" + in0 + ") - 1.0)";
    if (pr == "Log") return "log(" + in0 + ")";
    if (pr == "Log1p") return "log(1.0 + " + in0 + ")";
    if (pr == "Sin") return "sin(" + in0 + ")";
    if (pr == "Cos") return "cos(" + in0 + ")";
    if (pr == "Tan") return "tan(" + in0 + ")";
    if (pr == "ArcSin") return "asin(" + in0 + ")";
    if (pr == "ArcCos") return "acos(" + in0 + ")";
    if (pr == "ArcTan") return "atan(" + in0 + ")";
    if (pr == "Sinh") return "sinh(" + in0 + ")";
    if (pr == "Cosh") return "cosh(" + in0 + ")";
    if (pr == "Tanh") return "tanh(" + in0 + ")";
    if (pr == "Sqrt") return "sqrt(" + in0 + ")";
    if (pr == "Rsqrt") return "inversesqrt(" + in0 + ")";
    if (pr == "LogicalNot") return "(" + in0 + " == 0 ? 1 : 0)";
    if (pr == "Square") return "((" + in0 + ") * (" + in0 + "))";
  } else if (inputs.size() == 2) {
    const auto& in0 = input_names[0];
    const auto& in1 = input_names[1];
    if (pr == "Add") return in0 + " + " + in1;
    if (pr == "Subtract") return in0 + " - " + in1;
    if (pr == "Multiply") return in0 + " * " + in1;
    if (pr == "Divide") return in0 + " / " + in1;
    if (pr == "Maximum") return "max(" + in0 + ", " + in1 + ")";
    if (pr == "Minimum") return "min(" + in0 + ", " + in1 + ")";
    if (pr == "Power") return "pow(" + in0 + ", " + in1 + ")";
    if (pr == "Remainder") return "mod(" + in0 + ", " + in1 + ")";
    if (pr == "Equal") return "(" + in0 + " == " + in1 + " ? 1 : 0)";
    if (pr == "NotEqual") return "(" + in0 + " != " + in1 + " ? 1 : 0)";
    if (pr == "Greater") return "(" + in0 + " > " + in1 + " ? 1 : 0)";
    if (pr == "GreaterEqual") return "(" + in0 + " >= " + in1 + " ? 1 : 0)";
    if (pr == "Less") return "(" + in0 + " < " + in1 + " ? 1 : 0)";
    if (pr == "LessEqual") return "(" + in0 + " <= " + in1 + " ? 1 : 0)";
    if (pr == "LogicalAnd") return "((" + in0 + " != 0) && (" + in1 + " != 0) ? 1 : 0)";
    if (pr == "LogicalOr") return "((" + in0 + " != 0) || (" + in1 + " != 0) ? 1 : 0)";
  } else if (inputs.size() == 3) {
    const auto& in0 = input_names[0];
    const auto& in1 = input_names[1];
    const auto& in2 = input_names[2];
    if (pr == "Select") {
      return "(" + in0 + " != 0 ? " + in1 + " : " + in2 + ")";
    }
  }

  throw std::runtime_error("[Compiled::eval_gpu] AST to GLSL unhandled primitive: " + pr);
}

void build_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    int ndim,
    bool dynamic_dims,
    bool use_big_index = false,
    int work_per_thread = 1) {
    
    NodeNamer namer;
    os += "#version 450\n";
    os += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    
    int binding = 0;
    
    // Inputs
    std::vector<std::string> in_names;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (is_constant(i)) continue;
        const auto& x = inputs[i];
        std::string xname = namer.get_name(x);
        in_names.push_back(xname);
        os += fmt::format("layout(set = 0, binding = {}) readonly buffer In{} {{ {} {}[]; }};\n",
                      binding++, i, get_type_string(x.dtype()), xname);
    }
    
    // Outputs
    std::vector<std::string> out_names;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& x = outputs[i];
        std::string xname = namer.get_name(x);
        out_names.push_back(xname);
        os += fmt::format("layout(set = 0, binding = {}) writeonly buffer Out{} {{ {} {}[]; }};\n",
                      binding++, i, get_type_string(x.dtype()), xname);
    }
    
    // Push constants
    os += "layout(push_constant) uniform PushConstants {\n";
    os += "  uint size;\n";
    os += "} pc;\n";
    
    os += "void main() {\n";
    os += "  uint index = gl_GlobalInvocationID.x;\n";
    os += "  if (index >= pc.size) return;\n";
    
    // Reading inputs
    for (int i = 0; i < inputs.size(); ++i) {
        if(is_constant(i)) continue;
        auto xname = namer.get_name(inputs[i]);
        os += fmt::format("  {} tmp_{} = {}[index];\n", get_type_string(inputs[i].dtype()), xname, xname);
    }
    
    // Tape
    for (auto& x : tape) {
        std::string out_type = get_type_string(x.dtype());
        std::string out_name = namer.get_name(x);
        
        std::vector<std::string> inps;
        for (auto& in : x.inputs()) {
            inps.push_back("tmp_" + namer.get_name(in));
        }
        
        if (is_static_cast(x.primitive())) {
           os += fmt::format("  {} tmp_{} = {}({});\n", out_type, out_name, out_type, inps[0]);
        } else {
           std::string expr = to_glsl_op(x.primitive().name(), x.inputs(), inps);
           os += fmt::format("  {} tmp_{} = {};\n", out_type, out_name, expr);
        }
    }
    
    // Write outputs
    for (auto& x : outputs) {
        auto xname = namer.get_name(x);
        os += fmt::format("  {}[index] = tmp_{};\n", xname, xname);
    }
    os += "}\n";
}

} // namespace


void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  
  if (!outputs[0].flags().contiguous) {
    throw std::runtime_error("[Compiled::eval_gpu] Strided Vulkan compilation not implemented yet.");
  }
  
  auto& s = stream();
  auto& d = vulkan::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s);
  
  std::string kernel_name = kernel_lib_ + "_vulkan_contig";
  
  // Pipeline layout out
  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  
  int num_bindings = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (!is_constant_(i)) num_bindings++;
  }
  num_bindings += outputs.size();
  
  std::string glsl;
  build_kernel(
      glsl,
      kernel_name,
      inputs_,
      outputs_,
      tape_,
      is_constant_,
      true, // contiguous
      0,
      false,
      false,
      1
  );
  
  std::vector<uint32_t> spv = compile_glsl_to_spirv(kernel_name, glsl);
  
  VkPipeline pipeline = d.get_pipeline_from_spirv(
          kernel_name,
          spv,
          layout,
          ds_layout,
          num_bindings,
          sizeof(uint32_t) // push constant size
      );
  
  if (pipeline == VK_NULL_HANDLE) {
     throw std::runtime_error("[Compiled::eval_gpu] Vulkan pipeline creation failed");
  }

  // Allocate descriptor set
  VkDescriptorSet ds = d.alloc_descriptor_set(s, ds_layout);
  
  // Update Descriptor sets
  std::vector<VkDescriptorBufferInfo> buf_infos(num_bindings);
  std::vector<VkWriteDescriptorSet> writes(num_bindings);
  int bind_idx = 0;
  
  auto bind_arr = [&](const array& arr) {
    allocator::Buffer buf(const_cast<void*>(arr.data<void>()));
    buf_infos[bind_idx].buffer = vulkan::allocator().vk_buffer(buf);
    buf_infos[bind_idx].offset = 0;
    buf_infos[bind_idx].range = VK_WHOLE_SIZE;
    
    writes[bind_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[bind_idx].dstSet = ds;
    writes[bind_idx].dstBinding = bind_idx;
    writes[bind_idx].descriptorCount = 1;
    writes[bind_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[bind_idx].pBufferInfo = &buf_infos[bind_idx];
    bind_idx++;
  };
  
  for (int i=0; i<inputs.size(); i++) {
    if(!is_constant_(i)) {
      bind_arr(inputs[i]);
    }
  }
  
  // allocate outputs
  compiled_allocate_outputs(inputs, outputs, is_constant_, true);
  
  for (auto& out : outputs) {
     bind_arr(out);
  }
  
  vkUpdateDescriptorSets(d.vk_device(), num_bindings, writes.data(), 0, nullptr);
  
  // Bind pipeline
  vkCmdBindPipeline(compute_encoder.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

  // Bind DS
  vkCmdBindDescriptorSets(
        compute_encoder.cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        layout,
        0,
        1,
        &ds,
        0,
        nullptr);
        
  // Push constants
  uint32_t size = outputs[0].data_size();
  vkCmdPushConstants(
      compute_encoder.cmd,
      layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(uint32_t),
      &size);
      
  // Dispatch
  uint32_t nblock = (size + 255) / 256;
  vkCmdDispatch(compute_encoder.cmd, nblock, 1, 1);
  
  compute_encoder.op_count++;
}

} // namespace mlx::core
