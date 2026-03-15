// Copyright © 2025 Apple Inc.
#include <fmt/format.h>

#include <algorithm>
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace mlx::core {

namespace {

// In-memory SPIR-V cache to avoid re-invoking glslc for identical kernels
std::mutex spirv_cache_mutex_;
std::unordered_map<std::string, std::vector<uint32_t>> spirv_cache_;

struct KernelProfile {
  size_t executions{0};
  size_t tape_size{0};
  std::vector<std::string> ops;
  std::string inputs;
  std::string outputs;
};

std::mutex compile_profile_mutex_;
std::unordered_map<std::string, KernelProfile> compile_profiles_;

bool compile_profile_enabled() {
  static bool enabled = (std::getenv("MLX_LOG_COMPILE_TAPE") != nullptr);
  return enabled;
}

bool compile_profile_cpu_fallback_enabled() {
  static bool enabled =
      (std::getenv("MLX_VULKAN_PROFILE_COMPILE_FALLBACK") != nullptr);
  return enabled;
}

bool is_compiled_float_dtype(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

CopyType compiled_copy_type(const array& src, const array& dst) {
  if (src.shape() == dst.shape() && src.flags().row_contiguous &&
      dst.flags().row_contiguous) {
    return CopyType::Vector;
  }
  return CopyType::General;
}

std::string compiled_glsl_type(Dtype d) {
  if (is_compiled_float_dtype(d)) {
    return "float";
  }
  return get_type_string(d);
}

std::string shape_to_string(const Shape& shape) {
  if (shape.empty()) {
    return "scalar";
  }
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      out += "x";
    }
  }
  out += "]";
  return out;
}

std::string array_summary(const array& arr) {
  return fmt::format(
      "{}{}{}",
      get_type_string(arr.dtype()),
      shape_to_string(arr.shape()),
      arr.flags().contiguous ? ":contig" : ":strided");
}

std::string arrays_summary(const std::vector<array>& arrays) {
  std::string out = "[";
  for (size_t i = 0; i < arrays.size(); ++i) {
    out += array_summary(arrays[i]);
    if (i + 1 < arrays.size()) {
      out += ", ";
    }
  }
  out += "]";
  return out;
}

std::vector<std::string> tape_ops(const std::vector<array>& tape) {
  std::vector<std::string> ops;
  ops.reserve(tape.size());
  for (auto& x : tape) {
    if (x.has_primitive()) {
      ops.push_back(x.primitive().name());
    }
  }
  return ops;
}

void dump_compile_profile_summary() {
  if (!compile_profile_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lk(compile_profile_mutex_);
  if (compile_profiles_.empty()) {
    return;
  }

  size_t total_dispatches = 0;
  std::unordered_map<std::string, size_t> op_dispatch_counts;
  std::unordered_map<std::string, size_t> op_kernel_counts;
  std::vector<std::pair<std::string, KernelProfile>> kernels;
  kernels.reserve(compile_profiles_.size());

  for (const auto& [kernel_name, profile] : compile_profiles_) {
    total_dispatches += profile.executions;
    kernels.push_back({kernel_name, profile});
    std::unordered_map<std::string, size_t> seen_ops;
    for (const auto& op : profile.ops) {
      op_dispatch_counts[op] += profile.executions;
      seen_ops[op]++;
    }
    for (const auto& [op, _] : seen_ops) {
      op_kernel_counts[op]++;
    }
  }

  std::sort(
      kernels.begin(),
      kernels.end(),
      [](const auto& a, const auto& b) {
        if (a.second.executions != b.second.executions) {
          return a.second.executions > b.second.executions;
        }
        return a.first < b.first;
      });

  std::vector<std::pair<std::string, size_t>> ranked_ops(
      op_dispatch_counts.begin(), op_dispatch_counts.end());
  std::sort(
      ranked_ops.begin(),
      ranked_ops.end(),
      [](const auto& a, const auto& b) {
        if (a.second != b.second) {
          return a.second > b.second;
        }
        return a.first < b.first;
      });

  std::fprintf(
      stderr,
      "[MLX Vulkan][compile profile] summary kernels=%zu dispatches=%zu\n",
      kernels.size(),
      total_dispatches);
  for (const auto& [op, dispatches] : ranked_ops) {
    std::fprintf(
        stderr,
        "  op=%s dispatches=%zu unique_kernels=%zu\n",
        op.c_str(),
        dispatches,
        op_kernel_counts[op]);
  }
  for (const auto& [kernel_name, profile] : kernels) {
    std::fprintf(
        stderr,
        "  kernel=%s dispatches=%zu tape=%zu inputs=%s outputs=%s ops=",
        kernel_name.c_str(),
        profile.executions,
        profile.tape_size,
        profile.inputs.c_str(),
        profile.outputs.c_str());
    for (size_t i = 0; i < profile.ops.size(); ++i) {
      std::fprintf(stderr, "%s%s", i == 0 ? "" : " -> ", profile.ops[i].c_str());
    }
    std::fprintf(stderr, "\n");
  }
  std::fflush(stderr);
}

void maybe_log_compile_profile(
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape) {
  if (!compile_profile_enabled()) {
    return;
  }

  static bool registered = []() {
    std::atexit(dump_compile_profile_summary);
    return true;
  }();
  (void)registered;

  auto ops = tape_ops(tape);
  auto input_desc = arrays_summary(inputs);
  auto output_desc = arrays_summary(outputs);
  auto profile_key =
      fmt::format("{}|{}|{}", kernel_name, input_desc, output_desc);
  std::lock_guard<std::mutex> lk(compile_profile_mutex_);
  auto [it, inserted] = compile_profiles_.try_emplace(
      profile_key,
      KernelProfile{
          0,
          tape.size(),
          ops,
          std::move(input_desc),
          std::move(output_desc)});
  it->second.executions++;

  if (inserted) {
    std::fprintf(
        stderr,
        "[MLX Vulkan][compile profile] kernel=%s tape=%zu inputs=%s outputs=%s ops=",
        kernel_name.c_str(),
        it->second.tape_size,
        it->second.inputs.c_str(),
        it->second.outputs.c_str());
    for (size_t i = 0; i < it->second.ops.size(); ++i) {
      std::fprintf(stderr, "%s%s", i == 0 ? "" : " -> ", it->second.ops[i].c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
  }
}

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
    if (pr == "Broadcast") return in0;
    if (pr == "Abs") return "abs(" + in0 + ")";
    if (pr == "Negative") return "-(" + in0 + ")";
    if (pr == "Sign") return "sign(" + in0 + ")";
    if (pr == "Round") return "round(" + in0 + ")";
    if (pr == "Floor") return "floor(" + in0 + ")";
    if (pr == "Ceil") return "ceil(" + in0 + ")";
    if (pr == "Exp") return "exp(" + in0 + ")";
    if (pr == "Expm1") return "(exp(" + in0 + ") - 1.0)";
    if (pr == "Sigmoid") return "(1.0 / (1.0 + exp(-(" + in0 + "))))";
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
    if (pr == "LogAddExp") {
      return "(max(" + in0 + ", " + in1 + ") + log(exp(" + in0 +
          " - max(" + in0 + ", " + in1 + ")) + exp(" + in1 + " - max(" +
          in0 + ", " + in1 + "))))";
    }
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

struct TerminalSumReduction {
  Shape source_shape;
  Shape output_shape;
  size_t reduce_size{0};
};

std::optional<TerminalSumReduction> analyze_terminal_sum_reduction(
    const std::vector<array>& outputs,
    const std::vector<array>& tape) {
  if (outputs.size() != 1 || tape.empty()) {
    return std::nullopt;
  }
  const auto& reduce_node = outputs[0];
  if (reduce_node.id() != tape.back().id() || !reduce_node.has_primitive()) {
    return std::nullopt;
  }
  const auto& reduce_primitive = reduce_node.primitive();
  if (typeid(reduce_primitive) != typeid(Reduce)) {
    return std::nullopt;
  }
  for (size_t i = 0; i + 1 < tape.size(); ++i) {
    if (tape[i].has_primitive()) {
      const auto& p = tape[i].primitive();
      if (typeid(p) == typeid(Reduce)) {
        return std::nullopt;
      }
    }
  }

  const auto& reduce = static_cast<const Reduce&>(reduce_primitive);
  auto [reduce_type, axes] = reduce.state();
  if (reduce_type != Reduce::Sum || axes.size() != 1) {
    return std::nullopt;
  }

  const auto& reduce_input = reduce_node.inputs()[0];
  int axis = axes[0];
  if (axis != reduce_input.ndim() - 1 || reduce_input.ndim() != reduce_node.ndim() ||
      reduce_input.shape(axis) <= 1) {
      return std::nullopt;
  }

  auto expected_output_shape = reduce_input.shape();
  expected_output_shape[axis] = 1;
  if (reduce_node.shape() != expected_output_shape) {
      return std::nullopt;
  }

  return TerminalSumReduction{
      reduce_input.shape(),
      reduce_node.shape(),
      static_cast<size_t>(reduce_input.shape(axis))};
}

Strides compiled_broadcast_strides_for_shape(
    const array& x,
    const Shape& target_shape) {
  Strides xstrides;
  xstrides.reserve(target_shape.size());
  size_t j = 0;
  for (; j < target_shape.size() - x.ndim(); ++j) {
    xstrides.push_back(0);
  }
  for (int i = 0; i < x.ndim(); ++i, ++j) {
    xstrides.push_back(x.shape(i) == 1 ? 0 : x.strides()[i]);
  }
  return xstrides;
}

bool compiled_reduction_inputs_contiguous(
    const std::vector<array>& inputs,
    const Shape& source_shape,
    const std::function<bool(size_t)>& is_constant) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i) || is_scalar(inputs[i])) {
      continue;
    }
    if (!inputs[i].flags().row_contiguous || inputs[i].shape() != source_shape) {
      return false;
    }
  }
  return true;
}

std::vector<int32_t> compiled_reduction_metadata(
    const std::vector<array>& inputs,
    const Shape& source_shape,
    const std::function<bool(size_t)>& is_constant) {
  std::vector<int32_t> meta;
  size_t non_scalar_inputs = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!is_constant(i) && !is_scalar(inputs[i])) {
      non_scalar_inputs++;
    }
  }
  meta.reserve(source_shape.size() * (1 + non_scalar_inputs));
  for (auto dim : source_shape) {
    meta.push_back(static_cast<int32_t>(dim));
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i) || is_scalar(inputs[i])) {
      continue;
    }
    auto strides = compiled_broadcast_strides_for_shape(inputs[i], source_shape);
    for (auto stride : strides) {
      meta.push_back(static_cast<int32_t>(stride));
    }
  }
  return meta;
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
    (void)kernel_name;
    (void)ndim;
    (void)dynamic_dims;
    (void)use_big_index;
    (void)work_per_thread;

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
                      binding++, i, compiled_glsl_type(x.dtype()), xname);
    }

    bool add_meta = !contiguous;
    if (add_meta) {
        os += fmt::format(
            "layout(set = 0, binding = {}) readonly buffer Meta {{ int meta[]; }} meta_buf;\n",
            binding++);
    }
    
    // Outputs
    std::vector<std::string> out_names;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& x = outputs[i];
        std::string xname = namer.get_name(x);
        out_names.push_back(xname);
        os += fmt::format("layout(set = 0, binding = {}) writeonly buffer Out{} {{ {} {}[]; }};\n",
                      binding++, i, compiled_glsl_type(x.dtype()), xname);
    }
    
    // Push constants
    os += "layout(push_constant) uniform PushConstants {\n";
    os += "  uint size;\n";
    os += "  uint ndim;\n";
    os += "} pc;\n";
    
    os += "void main() {\n";
    os += "  uint index = gl_GlobalInvocationID.x;\n";
    os += "  if (index >= pc.size) return;\n";
    
    // Reading inputs
    int meta_block = 1;
    for (int i = 0; i < inputs.size(); ++i) {
        auto& x = inputs[i];
        auto xname = namer.get_name(x);
        if (is_constant(i)) {
            std::ostringstream ss;
            print_constant(ss, x);
            os += fmt::format(
                "  {} tmp_{} = {};\n",
                compiled_glsl_type(x.dtype()),
                xname,
                ss.str());
        } else if (is_scalar(x)) {
            os += fmt::format(
                "  {} tmp_{} = {}[0];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname);
        } else if (!contiguous) {
            os += fmt::format("  uint rem_{0} = index;\n", xname);
            os += fmt::format("  int idx_{0} = 0;\n", xname);
            os += "  for (int d = int(pc.ndim) - 1; d >= 0; --d) {\n";
            os += "    uint dim = uint(meta_buf.meta[d]);\n";
            os += "    uint coord = rem_" + xname + " % dim;\n";
            os += "    rem_" + xname + " /= dim;\n";
            os += fmt::format(
                "    idx_{0} += int(coord) * meta_buf.meta[int(pc.ndim) * {1} + d];\n",
                xname,
                meta_block);
            os += "  }\n";
            os += fmt::format(
                "  {} tmp_{} = {}[idx_{}];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname,
                xname);
            meta_block++;
        } else {
            os += fmt::format(
                "  {} tmp_{} = {}[index];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname);
        }
    }
    
    // Tape
    for (auto& x : tape) {
        std::string out_type = compiled_glsl_type(x.dtype());
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

void build_reduction_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const TerminalSumReduction& reduction,
    bool contiguous) {
    (void)kernel_name;
    (void)reduction;

    NodeNamer namer;
    os += "#version 450\n";
    os += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";

    int binding = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (is_constant(i)) continue;
        const auto& x = inputs[i];
        std::string xname = namer.get_name(x);
        os += fmt::format(
            "layout(set = 0, binding = {}) readonly buffer In{} {{ {} {}[]; }};\n",
            binding++,
            i,
            compiled_glsl_type(x.dtype()),
            xname);
    }

    if (!contiguous) {
        os += fmt::format(
            "layout(set = 0, binding = {}) readonly buffer Meta {{ int meta[]; }} meta_buf;\n",
            binding++);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& x = outputs[i];
        std::string xname = namer.get_name(x);
        os += fmt::format(
            "layout(set = 0, binding = {}) writeonly buffer Out{} {{ {} {}[]; }};\n",
            binding++,
            i,
            compiled_glsl_type(x.dtype()),
            xname);
    }

    os += "layout(push_constant) uniform PushConstants {\n";
    os += "  uint size;\n";
    os += "  uint ndim;\n";
    os += "  uint reduce_size;\n";
    os += "} pc;\n";

    os += "void main() {\n";
    os += "  uint index = gl_GlobalInvocationID.x;\n";
    os += "  if (index >= pc.size) return;\n";
    os += fmt::format(
        "  {} acc = 0.0;\n",
        compiled_glsl_type(outputs[0].dtype()));
    os += "  for (uint r = 0; r < pc.reduce_size; ++r) {\n";
    os += "    uint reduce_index = index * pc.reduce_size + r;\n";

    int meta_block = 1;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& x = inputs[i];
        auto xname = namer.get_name(x);
        if (is_constant(i)) {
            std::ostringstream ss;
            print_constant(ss, x);
            os += fmt::format(
                "    {} tmp_{} = {};\n",
                compiled_glsl_type(x.dtype()),
                xname,
                ss.str());
        } else if (is_scalar(x)) {
            os += fmt::format(
                "    {} tmp_{} = {}[0];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname);
        } else if (!contiguous) {
            os += fmt::format("    uint rem_{0} = reduce_index;\n", xname);
            os += fmt::format("    int idx_{0} = 0;\n", xname);
            os += "    for (int d = int(pc.ndim) - 1; d >= 0; --d) {\n";
            os += "      uint dim = uint(meta_buf.meta[d]);\n";
            os += "      uint coord = rem_" + xname + " % dim;\n";
            os += "      rem_" + xname + " /= dim;\n";
            os += fmt::format(
                "      idx_{0} += int(coord) * meta_buf.meta[int(pc.ndim) * {1} + d];\n",
                xname,
                meta_block);
            os += "    }\n";
            os += fmt::format(
                "    {} tmp_{} = {}[idx_{}];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname,
                xname);
            meta_block++;
        } else {
            os += fmt::format(
                "    {} tmp_{} = {}[reduce_index];\n",
                compiled_glsl_type(x.dtype()),
                xname,
                xname);
        }
    }

    for (size_t i = 0; i + 1 < tape.size(); ++i) {
        auto& x = tape[i];
        std::string out_type = compiled_glsl_type(x.dtype());
        std::string out_name = namer.get_name(x);
        std::vector<std::string> inps;
        for (auto& in : x.inputs()) {
            inps.push_back("tmp_" + namer.get_name(in));
        }

        if (is_static_cast(x.primitive())) {
            os += fmt::format(
                "    {} tmp_{} = {}({});\n",
                out_type,
                out_name,
                out_type,
                inps[0]);
        } else {
            std::string expr = to_glsl_op(x.primitive().name(), x.inputs(), inps);
            os += fmt::format("    {} tmp_{} = {};\n", out_type, out_name, expr);
        }
    }

    const auto& reduce_input = tape.back().inputs()[0];
    os += fmt::format("    acc += tmp_{};\n", namer.get_name(reduce_input));
    os += "  }\n";
    auto out_name = namer.get_name(outputs[0]);
    os += fmt::format("  {}[index] = acc;\n", out_name);
    os += "}\n";
}

} // namespace


void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  maybe_log_compile_profile(kernel_lib_, inputs, outputs, tape_);

  try {
    auto is_supported_array = [&](const array& arr) {
      return is_compiled_float_dtype(arr.dtype());
    };
    if (!std::all_of(inputs.begin(), inputs.end(), is_supported_array) ||
        !std::all_of(outputs.begin(), outputs.end(), is_supported_array) ||
        !std::all_of(tape_.begin(), tape_.end(), is_supported_array)) {
      throw std::runtime_error(
          "[Compiled::eval_gpu] Only float32/float16/bfloat16 compiled kernels are supported.");
    }

    if (!outputs[0].flags().contiguous) {
      throw std::runtime_error("[Compiled::eval_gpu] Strided Vulkan compilation not implemented yet.");
    }
    auto& s = stream();
    auto& d = vulkan::device(s.device);
    auto& compute_encoder = d.get_command_encoder(s);
    std::vector<array> shader_inputs = inputs;
    for (size_t i = 0; i < shader_inputs.size(); ++i) {
      if (is_constant_(i)) {
        continue;
      }
      if (shader_inputs[i].dtype() != float32) {
        array cast_input(shader_inputs[i].shape(), float32, nullptr, {});
        copy_gpu(
            shader_inputs[i],
            cast_input,
            compiled_copy_type(shader_inputs[i], cast_input),
            s);
        shader_inputs[i] = cast_input;
        d.add_temporary(s, shader_inputs[i]);
      }
    }
    std::vector<array> shader_outputs = outputs;
    std::vector<bool> output_needs_cast(outputs.size(), false);
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].dtype() != float32) {
        shader_outputs[i] = array(outputs[i].shape(), float32, nullptr, {});
        output_needs_cast[i] = true;
      }
    }

    auto reduction = analyze_terminal_sum_reduction(outputs_, tape_);
    if (reduction) {
      if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size()) {
        throw std::runtime_error(
            "[Compiled::eval_gpu] Reduction kernel input/output arity mismatch.");
      }
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].shape() != inputs_[i].shape()) {
          throw std::runtime_error(
              "[Compiled::eval_gpu] Vulkan compiled reductions are shape-specialized and do not yet support input shape changes.");
        }
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].shape() != outputs_[i].shape()) {
          throw std::runtime_error(
              "[Compiled::eval_gpu] Vulkan compiled reductions are shape-specialized and do not yet support output shape changes.");
        }
      }
    }

    std::string kernel_name =
        kernel_lib_ + (reduction ? "_vulkan_sum_last" : "_vulkan_contig");

    // Pipeline layout out
    VkPipelineLayout layout;
    VkDescriptorSetLayout ds_layout;

    int num_bindings = 0;
    for (int i = 0; i < shader_inputs.size(); i++) {
      if (!is_constant_(i)) num_bindings++;
    }

    bool contiguous = true;
    Shape shape;
    std::vector<Strides> strides;
    std::vector<int32_t> meta_data;
    uint32_t push_constant_size = sizeof(uint32_t) * 2;

    if (reduction) {
      shape = reduction->source_shape;
      contiguous =
          compiled_reduction_inputs_contiguous(shader_inputs, shape, is_constant_);
      if (!contiguous) {
        num_bindings++;
        meta_data =
            compiled_reduction_metadata(shader_inputs, shape, is_constant_);
      }
      push_constant_size = sizeof(uint32_t) * 3;
    } else {
      std::tie(contiguous, shape, strides) = compiled_collapse_contiguous_dims(
          shader_inputs, shader_outputs[0], is_constant_);
      if (!contiguous) {
        num_bindings++;
        meta_data.reserve(shape.size() * (strides.size()));
        for (auto dim : shape) {
          meta_data.push_back(static_cast<int32_t>(dim));
        }
        for (size_t i = 1; i < strides.size(); ++i) {
          for (auto stride : strides[i]) {
            meta_data.push_back(static_cast<int32_t>(stride));
          }
        }
      }
    }
    num_bindings += shader_outputs.size();

    std::string glsl;
    if (reduction) {
      build_reduction_kernel(
          glsl,
          kernel_name,
          inputs_,
          outputs_,
          tape_,
          is_constant_,
          *reduction,
          contiguous);
    } else {
      build_kernel(
          glsl,
          kernel_name,
          inputs_,
          outputs_,
          tape_,
          is_constant_,
          contiguous,
          static_cast<int>(shape.size()),
          !contiguous,
          false,
          1);
    }

    std::vector<uint32_t> spv = compile_glsl_to_spirv(kernel_name, glsl);

    VkPipeline pipeline = d.get_pipeline_from_spirv(
            kernel_name,
            spv,
            layout,
            ds_layout,
            num_bindings,
            push_constant_size
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
      buf_infos[bind_idx].buffer = vulkan::get_buffer(arr);
      buf_infos[bind_idx].offset = static_cast<VkDeviceSize>(arr.offset());
      buf_infos[bind_idx].range = VK_WHOLE_SIZE;
      
      writes[bind_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[bind_idx].dstSet = ds;
      writes[bind_idx].dstBinding = bind_idx;
      writes[bind_idx].descriptorCount = 1;
      writes[bind_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[bind_idx].pBufferInfo = &buf_infos[bind_idx];
      bind_idx++;
    };

    auto bind_buffer = [&](VkBuffer buf) {
      buf_infos[bind_idx].buffer = buf;
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
    
    for (int i=0; i<shader_inputs.size(); i++) {
      if(!is_constant_(i)) {
        bind_arr(shader_inputs[i]);
      }
    }

    if (!contiguous) {
      auto* staging =
          vulkan::allocator().alloc_staging(meta_data.size() * sizeof(int32_t));
      std::memcpy(
          staging->mapped_ptr,
          meta_data.data(),
          meta_data.size() * sizeof(int32_t));
      d.add_completed_handler(
          s, [staging]() { vulkan::allocator().free_staging(staging); });
      bind_buffer(staging->buffer);
    }

    if (reduction) {
      for (auto& out : shader_outputs) {
        out.set_data(allocator::malloc(out.nbytes()));
      }
    } else {
      compiled_allocate_outputs(
          shader_inputs, shader_outputs, is_constant_, contiguous);
    }

    for (auto& out : shader_outputs) {
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

    if (reduction) {
      struct PushConstants {
        uint32_t size;
        uint32_t ndim;
        uint32_t reduce_size;
      } pc{
          static_cast<uint32_t>(shader_outputs[0].size()),
          static_cast<uint32_t>(shape.size()),
          static_cast<uint32_t>(reduction->reduce_size),
      };
      vkCmdPushConstants(
          compute_encoder.cmd,
          layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0,
          sizeof(PushConstants),
          &pc);
    } else {
      struct PushConstants {
        uint32_t size;
        uint32_t ndim;
      } pc{
          static_cast<uint32_t>(shader_outputs[0].size()),
          static_cast<uint32_t>(shape.size()),
      };
      vkCmdPushConstants(
          compute_encoder.cmd,
          layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0,
          sizeof(PushConstants),
          &pc);
    }
        
    // Dispatch
    uint32_t nblock = (static_cast<uint32_t>(shader_outputs[0].size()) + 255) / 256;
    vkCmdDispatch(compute_encoder.cmd, nblock, 1, 1);
    
    compute_encoder.op_count++;

    for (size_t i = 0; i < shader_outputs.size(); ++i) {
      if (!output_needs_cast[i]) {
        continue;
      }
      d.add_temporary(s, shader_outputs[i]);
      copy_gpu(
          shader_outputs[i],
          outputs[i],
          compiled_copy_type(shader_outputs[i], outputs[i]),
          s);
    }
  } catch (const std::exception& e) {
    if (!compile_profile_cpu_fallback_enabled()) {
      throw;
    }
    std::fprintf(
        stderr,
        "[MLX Vulkan][compile profile] CPU fallback kernel=%s reason=%s\n",
        kernel_lib_.c_str(),
        e.what());
    std::fflush(stderr);
    eval_cpu(inputs, outputs);
  }
}

} // namespace mlx::core
