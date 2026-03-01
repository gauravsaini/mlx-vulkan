// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Primitives GPU dispatch

#include "mlx/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/distributed/primitives.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/allocator.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core {

// ─────────────────────────────────────────────────────────────────────────────
// Macros for unimplemented ops
// ─────────────────────────────────────────────────────────────────────────────

#define NO_GPU_MULTI(func)                                                     \
  void func::eval_gpu(                                                         \
      const std::vector<array>& inputs, std::vector<array>& outputs) {        \
    throw std::runtime_error(#func " has no Vulkan implementation.");          \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                           \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {         \
    throw std::runtime_error(#func " has no Vulkan implementation.");          \
  }

// ─────────────────────────────────────────────────────────────────────────────
// GPU dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Binary op IDs (must match binary.comp)
enum class BinaryOp : uint32_t {
  Add        = 0,
  Sub        = 1,
  Mul        = 2,
  Div        = 3,
  Max        = 4,
  Min        = 5,
  Pow        = 6,
  Equal      = 7,
  NotEq      = 8,
  Less       = 9,
  LessEq     = 10,
  Greater    = 11,
  GreaterEq  = 12,
  LogAddExp  = 13,
  Arctan2    = 14,
  Remainder  = 15,
  FloorDiv   = 16,
  LogicAnd   = 17,
  LogicOr    = 18,
  BitAnd     = 19,
  BitOr      = 20,
  BitXor     = 21,
  LeftShift  = 22,
  RightShift = 23,
};

// Unary op IDs (must match unary.comp)
enum class UnaryOp : uint32_t {
  Abs      = 0,
  Neg      = 1,
  Sign     = 2,
  Sqrt     = 3,
  Rsqrt    = 4,
  Square   = 5,
  Exp      = 6,
  Expm1    = 7,
  Log      = 8,
  Log1p    = 9,
  Sin      = 10,
  Cos      = 11,
  Tan      = 12,
  Sinh     = 13,
  Cosh     = 14,
  Tanh     = 15,
  Arcsin   = 16,
  Arccos   = 17,
  Arctan   = 18,
  Arcsinh  = 19,
  Arccosh  = 20,
  Arctanh  = 21,
  Ceil     = 22,
  Floor    = 23,
  Round    = 24,
  Erf      = 25,
  Erfinv   = 26,
  Sigmoid  = 27,
  Conjugate= 28,
  Log2     = 29,
  Log10    = 30,
  LogNot   = 33,
};

// Compute broadcast strides for an input array relative to the output shape.
// For each dimension: if input dim == 1, stride = 0 (broadcast); else use original stride.
// Input may have fewer dimensions than output (left-padded with 1s).
static void compute_broadcast_strides(
    const array& in, const Shape& out_shape, uint32_t strides[4]) {
  int out_ndim = out_shape.size();
  int in_ndim = in.ndim();
  for (int i = 0; i < 4; i++) strides[i] = 0;
  for (int i = 0; i < in_ndim; i++) {
    int out_i = out_ndim - in_ndim + i;
    if (out_i >= 0 && out_i < 4) {
      strides[out_i] = (in.shape(i) == 1) ? 0 : static_cast<uint32_t>(in.strides()[i]);
    }
  }
}

// The binary.comp shader has 4 bindings:
//   0=InA, 1=InB, 2=OutCFloat, 3=OutCBool
// Broadcast is handled in-shader via stride-based ND indexing (no pre-copy needed).
void dispatch_binary(
    const array& a,
    const array& b,
    array& out,
    uint32_t op_id,
    const Stream& s) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(s);
  auto& dev = vulkan::device(s.device);
  encoder.op_count++;

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer c_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  // 4 bindings: InA(uint), InB(uint), OutC(uint raw), OutCBool(uint8)
  // push constant size = 80 bytes (20 x uint32)
  VkPipeline pipeline = dev.get_pipeline("binary", layout, ds_layout, 4, 80);
  if (pipeline == VK_NULL_HANDLE) return;

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);

  VkDescriptorBufferInfo a_info{a_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo b_info{b_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo c_info{c_buf, 0, VK_WHOLE_SIZE};

  bool output_is_bool = (out.dtype() == bool_);

  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &a_info;
  writes[1].pBufferInfo = &b_info;
  writes[2].pBufferInfo = &c_info;
  writes[3].pBufferInfo = &c_info;
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  auto to_input_dtype = [](Dtype dt) -> uint32_t {
    switch (dt) {
      case float32: case float16: case bfloat16: return 0;
      case int8:    case int16:   case int32: case int64:   return 1;
      default:                                              return 2;
    }
  };

  Dtype in_dtype = a.dtype();

  // Layout must match binary.comp push constants exactly
  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t output_is_bool;
    uint32_t input_dtype;
    uint32_t input_elem_bytes;
    uint32_t ndim;
    uint32_t out_shape[4];
    uint32_t a_strides[4];
    uint32_t b_strides[4];
  } pc{};

  pc.n = static_cast<uint32_t>(out.size());
  pc.op = op_id;
  pc.output_is_bool = output_is_bool ? 1u : 0u;
  pc.input_dtype = to_input_dtype(in_dtype);
  pc.input_elem_bytes = static_cast<uint32_t>(in_dtype.size());

  // Set up ND dimensions for broadcast indexing
  int ndim = out.ndim();
  pc.ndim = static_cast<uint32_t>(ndim > 4 ? 4 : ndim);
  
  // Fill out_shape (right-aligned, padded with 1s)
  for (int i = 0; i < 4; i++) pc.out_shape[i] = 1;
  for (int i = 0; i < std::min(ndim, 4); i++) {
    int out_i = (ndim <= 4) ? i : (i + ndim - 4);
    pc.out_shape[i] = static_cast<uint32_t>(out.shape(out_i));
  }

  compute_broadcast_strides(a, out.shape(), pc.a_strides);
  compute_broadcast_strides(b, out.shape(), pc.b_strides);

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t groups = vulkan::div_ceil(out.size(), vulkan::WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// Dispatch a unary elementwise shader
void dispatch_unary(
    const array& in,
    array& out,
    uint32_t op_id,
    Stream stream) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream);
  auto& dev = vulkan::device(stream.device);
  encoder.op_count++;

  VkBuffer in_buf  = vulkan::get_buffer(in);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("unary", layout, ds_layout, 2, 16);
  if (pipeline == VK_NULL_HANDLE) return;

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);

  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &in_info;

  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &out_info;

  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t input_elem_bytes;
    uint32_t out_elem_bytes;
  } pc{
      static_cast<uint32_t>(out.size()),
      op_id,
      static_cast<uint32_t>(in.itemsize()),
      static_cast<uint32_t>(out.itemsize())};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t groups = vulkan::div_ceil(out.size(), vulkan::WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise Unary
// ─────────────────────────────────────────────────────────────────────────────

#define UNARY_GPU(cls, op)                                                     \
  void cls::eval_gpu(const std::vector<array>& inputs, array& out) {           \
    dispatch_unary(                                                             \
        inputs[0], out, static_cast<uint32_t>(UnaryOp::op), stream());        \
  }

UNARY_GPU(Abs,        Abs)
UNARY_GPU(ArcCos,     Arccos)
UNARY_GPU(ArcCosh,    Arccosh)
UNARY_GPU(ArcSin,     Arcsin)
UNARY_GPU(ArcSinh,    Arcsinh)
UNARY_GPU(ArcTan,     Arctan)
UNARY_GPU(ArcTanh,    Arctanh)
UNARY_GPU(Ceil,       Ceil)
UNARY_GPU(Conjugate,  Conjugate)
UNARY_GPU(Cos,        Cos)
UNARY_GPU(Cosh,       Cosh)
UNARY_GPU(Erf,        Erf)
UNARY_GPU(ErfInv,     Erfinv)
UNARY_GPU(Exp,        Exp)
UNARY_GPU(Expm1,      Expm1)
UNARY_GPU(Floor,      Floor)
// Log::eval_gpu handled below (supports base-e, base-2, base-10 via state())
UNARY_GPU(Log1p,      Log1p)
UNARY_GPU(Negative,   Neg)
UNARY_GPU(Round,      Round)
// LogicalNot uses a dedicated LOGNOT opcode (not NEG) to correctly handle bool byte-packing
void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_unary(inputs[0], out, static_cast<uint32_t>(UnaryOp::LogNot), stream());
}
UNARY_GPU(Sigmoid,    Sigmoid)
UNARY_GPU(Sign,       Sign)
UNARY_GPU(Sin,        Sin)
UNARY_GPU(Sinh,       Sinh)
UNARY_GPU(Sqrt,       Sqrt)
UNARY_GPU(Square,     Square)
UNARY_GPU(Tan,        Tan)
UNARY_GPU(Tanh,       Tanh)

// Log handles base-e, base-2, and base-10 via a single class with a Base field.
void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto base = state();
  uint32_t op;
  if (base == Log::Base::e)  op = static_cast<uint32_t>(UnaryOp::Log);
  else if (base == Log::Base::two)  op = static_cast<uint32_t>(UnaryOp::Log2);
  else                              op = static_cast<uint32_t>(UnaryOp::Log10);
  dispatch_unary(inputs[0], out, op, stream());
}

// BitwiseInvert: fall back to CPU (requires XOR with all-ones broadcast)
void BitwiseInvert::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise Binary
// ─────────────────────────────────────────────────────────────────────────────

// Broadcast arrays are handled by the shader via modulo indexing:
// idx % a_size / idx % b_size — works for scalar (size=1), broadcast, and full.
#define BINARY_GPU(cls, op)                                                    \
  void cls::eval_gpu(const std::vector<array>& inputs, array& out) {           \
    dispatch_binary(inputs[0], inputs[1], out,                                 \
        static_cast<uint32_t>(BinaryOp::op), stream());                        \
  }

BINARY_GPU(Add,          Add)
BINARY_GPU(ArcTan2,      Arctan2)
BINARY_GPU(Divide,       Div)
BINARY_GPU(Equal,        Equal)
BINARY_GPU(Greater,      Greater)
BINARY_GPU(GreaterEqual, GreaterEq)
BINARY_GPU(Less,         Less)
BINARY_GPU(LessEqual,    LessEq)
BINARY_GPU(LogAddExp,    LogAddExp)
BINARY_GPU(LogicalAnd,   LogicAnd)
BINARY_GPU(LogicalOr,    LogicOr)
BINARY_GPU(Maximum,      Max)
BINARY_GPU(Minimum,      Min)
BINARY_GPU(Multiply,     Mul)
BINARY_GPU(NotEqual,     NotEq)
BINARY_GPU(Power,        Pow)
BINARY_GPU(Remainder,    Remainder)
BINARY_GPU(Subtract,     Sub)

// BitwiseBinary dispatches to the correct op based on the stored enum
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  uint32_t op;
  switch (op_) {
    case BitwiseBinary::And:
      op = static_cast<uint32_t>(BinaryOp::BitAnd);
      break;
    case BitwiseBinary::Or:
      op = static_cast<uint32_t>(BinaryOp::BitOr);
      break;
    case BitwiseBinary::Xor:
      op = static_cast<uint32_t>(BinaryOp::BitXor);
      break;
    case BitwiseBinary::LeftShift:
      op = static_cast<uint32_t>(BinaryOp::LeftShift);
      break;
    case BitwiseBinary::RightShift:
      op = static_cast<uint32_t>(BinaryOp::RightShift);
      break;
    default:
      op = static_cast<uint32_t>(BinaryOp::BitAnd);
      break;
  }
  dispatch_binary(inputs[0], inputs[1], out, op, stream());
}

// ─────────────────────────────────────────────────────────────────────────────
// Ternary (Select / Where)
// ─────────────────────────────────────────────────────────────────────────────

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array& cond   = inputs[0];
  const array& true_  = inputs[1];
  const array& false_ = inputs[2];

  VkBuffer cond_buf  = vulkan::get_buffer(cond);
  VkBuffer true_buf  = vulkan::get_buffer(true_);
  VkBuffer false_buf = vulkan::get_buffer(false_);
  VkBuffer out_buf   = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("ternary", layout, ds_layout, 4, 16);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[4]{
    {cond_buf,  0, VK_WHOLE_SIZE},
    {true_buf,  0, VK_WHOLE_SIZE},
    {false_buf, 0, VK_WHOLE_SIZE},
    {out_buf,   0, VK_WHOLE_SIZE}
  };

  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t cond_scalar;
    uint32_t true_scalar;
    uint32_t false_scalar;
  } pc{
    static_cast<uint32_t>(out.size()),
    cond.data_size() == 1 ? 1u : 0u,
    true_.data_size() == 1 ? 1u : 0u,
    false_.data_size() == 1 ? 1u : 0u
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(
      cmd, vulkan::div_ceil(out.size(), vulkan::WORKGROUP_SIZE), 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Arange
// ─────────────────────────────────────────────────────────────────────────────

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("arange", layout, ds_layout, 1, 12);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = ds;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &out_info;
  vkUpdateDescriptorSets(dev.vk_device(), 1, &write, 0, nullptr);

  struct PushConst {
    uint32_t n;
    float start;
    float step;
  } pc{
    static_cast<uint32_t>(out.size()),
    static_cast<float>(start_),
    static_cast<float>(step_)
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(
      cmd, vulkan::div_ceil(out.size(), vulkan::WORKGROUP_SIZE), 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reduction
// ─────────────────────────────────────────────────────────────────────────────

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  // ── Compute inner_size for strided reduce shader ──────────────────────────
  // The shader uses the formula:
  //   in_idx = (out_idx / inner) * (reduce_size * inner) + j * inner + (out_idx % inner)
  // where inner = product of all dims after the LAST reduce axis.
  // This supports reducing any contiguous block of axes from a row-contiguous input.
  // For last-axis reduce: inner=1, formula = out_idx * reduce_size + j  (same as before).
  const array& raw_in = inputs[0];

  // Compute inner: product of dims after the highest reduce axis
  int max_reduce_ax = *std::max_element(axes_.begin(), axes_.end());
  uint32_t inner = 1;
  for (int i = max_reduce_ax + 1; i < raw_in.ndim(); i++) {
    inner *= static_cast<uint32_t>(raw_in.shape(i));
  }

  VkBuffer in_buf  = vulkan::get_buffer(raw_in);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  // 4 bindings: InRaw, OutFloat, OutBool, InBool
  // Push constant is now 32 bytes (8 × uint32) to include inner and outer_stride.
  VkPipeline pipeline = dev.get_pipeline("reduce", layout, ds_layout, 4, 32);
  if (pipeline == VK_NULL_HANDLE) {
    throw std::runtime_error("[vulkan::eval_gpu] reduce pipeline not found");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo in_info{in_buf,  0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &in_info;   // InRaw
  writes[1].pBufferInfo = &out_info;  // OutFloat
  writes[2].pBufferInfo = &out_info;  // OutBool
  writes[3].pBufferInfo = &in_info;   // InBool
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  uint32_t n_outputs = static_cast<uint32_t>(out.size() > 0 ? out.size() : 1);
  uint32_t reduce_size = static_cast<uint32_t>(raw_in.size() / n_outputs);
  uint32_t outer_stride = reduce_size * inner;  // = reduce_size when inner=1

  uint32_t op_id = 0;
  switch (reduce_type_) {
    case Reduce::ReduceType::Sum:  op_id = 0; break;
    case Reduce::ReduceType::Max:  op_id = 1; break;
    case Reduce::ReduceType::Min:  op_id = 2; break;
    case Reduce::ReduceType::Prod: op_id = 3; break;
    case Reduce::ReduceType::And:  op_id = 4; break;
    case Reduce::ReduceType::Or:   op_id = 5; break;
  }

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
    uint32_t op;
    uint32_t input_is_bool;
    uint32_t output_is_bool;
    uint32_t inner;        // product of dims after the reduce axes
    uint32_t outer_stride; // = reduce_size * inner
  } pc{
    static_cast<uint32_t>(raw_in.size()),
    reduce_size,
    n_outputs,
    op_id,
    raw_in.dtype() == bool_ ? 1u : 0u,
    out.dtype() == bool_ ? 1u : 0u,
    inner,
    outer_stride
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  // One workgroup per output element
  vkCmdDispatch(cmd, n_outputs, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf  = vulkan::get_buffer(inputs[0]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline =
      dev.get_pipeline("arg_reduce", layout, ds_layout, 2, 16);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = writes[1].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[1].dstBinding = 1;
  writes[0].descriptorCount = writes[1].descriptorCount = 1;
  writes[0].descriptorType = writes[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &in_info;
  writes[1].pBufferInfo = &out_info;
  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  uint32_t n_outputs = static_cast<uint32_t>(out.size() > 0 ? out.size() : 1);
  uint32_t reduce_size =
      static_cast<uint32_t>(inputs[0].size() / n_outputs);
  uint32_t op_id =
      (reduce_type_ == ArgReduce::ReduceType::ArgMax) ? 0u : 1u;

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
    uint32_t op;
  } pc{
    static_cast<uint32_t>(inputs[0].size()),
    reduce_size,
    n_outputs,
    op_id
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_outputs, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Matmul
// ─────────────────────────────────────────────────────────────────────────────

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array& a = inputs[0];
  const array& b = inputs[1];

  uint32_t M = (a.ndim() >= 2)
      ? static_cast<uint32_t>(a.shape(-2))
      : 1u;
  uint32_t K = static_cast<uint32_t>(a.shape(-1));
  uint32_t N = static_cast<uint32_t>(b.shape(-1));

  // Batch dimension
  uint32_t batch = 1u;
  for (int i = 0; i < static_cast<int>(out.ndim()) - 2; i++) {
    batch *= static_cast<uint32_t>(out.shape(i));
  }

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer c_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("matmul", layout, ds_layout, 3, 28);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo a_info{a_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo b_info{b_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo c_info{c_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[3]{};
  for (int i = 0; i < 3; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &a_info;
  writes[1].pBufferInfo = &b_info;
  writes[2].pBufferInfo = &c_info;
  vkUpdateDescriptorSets(dev.vk_device(), 3, writes, 0, nullptr);

  struct PushConst {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch;
    uint32_t a_batch_stride;
    uint32_t b_batch_stride;
    uint32_t c_batch_stride;
  } pc{M, N, K, batch, M * K, K * N, M * N};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  // Grid: (ceil(N/16), ceil(M/16), batch)
  vkCmdDispatch(
      cmd, vulkan::div_ceil(N, 16), vulkan::div_ceil(M, 16), batch);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────────────

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf  = vulkan::get_buffer(inputs[0]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("softmax", layout, ds_layout, 2, 8);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = writes[1].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[1].dstBinding = 1;
  writes[0].descriptorCount = writes[1].descriptorCount = 1;
  writes[0].descriptorType = writes[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &in_info;
  writes[1].pBufferInfo = &out_info;
  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  uint32_t axis_size = static_cast<uint32_t>(inputs[0].shape(-1));
  uint32_t n_rows = static_cast<uint32_t>(inputs[0].size() / axis_size);

  struct PushConst {
    uint32_t n;
    uint32_t n_rows;
  } pc{axis_size, n_rows};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  // One workgroup per row
  vkCmdDispatch(cmd, n_rows, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// LogSumExp
// ─────────────────────────────────────────────────────────────────────────────

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf  = vulkan::get_buffer(inputs[0]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline =
      dev.get_pipeline("logsumexp", layout, ds_layout, 2, 12);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = writes[1].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[1].dstBinding = 1;
  writes[0].descriptorCount = writes[1].descriptorCount = 1;
  writes[0].descriptorType = writes[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &in_info;
  writes[1].pBufferInfo = &out_info;
  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  uint32_t n_outputs =
      static_cast<uint32_t>(out.size() > 0 ? out.size() : 1);
  uint32_t reduce_size =
      static_cast<uint32_t>(inputs[0].size() / n_outputs);

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
  } pc{
    static_cast<uint32_t>(inputs[0].size()),
    reduce_size,
    n_outputs
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_outputs, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// DivMod
// ─────────────────────────────────────────────────────────────────────────────

void DivMod::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer a_buf    = vulkan::get_buffer(inputs[0]);
  VkBuffer b_buf    = vulkan::get_buffer(inputs[1]);
  VkBuffer out0_buf = vulkan::get_buffer(outputs[0]);
  VkBuffer out1_buf = vulkan::get_buffer(outputs[1]);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline =
      dev.get_pipeline("binary_two", layout, ds_layout, 4, 16);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[4]{
    {a_buf,    0, VK_WHOLE_SIZE},
    {b_buf,    0, VK_WHOLE_SIZE},
    {out0_buf, 0, VK_WHOLE_SIZE},
    {out1_buf, 0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t a_scalar;
    uint32_t b_scalar;
  } pc{
    static_cast<uint32_t>(outputs[0].size()),
    0u,
    inputs[0].data_size() == 1 ? 1u : 0u,
    inputs[1].data_size() == 1 ? 1u : 0u
  };

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(
      cmd,
      vulkan::div_ceil(outputs[0].size(), vulkan::WORKGROUP_SIZE),
      1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gather / Scatter (indexing)
// ─────────────────────────────────────────────────────────────────────────────

// Shared push constant layout for all indexing ops — 44 bytes (11 × uint32).
// Fields inner and src_outer_stride are only read by INDEX_GATHER_GEN (op=3).
struct IndexPushConst {
  uint32_t n;
  uint32_t op;
  uint32_t idx_size;
  uint32_t src_stride;        // INDEX_GATHER/SCATTER: stride of indexed dim in src
  uint32_t dst_stride;        // INDEX_GATHER/SCATTER: stride in dst
  uint32_t src_offset;
  uint32_t idx_offset;
  uint32_t dst_offset;
  uint32_t src_ax_size;
  uint32_t inner;             // INDEX_GATHER_GEN: product(src dims after gather axis)
  uint32_t src_outer_stride;  // INDEX_GATHER_GEN: d_ax * inner
};
static constexpr uint32_t kIndexPushSize = sizeof(IndexPushConst); // 44

static void indexing_dispatch(
    vulkan::Device& dev,
    vulkan::CommandEncoder& encoder,
    VkBuffer src_buf,
    VkBuffer idx_buf,
    VkBuffer out_buf,
    const IndexPushConst& pc,
    size_t n_out) {
  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("indexing", layout, ds_layout, 3, kIndexPushSize);
  if (pipeline == VK_NULL_HANDLE)
    throw std::runtime_error("[indexing_dispatch] Pipeline not found.");

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[3]{
    {src_buf, 0, VK_WHOLE_SIZE},
    {idx_buf, 0, VK_WHOLE_SIZE},
    {out_buf, 0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[3]{};
  for (int i = 0; i < 3; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 3, writes, 0, nullptr);

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kIndexPushSize, &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, vulkan::div_ceil(n_out, vulkan::WORKGROUP_SIZE), 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Handle single-axis gather (mx.take(src, indices, axis=ax)):
  //   axes_.size()==1, slice_sizes_[ax]==1 for the gathered axis
  // Covers 1D (ax=0) and ND cases via INDEX_GATHER_GEN.
  if (inputs.size() == 2 && axes_.size() == 1 &&
      slice_sizes_[axes_[0]] == 1) {
    out.set_data(allocator::malloc(out.nbytes()));
    if (out.size() == 0) return;

    auto& encoder = vulkan::get_command_encoder(stream());
    auto& dev = vulkan::device(stream().device);
    encoder.op_count++;

    uint32_t ax = static_cast<uint32_t>(axes_[0]);
    const array& src = inputs[0];
    const array& idx = inputs[1];

    // inner = product(src dims after the gather axis)
    uint32_t inner = 1;
    for (int d = ax + 1; d < (int)src.ndim(); d++) inner *= src.shape(d);

    uint32_t d_ax = static_cast<uint32_t>(src.shape(ax));
    IndexPushConst pc{};
    pc.n               = static_cast<uint32_t>(out.size());
    pc.op              = 3u; // INDEX_GATHER_GEN
    pc.idx_size        = static_cast<uint32_t>(idx.size());
    pc.src_stride      = 1u;  // unused by this op
    pc.dst_stride      = 1u;  // unused by this op
    pc.src_offset      = static_cast<uint32_t>(src.offset() * src.itemsize());
    pc.idx_offset      = static_cast<uint32_t>(idx.offset() * idx.itemsize());
    pc.dst_offset      = static_cast<uint32_t>(out.offset() * out.itemsize());
    pc.src_ax_size     = d_ax;
    pc.inner           = inner;
    pc.src_outer_stride= d_ax * inner;

    indexing_dispatch(dev, encoder,
                      vulkan::get_buffer(src),
                      vulkan::get_buffer(idx),
                      vulkan::get_buffer(out),
                      pc, out.size());
    return;
  }

  throw std::runtime_error("[vulkan::Gather::eval_gpu] Fallback to eval_cpu is unsupported");
}

// GatherAxis: simple axis-indexed gather → GPU dispatch
void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array& src = inputs[0];
  const array& idx = inputs[1];

  // Stride of the indexed axis in source and destination
  uint32_t src_stride = 1;
  for (int i = axis_ + 1; i < src.ndim(); i++)
    src_stride *= static_cast<uint32_t>(src.shape(i));
  uint32_t dst_stride = 1;
  for (int i = axis_ + 1; i < out.ndim(); i++)
    dst_stride *= static_cast<uint32_t>(out.shape(i));

  IndexPushConst pc{};
  pc.n           = static_cast<uint32_t>(out.size());
  pc.op          = 0u; // INDEX_GATHER
  pc.idx_size    = static_cast<uint32_t>(idx.size());
  pc.src_stride  = src_stride;
  pc.dst_stride  = dst_stride;
  pc.src_offset  = static_cast<uint32_t>(src.offset() * src.itemsize());
  pc.idx_offset  = static_cast<uint32_t>(idx.offset() * idx.itemsize());
  pc.dst_offset  = static_cast<uint32_t>(out.offset() * out.itemsize());
  pc.src_ax_size = static_cast<uint32_t>(src.shape(axis_));
  // inner / src_outer_stride unused for INDEX_GATHER — leave as 0

  indexing_dispatch(dev, encoder,
                    vulkan::get_buffer(src),
                    vulkan::get_buffer(idx),
                    vulkan::get_buffer(out),
                    pc, out.size());
}

// General Scatter: multi-axis → CPU fallback
void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::Scatter::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ScatterAxis: simple axis-indexed scatter → GPU dispatch
void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  // inputs: [src, indices, updates]
  // Copy src→out first, then scatter updates into it
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  copy_gpu_inplace(inputs[0], out, CopyType::Vector, stream());

  if (inputs.size() < 3 || inputs[2].size() == 0) return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array& idx     = inputs[1];
  const array& updates = inputs[2];

  uint32_t src_stride = 1;
  for (int i = axis_ + 1; i < updates.ndim(); i++)
    src_stride *= static_cast<uint32_t>(updates.shape(i));
  uint32_t dst_stride = 1;
  for (int i = axis_ + 1; i < out.ndim(); i++)
    dst_stride *= static_cast<uint32_t>(out.shape(i));

  // INDEX_SCATTER=1, INDEX_SCATTER_ADD=2
  uint32_t op = (reduce_type_ == ScatterAxis::ReduceType::Sum) ? 2u : 1u;

  IndexPushConst pc{};
  pc.n           = static_cast<uint32_t>(updates.size());
  pc.op          = op;
  pc.idx_size    = static_cast<uint32_t>(idx.size());
  pc.src_stride  = src_stride;
  pc.dst_stride  = dst_stride;
  pc.src_offset  = static_cast<uint32_t>(updates.offset() * updates.itemsize());
  pc.idx_offset  = static_cast<uint32_t>(idx.offset() * idx.itemsize());
  pc.dst_offset  = static_cast<uint32_t>(out.offset() * out.itemsize());
  pc.src_ax_size = static_cast<uint32_t>(out.shape(axis_)); // axis size for neg-idx wrap
  // inner / src_outer_stride unused — leave as 0

  indexing_dispatch(dev, encoder,
                    vulkan::get_buffer(updates),
                    vulkan::get_buffer(idx),
                    vulkan::get_buffer(out),
                    pc, updates.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// Sort / ArgSort → GPU dispatch via sort.comp (bitonic sort)
// ─────────────────────────────────────────────────────────────────────────────

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  const auto& in = inputs[0];

  // Bitonic sort works on the last axis, within workgroup shared memory (512)
  int sort_axis = axis_ < 0 ? in.ndim() + axis_ : axis_;
  uint32_t sort_size = static_cast<uint32_t>(in.shape(sort_axis));

  // Round up to next power of 2 for bitonic sort
  uint32_t sort_pow2 = 1;
  while (sort_pow2 < sort_size) sort_pow2 <<= 1;

  // Throw Error if sort dimension > 256 or not last axis
  if (sort_pow2 > 256 || sort_axis != in.ndim() - 1) {
    throw std::runtime_error("[vulkan::Sort::eval_gpu] Fallback to eval_cpu is unsupported");
  }

  // Copy input to output (sort is in-place on output)
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu_inplace(in, out, CopyType::Vector, stream());

  uint32_t n = static_cast<uint32_t>(out.size());
  uint32_t n_sorts = n / sort_size;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer data_buf = vulkan::get_buffer(out);

  // Allocate dummy index buffer (sort shader requires both bindings)
  auto idx_alloc = allocator::malloc(n_sorts * sort_size * sizeof(uint32_t));
  VkBuffer idx_buf = static_cast<vulkan::VulkanBuffer*>(idx_alloc.ptr())->buffer;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("sort", layout, ds_layout, 2, 20);
  if (pipeline == VK_NULL_HANDLE) {
    allocator::free(idx_alloc);
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[2]{
    {data_buf, 0, VK_WHOLE_SIZE},
    {idx_buf,  0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[2]{};
  for (int i = 0; i < 2; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t sort_size;
    uint32_t n_sorts;
    uint32_t ascending;
    uint32_t with_index;
  } pc{n, sort_pow2, n_sorts, 1u, 0u};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_sorts, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);

  // Free index buffer after GPU finishes
  encoder.add_completed_handler([idx_alloc]() mutable {
    allocator::free(idx_alloc);
  });
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::ArgSort::eval_gpu] Fallback to eval_cpu is unsupported");
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::Partition::eval_gpu] Fallback to eval_cpu is unsupported");
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::ArgPartition::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan → GPU dispatch via scan.comp (prefix scan)
// ─────────────────────────────────────────────────────────────────────────────

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  uint32_t scan_size = static_cast<uint32_t>(in.shape(axis_));

  // Shader supports scan_size ≤ 1024 (256 threads × chunk_size ≤ 4).
  if (scan_size > 1024) {
    throw std::runtime_error(
        "[vulkan::Scan::eval_gpu] scan_size > 1024: unsupported");
  }

  // Map reduce type to shader op code
  uint32_t op;
  switch (reduce_type_) {
    case Sum:       op = 0u; break;
    case Prod:      op = 1u; break;
    case Max:       op = 2u; break;
    case Min:       op = 3u; break;
    case LogAddExp: op = 4u; break;
    default:
      throw std::runtime_error("[vulkan::Scan::eval_gpu] unsupported reduce_type");
  }

  uint32_t n_scans = static_cast<uint32_t>(in.size() / scan_size);

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  struct ScanPushConst {
    uint32_t n;
    uint32_t scan_size;
    uint32_t n_scans;
    uint32_t op;
    uint32_t inclusive;
    uint32_t reverse;
  } pc{
    static_cast<uint32_t>(out.size()),
    scan_size,
    n_scans,
    op,
    inclusive_ ? 1u : 0u,
    reverse_   ? 1u : 0u
  };

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("scan", layout, ds_layout, 2, sizeof(pc));
  if (pipeline == VK_NULL_HANDLE)
    throw std::runtime_error("[vulkan::Scan::eval_gpu] scan pipeline not found");

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[2]{
    {vulkan::get_buffer(in),  0, VK_WHOLE_SIZE},
    {vulkan::get_buffer(out), 0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[2]{};
  for (int i = 0; i < 2; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  // Dispatch one workgroup per independent scan segment
  vkCmdDispatch(cmd, n_scans, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}


// ─────────────────────────────────────────────────────────────────────────────
// AddMM - CPU fallback until matmul+add pipeline is integrated
// ─────────────────────────────────────────────────────────────────────────────

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  auto& s = stream();
  auto& d = vulkan::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  const array& a = inputs[0];
  const array& b = inputs[1];
  const array& c = inputs[2];

  if (a.size() == 0 || b.size() == 0) {
    if (beta_ == 1.0f) {
      array zero_arr(0.0f, out.dtype());
      dispatch_binary(c, zero_arr, out, 0 /* Add */, s);
      encoder.add_temporary(zero_arr);
    } else {
      array beta_arr(beta_, out.dtype());
      dispatch_binary(c, beta_arr, out, 2 /* Mul */, s);
      encoder.add_temporary(beta_arr);
    }
    return;
  }

  array temp_ab(out.shape(), out.dtype(), nullptr, {});
  Matmul(s).eval_gpu({a, b}, temp_ab);
  encoder.add_temporary(temp_ab);

  array scaled_ab = temp_ab;
  if (alpha_ != 1.0f) {
    array alpha_arr(alpha_, out.dtype());
    array temp(out.shape(), out.dtype(), nullptr, {});
    dispatch_binary(temp_ab, alpha_arr, temp, 2 /* Mul */, s);
    encoder.add_temporary(alpha_arr);
    encoder.add_temporary(temp);
    scaled_ab = temp;
  }

  array scaled_c = c;
  if (beta_ != 1.0f && beta_ != 0.0f) {
    array beta_arr(beta_, out.dtype());
    array temp(out.shape(), out.dtype(), nullptr, {});
    dispatch_binary(c, beta_arr, temp, 2 /* Mul */, s);
    encoder.add_temporary(beta_arr);
    encoder.add_temporary(temp);
    scaled_c = temp;
  }

  if (beta_ == 0.0f) {
    array zero_arr(0.0f, out.dtype());
    dispatch_binary(scaled_ab, zero_arr, out, 0 /* Add */, s);
    encoder.add_temporary(zero_arr);
  } else {
    dispatch_binary(scaled_ab, scaled_c, out, 0 /* Add */, s);
  }

  encoder.add_temporary(temp_ab);
  if (alpha_ != 1.0f) encoder.add_temporary(scaled_ab);
  if (beta_ != 1.0f && beta_ != 0.0f) encoder.add_temporary(scaled_c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution - CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& s = stream();
  auto& dev = vulkan::device(s.device);

  const array& in = inputs[0];
  const array& wt = inputs[1];

  // The simplified conv.comp shader currently supports basic 2D convs with:
  // - batch size = 1
  // - groups = 1
  // - no dilation
  // - 4D input shapes (N, H, W, C)
  // - Weight shapes (O, H, W, I)
  // For anything more complex (1D, 3D, batched, dilated, groups), we fallback to CPU.
  bool can_use_gpu = true;
  if (in.ndim() != 4 || wt.ndim() != 4) can_use_gpu = false;
  if (groups_ != 1) can_use_gpu = false;
  if (in.shape(0) != 1) can_use_gpu = false; // batch > 1
  for (auto d : kernel_dilation_) if (d != 1) can_use_gpu = false;
  for (auto d : input_dilation_) if (d != 1) can_use_gpu = false;

  if (!can_use_gpu) {
    throw std::runtime_error("[vulkan::Convolution::eval_gpu] Fallback to eval_cpu is unsupported");
  }

  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("conv", layout, ds_layout, 3, 52);
  if (pipeline == VK_NULL_HANDLE) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
      return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkBuffer in_buf = vulkan::get_buffer(in);
  VkBuffer wt_buf = vulkan::get_buffer(wt);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkDescriptorBufferInfo infos[3] = {
      {in_buf, 0, VK_WHOLE_SIZE},
      {wt_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}
  };

  VkWriteDescriptorSet writes[3]{};
  for (int i = 0; i < 3; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 3, writes, 0, nullptr);

  // Extracted shapes based on NHWC assumed layout
  uint32_t N = in.shape(0);
  uint32_t H = in.shape(1);
  uint32_t W = in.shape(2);
  uint32_t C = in.shape(3);

  uint32_t OC = wt.shape(0);
  uint32_t KH = wt.shape(1);
  uint32_t KW = wt.shape(2);

  uint32_t OH = out.shape(1);
  uint32_t OW = out.shape(2);

  uint32_t SH = kernel_strides_[0];
  uint32_t SW = kernel_strides_[1];
  uint32_t PH = padding_lo_[0];
  uint32_t PW = padding_lo_[1];

  struct PushConst {
      uint32_t N, C, H, W, KH, KW;
      uint32_t OH, OW, SH, SW, PH, PW, OC;
  } pc{N, C, H, W, KH, KW, OH, OW, SH, SW, PH, PW, OC};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t grid_x = vulkan::div_ceil(OW, 16u);
  uint32_t grid_y = vulkan::div_ceil(OH, 16u);
  uint32_t grid_z = vulkan::div_ceil(OC, 1u);

  vkCmdDispatch(cmd, grid_x, grid_y, grid_z);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// RandomBits (Philox PRNG) - CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  uint32_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  uint32_t half_size = out_per_key / 2;
  uint32_t odd = out_per_key % 2;

  auto& s = stream();
  auto& encoder = vulkan::get_command_encoder(s);
  auto& dev = vulkan::device(s.device);

  encoder.op_count++;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("rbits", layout, ds_layout, 2, 12);
  if (pipeline == VK_NULL_HANDLE) {
    throw std::runtime_error("[vulkan::RandomBits::eval_gpu] Fallback to eval_cpu is unsupported");
  }

  uint32_t grid_y = half_size + odd;
  uint32_t grid_x = vulkan::div_ceil(num_keys, 256u);
  if (grid_y == 0) grid_y = 1; // avoid zero dispatch

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkBuffer keys_buf = vulkan::get_buffer(keys);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkDescriptorBufferInfo keys_info{keys_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &keys_info;

  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &out_info;

  vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);

  struct PushConst {
    uint32_t odd;
    uint32_t bytes_per_key;
    uint32_t ndim;
    uint32_t num_keys;
  } pc{odd, static_cast<uint32_t>(bytes_per_key), static_cast<uint32_t>(keys.ndim()), static_cast<uint32_t>(num_keys)};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  vkCmdDispatch(cmd, grid_x, grid_y, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load (memory-mapped files)
// ─────────────────────────────────────────────────────────────────────────────

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// Imag / Real (complex)
// ─────────────────────────────────────────────────────────────────────────────

void Imag::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

void Real::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// MaskedScatter
// ─────────────────────────────────────────────────────────────────────────────

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// Compiled / JIT
// ─────────────────────────────────────────────────────────────────────────────

void Compiled::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[vulkan::eval_gpu] Fallback to eval_cpu is unsupported");
}

// ─────────────────────────────────────────────────────────────────────────────
// Matmul variants — CPU fallback stubs
//
// BlockMaskedMM, GatherMM, SegmentedMM: all are UnaryPrimitive (single output).
// The Vulkan eval.cpp catch block intercepts exceptions containing "has no
// Vulkan" and routes them to eval_cpu() via unified-memory access (MoltenVK).
// This means throwing from eval_gpu with that substring enables transparent
// CPU fallback. Users can also explicitly use mx.stream(mx.cpu).
//
// GatherQMM: quantized variant, similarly complex — kept as NO_GPU.
// ─────────────────────────────────────────────────────────────────────────────

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Block-sparse masked matmul.  The Metal implementation requires dedicated
  // block-sparse kernels.  CUDA also has no GPU implementation for this op.
  // eval.cpp will catch this and call eval_cpu() via the unified-memory path.
  throw std::runtime_error(
      "[vulkan] BlockMaskedMM has no Vulkan GPU implementation. "
      "Falling back to CPU eval.");
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Sparse gather + matmul (used in MoE routing).
  // inputs[0]: A (lhs matrices), inputs[1]: B (rhs matrices)
  // inputs[2]: lhs_indices, inputs[3]: rhs_indices
  // A GPU implementation would use indexing.comp to gather then dispatch
  // matmul.comp per gathered pair.  For now, fall back to eval_cpu().
  throw std::runtime_error(
      "[vulkan] GatherMM has no Vulkan GPU implementation. "
      "Falling back to CPU eval.");
}

NO_GPU(GatherQMM)

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Segmented (variable-length batch) matmul.
  // CUDA backend also has no GPU implementation for this op.
  // eval.cpp catches this and calls eval_cpu() via the unified-memory path.
  throw std::runtime_error(
      "[vulkan] SegmentedMM has no Vulkan GPU implementation. "
      "Falling back to CPU eval.");
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedMatmul — two-pass GPU: dequantize weights → float matmul
//
// quantized.comp push_constant (6 x uint32 = 24 bytes):
//   {n, op, group_size, bits, N_dim, K_dim}
//
// Op 1 (DEQUANT_AFFINE):      src[N,K_packed] -> dst[N,K]  (flat, same order)
// Op 2 (DEQUANT_AFFINE_TRANS): src[N,K_packed] -> dst[K,N]  (transposed)
//
// transpose_=false: weight is [K_packed, N], dequant op=1, dq shape=[K,N]
//                   matmul: x[M,K] @ dq[K,N] -> out[M,N]
// transpose_=true:  weight is [N, K_packed], dequant op=2, dq shape=[K,N]
//                   matmul: x[M,K] @ dq[K,N] -> out[M,N]
// ─────────────────────────────────────────────────────────────────────────────

void QuantizedMatmul::eval_gpu(
    const std::vector<array>& inputs, array& out) {
  assert(inputs.size() >= 4);
  auto& s   = stream();
  auto& dev = vulkan::device(s.device);
  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;

  // inputs[0] = float x      [M, K]
  // inputs[1] = packed uint  w_pack
  //               transpose_=false: [K_packed, N]
  //               transpose_=true:  [N, K_packed]
  // inputs[2] = float scales
  //               transpose_=false: [K_groups, N]
  //               transpose_=true:  [N, K_groups]
  // inputs[3] = float biases (same shape as scales)
  const array& x      = inputs[0];
  const array& w_pack = inputs[1];
  const array& scales = inputs[2];
  const array& biases = inputs[3];

  // Do NOT pre-allocate out here: Matmul::eval_gpu will call out.set_data().
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  int K = static_cast<int>(x.shape(-1));
  int N = static_cast<int>(out.shape(-1));
  uint32_t n_dequant = static_cast<uint32_t>(K) * static_cast<uint32_t>(N);

  // After dequantization both paths produce a [K, N] float buffer suitable
  // for Matmul: x[M,K] @ dq[K,N] -> out[M,N].
  array dq_weights(Shape{K, N}, float32, nullptr, {});
  dq_weights.set_data(allocator::malloc(dq_weights.nbytes()));

  // ── Pass 1: Dequantize ───────────────────────────────────────────────────
  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  // 4 bindings, push_constant = 6 x uint32 = 24 bytes
  VkPipeline pipeline = dev.get_pipeline("quantized", layout, ds_layout, 4, 24);
  if (pipeline == VK_NULL_HANDLE) {
    encoder.add_temporary(dq_weights);
    throw std::runtime_error(
        "[vulkan::QuantizedMatmul] failed to get quantized dequant pipeline");
  }

  VkBuffer src_buf   = vulkan::get_buffer(w_pack);
  VkBuffer scale_buf = vulkan::get_buffer(scales);
  VkBuffer bias_buf  = vulkan::get_buffer(biases);
  VkBuffer dq_buf    = vulkan::get_buffer(dq_weights);

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo bufs[4] = {
    {src_buf,   0, VK_WHOLE_SIZE},
    {scale_buf, 0, VK_WHOLE_SIZE},
    {bias_buf,  0, VK_WHOLE_SIZE},
    {dq_buf,    0, VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet          = ds;
    writes[i].dstBinding      = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo     = &bufs[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  // op=1: DEQUANT_AFFINE (flat, for transpose_=false)
  // op=2: DEQUANT_AFFINE_TRANS (2D-aware transpose, for transpose_=true)
  uint32_t dq_op = transpose_ ? 2u : 1u;
  struct QuantPushConst {
    uint32_t n;
    uint32_t op;
    uint32_t group_size;
    uint32_t bits;
    uint32_t N_dim;
    uint32_t K_dim;
  } pc{n_dequant, dq_op,
       static_cast<uint32_t>(group_size_),
       static_cast<uint32_t>(bits_),
       static_cast<uint32_t>(N),
       static_cast<uint32_t>(K)};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, vulkan::div_ceil(n_dequant, 256u), 1, 1);

  // Barrier: dequant writes must be visible before matmul reads
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);

  // ── Pass 2: float matmul x[M,K] @ dq[K,N] -> out[M,N] ───────────────────
  Matmul(s).eval_gpu({x, dq_weights}, out);
  encoder.add_temporary(dq_weights);
}

// ─────────────────────────────────────────────────────────────────────────────
// QQMatmul — quantized x quantized (both operands quantized)
// Two-pass GPU: dequantize both -> float matmul.
// For now, only the RHS is dequantized on GPU (same as QuantizedMatmul).
// The LHS (also quantized) dequantize is not yet wired up; use NO_GPU.
// ─────────────────────────────────────────────────────────────────────────────

NO_GPU(QQMatmul)


// ─────────────────────────────────────────────────────────────────────────────
// FFT / Hadamard
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra -- CPU fallbacks (unified memory / MoltenVK)
//
// On MoltenVK (Apple Silicon unified memory) the VMA-allocated buffers are
// CPU-accessible through their mapped pointer, so calling eval_cpu from inside
// eval_gpu works directly.  The cpu::CommandEncoder::dispatch() path already
// detects a GPU stream and executes the lambda synchronously (see encoder.h).
//
// TODO: On discrete GPUs (AMD/NVIDIA) these ops would need an explicit
//       staging-buffer round-trip before delegating to eval_cpu.
// ─────────────────────────────────────────────────────────────────────────────

// -- Multi-output ops ---------------------------------------------------------

void LUF::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  // eval_cpu allocates lu / pivots / row_indices internally.
  eval_cpu(inputs, outputs);
}

void QRF::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  // eval_cpu allocates Q and R internally via set_data.
  eval_cpu(inputs, outputs);
}

void SVD::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  // eval_cpu allocates U, S, Vt (or just S) internally.
  eval_cpu(inputs, outputs);
}

void Eig::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  // eval_cpu allocates eigenvalues / eigenvectors internally.
  eval_cpu(inputs, outputs);
}

void Eigh::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  // eval_cpu allocates eigenvalues and copies input to eigenvectors internally.
  eval_cpu(inputs, outputs);
}

// -- Single-output ops --------------------------------------------------------

void Inverse::eval_gpu(const std::vector<array>& inputs, array& out) {
  // eval_cpu calls copy_cpu which invokes set_copy_output_data -> allocates out.
  eval_cpu(inputs, out);
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  // eval_cpu calls copy_cpu which invokes set_copy_output_data -> allocates out.
  eval_cpu(inputs, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// fast:: namespace - neural net ops
// ─────────────────────────────────────────────────────────────────────────────

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  return true; // Vulkan SDPA not yet implemented
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

bool ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s) {
  return true;
}

// ─── LayerNorm GPU dispatch ─────────────────────────────────────────────────

bool LayerNorm::use_fallback(Stream s) {
  return false;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  uint32_t n_cols = static_cast<uint32_t>(in.shape().back());
  uint32_t n_rows = static_cast<uint32_t>(in.size() / n_cols);

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  bool has_weight = inputs.size() > 1 && inputs[1].size() > 0;
  bool has_bias   = inputs.size() > 2 && inputs[2].size() > 0;

  VkBuffer in_buf     = vulkan::get_buffer(in);
  VkBuffer weight_buf = has_weight ? vulkan::get_buffer(inputs[1]) : in_buf;
  VkBuffer bias_buf   = has_bias   ? vulkan::get_buffer(inputs[2]) : in_buf;
  VkBuffer out_buf    = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("normalization", layout, ds_layout, 4, 24);
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[4]{
    {in_buf,     0, VK_WHOLE_SIZE},
    {weight_buf, 0, VK_WHOLE_SIZE},
    {bias_buf,   0, VK_WHOLE_SIZE},
    {out_buf,    0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  struct PushConst {
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t op;        // 0 = NORM_LAYER_NORM
    float eps;
    uint32_t has_weight;
    uint32_t has_bias;
  } pc{n_rows, n_cols, 0u, eps_, has_weight ? 1u : 0u, has_bias ? 1u : 0u};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_rows, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

NO_GPU_MULTI(LayerNormVJP)

// ─── RMSNorm GPU dispatch ───────────────────────────────────────────────────

bool RMSNorm::use_fallback(Stream s) {
  return false;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  uint32_t n_cols = static_cast<uint32_t>(in.shape().back());
  uint32_t n_rows = static_cast<uint32_t>(in.size() / n_cols);

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  bool has_weight = inputs.size() > 1 && inputs[1].size() > 0;

  VkBuffer in_buf     = vulkan::get_buffer(in);
  VkBuffer weight_buf = has_weight ? vulkan::get_buffer(inputs[1]) : in_buf;
  VkBuffer out_buf    = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("normalization", layout, ds_layout, 4, 24);
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[4]{
    {in_buf,     0, VK_WHOLE_SIZE},
    {weight_buf, 0, VK_WHOLE_SIZE},
    {in_buf,     0, VK_WHOLE_SIZE},   // bias_buf placeholder (unused)
    {out_buf,    0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  struct PushConst {
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t op;        // 1 = NORM_RMS_NORM
    float eps;
    uint32_t has_weight;
    uint32_t has_bias;
  } pc{n_rows, n_cols, 1u, eps_, has_weight ? 1u : 0u, 0u};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_rows, 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

NO_GPU_MULTI(RMSNormVJP)

// ─── RoPE GPU dispatch ──────────────────────────────────────────────────────

bool RoPE::use_fallback(Stream s) {
  return false;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  // inputs: [x, cos_freqs, sin_freqs]
  if (inputs.size() < 3) {
    fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf  = vulkan::get_buffer(in);
  VkBuffer cos_buf = vulkan::get_buffer(inputs[1]);
  VkBuffer sin_buf = vulkan::get_buffer(inputs[2]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline("rope", layout, ds_layout, 4, 12);
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(ds_layout);
  VkDescriptorBufferInfo infos[4]{
    {in_buf,  0, VK_WHOLE_SIZE},
    {cos_buf, 0, VK_WHOLE_SIZE},
    {sin_buf, 0, VK_WHOLE_SIZE},
    {out_buf, 0, VK_WHOLE_SIZE}
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  uint32_t D = static_cast<uint32_t>(dims_);
  uint32_t n_heads = static_cast<uint32_t>(in.size() / D);

  struct PushConst {
    uint32_t n;
    uint32_t D;
    uint32_t n_heads;
  } pc{static_cast<uint32_t>(in.size()), D, n_heads};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, vulkan::div_ceil(in.size() / 2, vulkan::WORKGROUP_SIZE), 1, 1);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
NO_GPU_MULTI(ConvertFP8)
NO_GPU_MULTI(Quantize)
NO_GPU_MULTI(CustomKernel)

} // namespace fast

// ─────────────────────────────────────────────────────────────────────────────
// distributed namespace
// ─────────────────────────────────────────────────────────────────────────────

namespace distributed {

NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)

} // namespace distributed

} // namespace mlx::core

