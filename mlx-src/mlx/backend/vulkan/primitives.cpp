// Copyright © 2025 Apple Inc.
// MLX Vulkan Backend - Primitives GPU dispatch

#include "mlx/primitives.h"
#include "mlx/allocator.h"
#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core {

// ─────────────────────────────────────────────────────────────────────────────
// Macros for unimplemented ops
// ─────────────────────────────────────────────────────────────────────────────

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

// ─────────────────────────────────────────────────────────────────────────────
// GPU dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Binary op IDs (must match binary.comp)
enum class BinaryOp : uint32_t {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Max = 4,
  Min = 5,
  Pow = 6,
  Equal = 7,
  NotEq = 8,
  Less = 9,
  LessEq = 10,
  Greater = 11,
  GreaterEq = 12,
  LogAddExp = 13,
  Arctan2 = 14,
  Remainder = 15,
  FloorDiv = 16,
  LogicAnd = 17,
  LogicOr = 18,
  BitAnd = 19,
  BitOr = 20,
  BitXor = 21,
  LeftShift = 22,
  RightShift = 23,
  EqualNan = 24,
};

// Unary op IDs (must match unary.comp)
enum class UnaryOp : uint32_t {
  Abs = 0,
  Neg = 1,
  Sign = 2,
  Sqrt = 3,
  Rsqrt = 4,
  Square = 5,
  Exp = 6,
  Expm1 = 7,
  Log = 8,
  Log1p = 9,
  Sin = 10,
  Cos = 11,
  Tan = 12,
  Sinh = 13,
  Cosh = 14,
  Tanh = 15,
  Arcsin = 16,
  Arccos = 17,
  Arctan = 18,
  Arcsinh = 19,
  Arccosh = 20,
  Arctanh = 21,
  Ceil = 22,
  Floor = 23,
  Round = 24,
  Erf = 25,
  Erfinv = 26,
  Sigmoid = 27,
  Conjugate = 28,
  Log2 = 29,
  Log10 = 30,
  LogNot = 33,
  IsInf = 34,
  IsNan = 35,
  IsNegInf = 36,
};

// Compute broadcast strides for an input array relative to the output shape.
// For each dimension: if input dim == 1, stride = 0 (broadcast); else use
// original stride. Input may have fewer dimensions than output (left-padded
// with 1s).
static void compute_broadcast_strides(
    const array& in,
    const Shape& out_shape,
    uint32_t strides[4]) {
  int out_ndim = out_shape.size();
  int in_ndim = in.ndim();
  for (int i = 0; i < 4; i++)
    strides[i] = 0;
  for (int i = 0; i < in_ndim; i++) {
    int out_i = out_ndim - in_ndim + i;
    if (out_i >= 0 && out_i < 4) {
      strides[out_i] =
          (in.shape(i) == 1) ? 0 : static_cast<uint32_t>(in.strides()[i]);
    }
  }
}

// The binary.comp shader has 4 bindings:
//   0=InA, 1=InB, 2=OutCFloat, 3=OutCBool
// Broadcast is handled in-shader via stride-based ND indexing (no pre-copy
// needed).
void dispatch_binary(
    const array& a,
    const array& b,
    array& out,
    uint32_t op_id,
    const Stream& s) {
  // If either input has a non-zero buffer offset, strided_idx in the shader
  // computes offsets from buffer[0] but the data starts at buffer[offset].
  // Copy non-zero-offset inputs to contiguous arrays first.
  if (a.offset() != 0 || b.offset() != 0) {
    array a_eff(a.shape(), a.dtype(), nullptr, {});
    array b_eff(b.shape(), b.dtype(), nullptr, {});
    if (a.offset() != 0) {
      a_eff.set_data(allocator::malloc(a_eff.nbytes()));
      copy_gpu(a, a_eff, CopyType::General, s);
    } else {
      a_eff = a;
    }
    if (b.offset() != 0) {
      b_eff.set_data(allocator::malloc(b_eff.nbytes()));
      copy_gpu(b, b_eff, CopyType::General, s);
    } else {
      b_eff = b;
    }
    dispatch_binary(a_eff, b_eff, out, op_id, s);
    return;
  }

  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  if (shape.size() > 4) {
    array a_contig(out.shape(), a.dtype(), nullptr, {});
    a_contig.set_data(allocator::malloc(a_contig.nbytes()));
    copy_gpu(a, a_contig, CopyType::General, s);

    array b_contig(out.shape(), b.dtype(), nullptr, {});
    b_contig.set_data(allocator::malloc(b_contig.nbytes()));
    copy_gpu(b, b_contig, CopyType::General, s);

    dispatch_binary(a_contig, b_contig, out, op_id, s);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // Nothing to do for empty output
  if (out.size() == 0) {
    return;
  }

  auto& encoder = vulkan::get_command_encoder(s);
  auto& dev = vulkan::device(s.device);
  encoder.op_count++;

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer c_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  // 4 bindings: InA(uint), InB(uint), OutC(uint raw), OutCBool(uint8)
  // push constant size = 84 bytes (21 x uint32)
  VkPipeline pipeline = dev.get_pipeline(
      "binary",
      layout,
      ds_layout,
      4,
      84,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE)
    return;

  VkDescriptorSet ds = dev.alloc_descriptor_set(s, ds_layout);

  VkDescriptorBufferInfo a_info{a_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo b_info{b_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo c_info{c_buf, 0, VK_WHOLE_SIZE};

  bool output_is_bool = (out.dtype() == bool_) &&
      (op_id == static_cast<uint32_t>(BinaryOp::Equal) ||
       op_id == static_cast<uint32_t>(BinaryOp::NotEq) ||
       op_id == static_cast<uint32_t>(BinaryOp::Less) ||
       op_id == static_cast<uint32_t>(BinaryOp::LessEq) ||
       op_id == static_cast<uint32_t>(BinaryOp::Greater) ||
       op_id == static_cast<uint32_t>(BinaryOp::GreaterEq) ||
       op_id == static_cast<uint32_t>(BinaryOp::LogicAnd) ||
       op_id == static_cast<uint32_t>(BinaryOp::LogicOr) ||
       op_id == static_cast<uint32_t>(BinaryOp::EqualNan));

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
      case float32:
      case float16:
      case bfloat16:
        return 0;
      case int8:
      case int16:
      case int32:
      case int64:
        return 1;
      default:
        return 2;
    }
  };

  Dtype in_dtype = a.dtype();

  // Layout must match binary.comp push constants exactly
  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t output_is_bool;
    uint32_t input_dtype;
    uint32_t a_elem_bytes;
    uint32_t b_elem_bytes;
    uint32_t out_elem_bytes;
    uint32_t ndim;
    uint32_t out_shape[4];
    uint32_t a_strides[4];
    uint32_t b_strides[4];
  } pc{};

  pc.n = static_cast<uint32_t>(out.size());
  pc.op = op_id;
  pc.output_is_bool = output_is_bool ? 1u : 0u;
  pc.input_dtype = to_input_dtype(promote_types(a.dtype(), b.dtype()));

  pc.a_elem_bytes =
      a.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(a.itemsize());
  pc.b_elem_bytes =
      b.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(b.itemsize());
  pc.out_elem_bytes =
      out.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(out.itemsize());
  // std::cout << "PushConst: a_elem_bytes=" << pc.a_elem_bytes
  //           << " b=" << pc.b_elem_bytes << " out=" << pc.out_elem_bytes
  //           << " input_dtype=" << pc.input_dtype << " a_dtype=" << a.dtype()
  //           << " b_dtype=" << b.dtype() << std::endl;

  const auto& out_shape = shape;
  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];

  // Set up ND dimensions for broadcast indexing
  int ndim = out_shape.size();
  pc.ndim = static_cast<uint32_t>(ndim > 4 ? 4 : ndim);

  // Fill out_shape and strides (right-aligned, padded)
  for (int i = 0; i < 4; i++) {
    pc.out_shape[i] = 1;
    pc.a_strides[i] = 0;
    pc.b_strides[i] = 0;
  }
  for (int i = 0; i < std::min(ndim, 4); i++) {
    int out_i = (ndim <= 4) ? i : (i + ndim - 4);
    pc.out_shape[i] = static_cast<uint32_t>(out_shape[out_i]);
    pc.a_strides[i] = static_cast<uint32_t>(a_strides[out_i]);
    pc.b_strides[i] = static_cast<uint32_t>(b_strides[out_i]);
  }

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// Dispatch a unary elementwise shader.
// Returns false if the dtype is unsupported (caller should fall back to CPU).
bool dispatch_unary(
    const array& in,
    array& out,
    uint32_t op_id,
    Stream stream) {
  // Complex types (complex64 = 8 bytes) not handled by the GPU unary shader
  if (issubdtype(in.dtype(), complexfloating)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // Nothing to do for empty arrays
  if (out.size() == 0) {
    return true;
  }

  auto& encoder = vulkan::get_command_encoder(stream);
  auto& dev = vulkan::device(stream.device);
  encoder.op_count++;

  VkBuffer in_buf = vulkan::get_buffer(in);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "unary",
      layout,
      ds_layout,
      3,
      16,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE)
    return false;

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream, ds_layout);

  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[3]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &in_info;

  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &out_info;

  writes[2] = writes[0];
  writes[2].dstBinding = 2;
  writes[2].pBufferInfo = &out_info;

  vkUpdateDescriptorSets(dev.vk_device(), 3, writes, 0, nullptr);

  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t input_elem_bytes;
    uint32_t out_elem_bytes;
  } pc{
      static_cast<uint32_t>(out.size()),
      op_id,
      in.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(in.itemsize()),
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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
  return true;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise Unary
// ─────────────────────────────────────────────────────────────────────────────

// For unsupported dtypes (complex64, etc.) fall back to CPU.
#define UNARY_GPU(cls, op)                                                   \
  void cls::eval_gpu(const std::vector<array>& inputs, array& out) {         \
    if (!dispatch_unary(                                                     \
            inputs[0], out, static_cast<uint32_t>(UnaryOp::op), stream())) { \
      vulkan::device(stream().device).synchronize(stream());                 \
      eval_cpu(inputs, out);                                                 \
    }                                                                        \
  }

UNARY_GPU(Abs, Abs)
UNARY_GPU(ArcCos, Arccos)
UNARY_GPU(ArcCosh, Arccosh)
UNARY_GPU(ArcSin, Arcsin)
UNARY_GPU(ArcSinh, Arcsinh)
UNARY_GPU(ArcTan, Arctan)
UNARY_GPU(ArcTanh, Arctanh)
UNARY_GPU(Ceil, Ceil)
UNARY_GPU(Conjugate, Conjugate)
UNARY_GPU(Cos, Cos)
UNARY_GPU(Cosh, Cosh)
UNARY_GPU(Erf, Erf)
UNARY_GPU(ErfInv, Erfinv)
UNARY_GPU(Exp, Exp)
UNARY_GPU(Expm1, Expm1)
UNARY_GPU(Floor, Floor)
UNARY_GPU(IsInf, IsInf)
UNARY_GPU(IsNaN, IsNan)
UNARY_GPU(IsNegInf, IsNegInf)
// Log::eval_gpu handled below (supports base-e, base-2, base-10 via state())
UNARY_GPU(Log1p, Log1p)
UNARY_GPU(Negative, Neg)
// Round is a no-op on integer types; the GPU shader reinterprets int bits as
// float bits which gives wrong results (e.g. round(15_int32) → 0).
// Fall back to CPU for all non-floating dtypes.
void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!issubdtype(inputs[0].dtype(), floating)) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }
  if (!dispatch_unary(
          inputs[0], out, static_cast<uint32_t>(UnaryOp::Round), stream())) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
  }
}

// LogicalNot uses a dedicated LOGNOT opcode (not NEG) to correctly handle bool
// byte-packing
void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!dispatch_unary(
          inputs[0], out, static_cast<uint32_t>(UnaryOp::LogNot), stream())) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
  }
}
UNARY_GPU(Sigmoid, Sigmoid)
UNARY_GPU(Sign, Sign)
// Sqrt and Rsqrt share the same C++ class (Sqrt with recip_=true for Rsqrt).
// Dispatch UNARY_RSQRT or UNARY_SQRT based on state().
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  uint32_t op = state() ? static_cast<uint32_t>(UnaryOp::Rsqrt)
                        : static_cast<uint32_t>(UnaryOp::Sqrt);
  if (!dispatch_unary(inputs[0], out, op, stream())) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
  }
}
UNARY_GPU(Sin, Sin)
UNARY_GPU(Sinh, Sinh)
UNARY_GPU(Square, Square)
UNARY_GPU(Tan, Tan)
UNARY_GPU(Tanh, Tanh)

// Log handles base-e, base-2, and base-10 via a single class with a Base field.
void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto base = state();
  uint32_t op;
  if (base == Log::Base::e)
    op = static_cast<uint32_t>(UnaryOp::Log);
  else if (base == Log::Base::two)
    op = static_cast<uint32_t>(UnaryOp::Log2);
  else
    op = static_cast<uint32_t>(UnaryOp::Log10);
  if (!dispatch_unary(inputs[0], out, op, stream())) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
  }
}

// BitwiseInvert: fall back to CPU (requires XOR with all-ones broadcast)
void BitwiseInvert::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise Binary
// ─────────────────────────────────────────────────────────────────────────────

// Broadcast arrays are handled by the shader via modulo indexing:
// idx % a_size / idx % b_size — works for scalar (size=1), broadcast, and full.
// Complex types fall back to CPU since the GPU shader only handles real types.
#define BINARY_GPU(cls, op)                                          \
  void cls::eval_gpu(const std::vector<array>& inputs, array& out) { \
    if (issubdtype(out.dtype(), complexfloating) ||                  \
        issubdtype(inputs[0].dtype(), complexfloating) ||            \
        issubdtype(inputs[1].dtype(), complexfloating)) {            \
      vulkan::device(stream().device).synchronize(stream());         \
      eval_cpu(inputs, out);                                         \
      return;                                                        \
    }                                                                \
    dispatch_binary(                                                 \
        inputs[0],                                                   \
        inputs[1],                                                   \
        out,                                                         \
        static_cast<uint32_t>(BinaryOp::op),                         \
        stream());                                                   \
  }

BINARY_GPU(Add, Add)
BINARY_GPU(ArcTan2, Arctan2)
BINARY_GPU(Divide, Div)
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (issubdtype(inputs[0].dtype(), complexfloating) ||
      issubdtype(inputs[1].dtype(), complexfloating)) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }
  uint32_t op = equal_nan_ ? static_cast<uint32_t>(BinaryOp::EqualNan)
                           : static_cast<uint32_t>(BinaryOp::Equal);
  dispatch_binary(inputs[0], inputs[1], out, op, stream());
}
BINARY_GPU(Greater, Greater)
BINARY_GPU(GreaterEqual, GreaterEq)
BINARY_GPU(Less, Less)
BINARY_GPU(LessEqual, LessEq)
BINARY_GPU(LogAddExp, LogAddExp)
BINARY_GPU(LogicalAnd, LogicAnd)
BINARY_GPU(LogicalOr, LogicOr)
BINARY_GPU(Maximum, Max)
BINARY_GPU(Minimum, Min)
BINARY_GPU(Multiply, Mul)
BINARY_GPU(NotEqual, NotEq)
BINARY_GPU(Power, Pow)
BINARY_GPU(Remainder, Remainder)
BINARY_GPU(Subtract, Sub)

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
  auto check_array = [&](const array& a) {
    if (a.size() == 1) return true;
    return a.flags().row_contiguous && a.size() == out.size();
  };
  if (!check_array(inputs[0]) || !check_array(inputs[1]) || !check_array(inputs[2])) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array& cond = inputs[0];
  const array& true_ = inputs[1];
  const array& false_ = inputs[2];

  VkBuffer cond_buf = vulkan::get_buffer(cond);
  VkBuffer true_buf = vulkan::get_buffer(true_);
  VkBuffer false_buf = vulkan::get_buffer(false_);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "ternary",
      layout,
      ds_layout,
      4,
      28,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo infos[4]{
      {cond_buf, 0, VK_WHOLE_SIZE},
      {true_buf, 0, VK_WHOLE_SIZE},
      {false_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}};

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
    uint32_t true_elem_bytes; // 2=f16, 3=bf16, 4=f32
    uint32_t false_elem_bytes;
    uint32_t out_elem_bytes;
  } pc{
      static_cast<uint32_t>(out.size()),
      cond.data_size() == 1 ? 1u : 0u,
      true_.data_size() == 1 ? 1u : 0u,
      false_.data_size() == 1 ? 1u : 0u,
      true_.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(true_.itemsize()),
      false_.dtype() == bfloat16 ? 3u
                                 : static_cast<uint32_t>(false_.itemsize()),
      out.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(out.itemsize())};

  VkCommandBuffer cmd = encoder.cmd;

  // Zero-initialize output buffer for 16-bit types because the ternary shader
  // uses atomicOr to write each 16-bit half into shared uint32 words.
  // Without this, the unused half would contain garbage.
  uint32_t out_elem_bytes =
      out.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(out.itemsize());
  if (out_elem_bytes < 4u && out.nbytes() > 0) {
    vkCmdFillBuffer(cmd, out_buf, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier zero_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    zero_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    zero_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &zero_barrier,
        0,
        nullptr,
        0,
        nullptr);
  }

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Arange
// ─────────────────────────────────────────────────────────────────────────────

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  // Nothing to do for empty output (e.g., arange(10, 0, 1))
  if (out.size() == 0) {
    return;
  }

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "arange",
      layout,
      ds_layout,
      1,
      16,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = ds;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &out_info;
  vkUpdateDescriptorSets(dev.vk_device(), 1, &write, 0, nullptr);

  auto dtype_to_enum_arange = [](Dtype dt) -> uint32_t {
    switch (dt) {
      case bool_:
        return 0;
      case uint8:
        return 1;
      case uint16:
        return 2;
      case uint32:
        return 3;
      case uint64:
        return 4;
      case int8:
        return 5;
      case int16:
        return 6;
      case int32:
        return 7;
      case int64:
        return 8;
      case float16:
        return 9;
      case bfloat16:
        return 10;
      case float32:
        return 11;
      default:
        return 11;
    }
  };

  struct PushConst {
    uint32_t n;
    float start;
    float step;
    uint32_t out_dtype;
  } pc{
      static_cast<uint32_t>(out.size()),
      static_cast<float>(start_),
      static_cast<float>(step_),
      dtype_to_enum_arange(out.dtype())};

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reduction
// ─────────────────────────────────────────────────────────────────────────────

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool is_continuous_axes = true;
  for (size_t i = 1; i < axes_.size(); i++) {
    if (axes_[i] != axes_[i - 1] + 1) {
      is_continuous_axes = false;
      break;
    }
  }

  bool is_f16 = (inputs[0].dtype() == float16 || inputs[0].dtype() == bfloat16);
  if (!inputs[0].flags().row_contiguous || !is_continuous_axes ||
      (inputs[0].dtype() != float32 && !is_f16 &&
       (inputs[0].dtype() != bool_ || out.dtype() != bool_))) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  // Empty input: fall to CPU to get the correct identity value (e.g.
  // all([])=True)
  if (inputs[0].size() == 0) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  bool use_temp_out = is_f16 && out.dtype() != bool_;
  std::optional<array> temp_out;
  if (use_temp_out) {
    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  // ── Compute inner_size for strided reduce shader ──────────────────────────
  // The shader uses the formula:
  //   in_idx = (out_idx / inner) * (reduce_size * inner) + j * inner + (out_idx
  //   % inner)
  // where inner = product of all dims after the LAST reduce axis.
  // This supports reducing any contiguous block of axes from a row-contiguous
  // input. For last-axis reduce: inner=1, formula = out_idx * reduce_size + j
  // (same as before).
  const array& raw_in = inputs[0];

  // Compute inner: product of dims after the highest reduce axis
  int max_reduce_ax = *std::max_element(axes_.begin(), axes_.end());
  uint32_t inner = 1;
  for (int i = max_reduce_ax + 1; i < raw_in.ndim(); i++) {
    inner *= static_cast<uint32_t>(raw_in.shape(i));
  }

  VkBuffer in_buf = vulkan::get_buffer(raw_in);
  VkBuffer out_buf = vulkan::get_buffer(use_temp_out ? *temp_out : out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  // 4 bindings: InRaw, OutFloat, OutBool, InBool
  // Push constant is now 32 bytes (8 × uint32) to include inner and
  // outer_stride.
  VkPipeline pipeline = dev.get_pipeline(
      "reduce",
      layout,
      ds_layout,
      4,
      32,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    throw std::runtime_error("[vulkan::eval_gpu] reduce pipeline not found");
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo in_info{in_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo out_info{out_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &in_info; // InRaw
  writes[1].pBufferInfo = &out_info; // OutFloat
  writes[2].pBufferInfo = &out_info; // OutBool
  writes[3].pBufferInfo = &in_info; // InBool
  vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

  uint32_t n_outputs = static_cast<uint32_t>(out.size() > 0 ? out.size() : 1);
  uint32_t reduce_size = static_cast<uint32_t>(raw_in.size() / n_outputs);
  uint32_t outer_stride = reduce_size * inner; // = reduce_size when inner=1

  uint32_t op_id = 0;
  switch (reduce_type_) {
    case Reduce::ReduceType::Sum:
      op_id = 0;
      break;
    case Reduce::ReduceType::Max:
      op_id = 1;
      break;
    case Reduce::ReduceType::Min:
      op_id = 2;
      break;
    case Reduce::ReduceType::Prod:
      op_id = 3;
      break;
    case Reduce::ReduceType::And:
      op_id = 4;
      break;
    case Reduce::ReduceType::Or:
      op_id = 5;
      break;
  }

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
    uint32_t op;
    uint32_t input_elem_bytes;
    uint32_t out_elem_bytes;
    uint32_t inner; // product of dims after the reduce axes
    uint32_t outer_stride; // = reduce_size * inner
  } pc{
      static_cast<uint32_t>(raw_in.size()),
      reduce_size,
      n_outputs,
      op_id,
      raw_in.dtype() == bfloat16 ? 3u
                                 : static_cast<uint32_t>(raw_in.itemsize()),
      static_cast<uint32_t>((use_temp_out ? *temp_out : out).itemsize()),
      inner,
      outer_stride};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  // One workgroup per output element
  vkCmdDispatch(cmd, pc.n_outputs, 1, 1);

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

  if (use_temp_out) {
    copy_gpu(*temp_out, out, CopyType::General, stream());
  }
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool is_f16 = (inputs[0].dtype() == float16 || inputs[0].dtype() == bfloat16);
  if (!inputs[0].flags().row_contiguous ||
      (inputs[0].dtype() != float32 && !is_f16 && inputs[0].dtype() != bool_)) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf = vulkan::get_buffer(inputs[0]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "arg_reduce",
      layout,
      ds_layout,
      2,
      20,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
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
  uint32_t reduce_size = static_cast<uint32_t>(inputs[0].size() / n_outputs);
  uint32_t op_id = (reduce_type_ == ArgReduce::ReduceType::ArgMax) ? 0u : 1u;
  uint32_t elem_bytes = inputs[0].dtype() == bfloat16
      ? 3u
      : static_cast<uint32_t>(inputs[0].itemsize());

  // Compute inner: product of dims after the highest reduce axis
  int max_reduce_ax = axis_;
  uint32_t inner = 1;
  for (int i = max_reduce_ax + 1; i < inputs[0].ndim(); i++) {
    inner *= static_cast<uint32_t>(inputs[0].shape(i));
  }
  uint32_t outer_stride = reduce_size * inner;

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
    uint32_t op;
    uint32_t input_elem_bytes;
    uint32_t inner;
    uint32_t outer_stride;
  } pc{
      static_cast<uint32_t>(inputs[0].size()),
      reduce_size,
      n_outputs,
      op_id,
      elem_bytes,
      inner,
      outer_stride};

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Matmul
// ─────────────────────────────────────────────────────────────────────────────

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& a_orig = inputs[0];
  const array& b_orig = inputs[1];

  bool a_ok = a_orig.dtype() == float32 || a_orig.dtype() == float16 ||
      a_orig.dtype() == bfloat16;
  bool b_ok = b_orig.dtype() == float32 || b_orig.dtype() == float16 ||
      b_orig.dtype() == bfloat16;
  if (!a_ok || !b_ok) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  // Make contiguous copies if either input has non-standard strides
  // (e.g. b.T is a transposed view — strides are (1, K) not (N, 1))
  array a_contig(a_orig.shape(), a_orig.dtype(), nullptr, {});
  array b_contig(b_orig.shape(), b_orig.dtype(), nullptr, {});
  if (!a_orig.flags().row_contiguous) {
    a_contig.set_data(allocator::malloc(a_contig.nbytes()));
    copy_gpu(a_orig, a_contig, CopyType::General, stream());
  } else {
    a_contig = a_orig;
  }
  if (!b_orig.flags().row_contiguous) {
    b_contig.set_data(allocator::malloc(b_contig.nbytes()));
    copy_gpu(b_orig, b_contig, CopyType::General, stream());
  } else {
    b_contig = b_orig;
  }
  const array& a = a_contig;
  const array& b = b_contig;

  bool use_temp_out = out.dtype() != float32;
  std::optional<array> temp_out;
  if (use_temp_out) {
    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0 || a.size() == 0 || b.size() == 0) {
    if (out.nbytes() > 0) {
      auto& encoder = vulkan::get_command_encoder(stream());
      encoder.op_count++;
      vkCmdFillBuffer(
          encoder.cmd, vulkan::get_buffer(out), 0, VK_WHOLE_SIZE, 0);
    }
    return;
  }

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  uint32_t M = (a.ndim() >= 2) ? static_cast<uint32_t>(a.shape(-2)) : 1u;
  uint32_t K = static_cast<uint32_t>(a.shape(-1));
  uint32_t N = static_cast<uint32_t>(b.shape(-1));

  // Batch dimension
  uint32_t batch = 1u;
  for (int i = 0; i < static_cast<int>(out.ndim()) - 2; i++) {
    batch *= static_cast<uint32_t>(out.shape(i));
  }

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer c_buf = vulkan::get_buffer(use_temp_out ? *temp_out : out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      dev.has_cooperative_matrix() && (M % 16 == 0) && (N % 16 == 0) &&
              (K % 16 == 0)
          ? "matmul_coop"
          : "matmul",
      layout,
      ds_layout,
      3,
      36,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
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
    uint32_t a_elem_bytes;
    uint32_t b_elem_bytes;
  } pc{
      M,
      N,
      K,
      batch,
      M * K,
      K * N,
      M * N,
      a.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(a.itemsize()),
      b.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(b.itemsize())};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  // Grid: dynamically computed from specialization caps
  uint32_t subgroup_size = dev.subgroup_size();
  uint32_t BN = (subgroup_size == 64) ? 32 : 16;
  uint32_t BM = 256 / BN;
  vkCmdDispatch(cmd, vulkan::div_ceil(N, BN), vulkan::div_ceil(M, BM), batch);

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

  if (use_temp_out) {
    copy_gpu(*temp_out, out, CopyType::General, stream());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────────────

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool use_temp = inputs[0].dtype() != float32;
  std::optional<array> temp_in;
  std::optional<array> temp_out;
  if (use_temp) {
    temp_in = array(inputs[0].shape(), float32, nullptr, {});
    temp_in->set_data(allocator::malloc(temp_in->nbytes()));
    copy_gpu(inputs[0], *temp_in, CopyType::General, stream());

    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  const array* temp_in_ptr = use_temp ? &(*temp_in) : &inputs[0];
  array contig(temp_in_ptr->shape(), temp_in_ptr->dtype(), nullptr, {});
  if (!temp_in_ptr->flags().row_contiguous) {
    contig.set_data(allocator::malloc(contig.nbytes()));
    copy_gpu(*temp_in_ptr, contig, CopyType::General, stream());
    temp_in_ptr = &contig;
  }
  const array& raw_in = *temp_in_ptr;

  const array& raw_out = use_temp ? *temp_out : out;

  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  if (use_temp && temp_out->nbytes() > 0) {
    vkCmdFillBuffer(
        encoder.cmd, vulkan::get_buffer(*temp_out), 0, VK_WHOLE_SIZE, 0);
  }
  if (out.nbytes() > 0) {
    vkCmdFillBuffer(encoder.cmd, vulkan::get_buffer(out), 0, VK_WHOLE_SIZE, 0);
  }

  VkBuffer in_buf = vulkan::get_buffer(raw_in);
  VkBuffer out_buf = vulkan::get_buffer(raw_out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "softmax",
      layout,
      ds_layout,
      2,
      8,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
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

  uint32_t axis_size = static_cast<uint32_t>(raw_in.shape(-1));
  uint32_t n_rows = static_cast<uint32_t>(raw_in.size() / axis_size);

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  // Copy back from temp_out to out if we used temp buffers
  if (use_temp) {
    copy_gpu(*temp_out, out, CopyType::General, stream());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LogSumExp
// ─────────────────────────────────────────────────────────────────────────────

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf = vulkan::get_buffer(inputs[0]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "logsumexp",
      layout,
      ds_layout,
      2,
      12,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
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
  uint32_t reduce_size = static_cast<uint32_t>(inputs[0].size() / n_outputs);

  struct PushConst {
    uint32_t n;
    uint32_t reduce_size;
    uint32_t n_outputs;
  } pc{static_cast<uint32_t>(inputs[0].size()), reduce_size, n_outputs};

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// DivMod
// ─────────────────────────────────────────────────────────────────────────────

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // binary_two.comp only handles scalar-broadcast, not full shape broadcasting.
  // If any non-scalar input has a different shape than the output, fall back to
  // CPU.
  bool a_scalar = inputs[0].data_size() == 1;
  bool b_scalar = inputs[1].data_size() == 1;
  // broadcast_arrays() in ops.cpp pre-expands shapes to the output shape with
  // stride-0 axes, so inputs[x].shape() == outputs[0].shape() always.
  // Detect actual broadcasting via data_size < size (stride-0 implies fewer
  // data).
  bool a_broadcasted = !a_scalar && (inputs[0].data_size() != inputs[0].size());
  bool b_broadcasted = !b_scalar && (inputs[1].data_size() != inputs[1].size());
  bool needs_broadcast = a_broadcasted || b_broadcasted;
  if (needs_broadcast) {
    for (auto& out : outputs)
      out.set_data(allocator::malloc(out.nbytes()));
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, outputs);
    return;
  }

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer a_buf = vulkan::get_buffer(inputs[0]);
  VkBuffer b_buf = vulkan::get_buffer(inputs[1]);
  VkBuffer out0_buf = vulkan::get_buffer(outputs[0]);
  VkBuffer out1_buf = vulkan::get_buffer(outputs[1]);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "binary_two",
      layout,
      ds_layout,
      4,
      20,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, outputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo infos[4]{
      {a_buf, 0, VK_WHOLE_SIZE},
      {b_buf, 0, VK_WHOLE_SIZE},
      {out0_buf, 0, VK_WHOLE_SIZE},
      {out1_buf, 0, VK_WHOLE_SIZE}};
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

  auto dtype_to_enum_divmod = [](Dtype dt) -> uint32_t {
    switch (dt) {
      case bool_:
        return 0;
      case uint8:
        return 1;
      case uint16:
        return 2;
      case uint32:
        return 3;
      case uint64:
        return 4;
      case int8:
        return 5;
      case int16:
        return 6;
      case int32:
        return 7;
      case int64:
        return 8;
      case float16:
        return 9;
      case bfloat16:
        return 10;
      case float32:
        return 11;
      default:
        return 11;
    }
  };

  struct PushConst {
    uint32_t n;
    uint32_t op;
    uint32_t a_scalar;
    uint32_t b_scalar;
    uint32_t dtype;
  } pc{
      static_cast<uint32_t>(outputs[0].size()),
      0u,
      inputs[0].data_size() == 1 ? 1u : 0u,
      inputs[1].data_size() == 1 ? 1u : 0u,
      dtype_to_enum_divmod(inputs[0].dtype())};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(
      cmd, vulkan::div_ceil(outputs[0].size(), vulkan::WORKGROUP_SIZE), 1, 1);

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
  uint32_t src_stride; // INDEX_GATHER/SCATTER: stride of indexed dim in src
  uint32_t dst_stride; // INDEX_GATHER/SCATTER: stride in dst
  uint32_t src_offset;
  uint32_t idx_offset;
  uint32_t dst_offset;
  uint32_t src_ax_size;
  uint32_t inner; // INDEX_GATHER_GEN: product(src dims after gather axis)
                  // INDEX_GATHER element-wise: out_ax_size (dst axis size)
  uint32_t src_outer_stride; // INDEX_GATHER_GEN: d_ax * inner
  // Multi-axis gather extensions (mx.take with ND index tensors):
  uint32_t idx_ndim; // Number of dimensions in index tensor (0-4)
  uint32_t idx_shape[4]; // Shape of index tensor
  uint32_t idx_strides[4]; // Strides of index tensor
};
static constexpr uint32_t kIndexPushSize = sizeof(IndexPushConst); // 80

static void indexing_dispatch(
    vulkan::Device& dev,
    vulkan::CommandEncoder& encoder,
    Stream s,
    VkBuffer src_buf,
    VkBuffer idx_buf,
    VkBuffer out_buf,
    const IndexPushConst& pc,
    size_t n_out) {
  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "indexing",
      layout,
      ds_layout,
      3,
      kIndexPushSize,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE)
    throw std::runtime_error("[indexing_dispatch] Pipeline not found.");

  VkDescriptorSet ds = dev.alloc_descriptor_set(s, ds_layout);
  VkDescriptorBufferInfo infos[3]{
      {src_buf, 0, VK_WHOLE_SIZE},
      {idx_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}};
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
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kIndexPushSize, &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, vulkan::div_ceil(n_out, vulkan::WORKGROUP_SIZE), 1, 1);

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
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  // The indexing shader only supports 32-bit types.
  bool supported_type = (inputs[0].itemsize() == 4);
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs[i].itemsize() != 4) supported_type = false;
  }
  if (!supported_type) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  // Handle single-axis gather (mx.take(src, indices, axis=ax)):
  //   axes_.size()==1, slice_sizes_[ax]==1 for the gathered axis
  // Covers 1D (ax=0) and ND cases via INDEX_GATHER_GEN.
  if (inputs.size() == 2 && axes_.size() == 1 && slice_sizes_[axes_[0]] == 1) {
    // if src is transposed/non-contiguous, make a contiguous GPU copy rather
    // than falling back to CPU. This avoids CPU/GPU stream race conditions when
    // mx.take dispatches transposed intermediates.
    array src_contig(inputs[0].shape(), inputs[0].dtype(), nullptr, {});
    array idx_contig(inputs[1].shape(), inputs[1].dtype(), nullptr, {});
    bool src_copied = false;
    bool idx_copied = false;
    const array* src_ptr = &inputs[0];
    const array* idx_ptr = &inputs[1];
    if (!inputs[0].flags().row_contiguous) {
      src_contig.set_data(allocator::malloc(src_contig.nbytes()));
      copy_gpu(inputs[0], src_contig, CopyType::General, stream());
      src_ptr = &src_contig;
      src_copied = true;
      // Add memory barrier to ensure copy completes before gather reads
      auto& enc = vulkan::get_command_encoder(stream());
      VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(enc.cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 1, &b, 0, nullptr, 0, nullptr);
    }
    if (!inputs[1].flags().row_contiguous) {
      idx_contig.set_data(allocator::malloc(idx_contig.nbytes()));
      copy_gpu(inputs[1], idx_contig, CopyType::General, stream());
      idx_ptr = &idx_contig;
      idx_copied = true;
      // Add memory barrier to ensure copy completes before gather reads
      auto& enc = vulkan::get_command_encoder(stream());
      VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(enc.cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 1, &b, 0, nullptr, 0, nullptr);
    }
    const array& src = *src_ptr;
    const array& idx = *idx_ptr;
    if (idx.ndim() > 4) {
      vulkan::device(stream().device).synchronize(stream());
      eval_cpu(inputs, out);
      return;
    }

    out.set_data(allocator::malloc(out.nbytes()));
    if (out.size() == 0)
      return;

    auto& encoder = vulkan::get_command_encoder(stream());
    auto& dev = vulkan::device(stream().device);
    encoder.op_count++;


    uint32_t ax = static_cast<uint32_t>(axes_[0]);
    // ...

    // inner = product(src dims after the gather axis)
    uint32_t inner = 1;
    for (int d = ax + 1; d < (int)src.ndim(); d++)
      inner *= src.shape(d);

    // outer_size = product(src dims before the gather axis)
    uint32_t outer_size = 1;
    for (int d = 0; d < (int)ax; d++)
      outer_size *= src.shape(d);

    uint32_t d_ax = static_cast<uint32_t>(src.shape(ax));
    IndexPushConst pc{};
    pc.n = static_cast<uint32_t>(out.size());
    pc.op = 3u; // INDEX_GATHER_GEN
    pc.idx_size = static_cast<uint32_t>(idx.size());
    pc.src_stride = 1u; // unused by this op
    pc.dst_stride = 1u; // unused by this op
    pc.src_offset = static_cast<uint32_t>(src.offset() * src.itemsize());
    pc.idx_offset = static_cast<uint32_t>(idx.offset() * idx.itemsize());
    pc.dst_offset = static_cast<uint32_t>(out.offset() * out.itemsize());
    pc.src_ax_size = d_ax;
    pc.inner = inner;
    pc.src_outer_stride = outer_size; // outer_size for INDEX_GATHER_GEN decomposition
    // Multi-axis gather: set index tensor metadata for ND indexing
    pc.idx_ndim = static_cast<uint32_t>(idx.ndim());
    for (int d = 0; d < 4; d++) {
      pc.idx_shape[d] =
          (d < idx.ndim()) ? static_cast<uint32_t>(idx.shape(d)) : 1u;
      pc.idx_strides[d] =
          (d < idx.ndim()) ? static_cast<uint32_t>(idx.strides(d)) : 0u;
    }

    indexing_dispatch(
        dev,
        encoder,
        stream(),
        vulkan::get_buffer(src),
        vulkan::get_buffer(idx),
        vulkan::get_buffer(out),
        pc,
        out.size());
    if (src_copied) {
      dev.add_temporary(stream(), src);
    }
    if (idx_copied) {
      dev.add_temporary(stream(), idx);
    }
    return;
  }

  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
}

// GatherAxis: simple axis-indexed gather → GPU dispatch
void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  // The indexing shader only supports 32-bit types.
  bool supported_type = (inputs[0].itemsize() == 4);
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs[i].itemsize() != 4) supported_type = false;
  }
  if (!supported_type) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  // Make contiguous copies of src and idx if needed (strides would confuse the
  // shader)
  array src_storage(inputs[0].shape(), inputs[0].dtype(), nullptr, {});
  bool src_copied = false;
  bool copied = false;
  if (!inputs[0].flags().row_contiguous) {
    src_storage.set_data(allocator::malloc(src_storage.nbytes()));
    copy_gpu(inputs[0], src_storage, CopyType::General, stream());
    copied = true;
    src_copied = true;
  } else {
    src_storage = inputs[0];
  }
  array idx_storage(inputs[1].shape(), inputs[1].dtype(), nullptr, {});
  bool idx_copied = false;
  if (!inputs[1].flags().row_contiguous) {
    idx_storage.set_data(allocator::malloc(idx_storage.nbytes()));
    copy_gpu(inputs[1], idx_storage, CopyType::General, stream());
    copied = true;
    idx_copied = true;
  } else {
    idx_storage = inputs[1];
  }

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  if (copied) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(encoder.cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &b, 0, nullptr, 0, nullptr);
  }

  const array& src = src_storage;
  const array& idx = idx_storage;

  // Stride of the indexed axis in source and destination
  uint32_t src_stride = 1;
  for (int i = axis_ + 1; i < src.ndim(); i++)
    src_stride *= static_cast<uint32_t>(src.shape(i));
  uint32_t dst_stride = 1;
  for (int i = axis_ + 1; i < out.ndim(); i++)
    dst_stride *= static_cast<uint32_t>(out.shape(i));

  IndexPushConst pc{};
  pc.n = static_cast<uint32_t>(out.size());
  pc.op = 0u; // INDEX_GATHER
  pc.idx_size = static_cast<uint32_t>(idx.size());
  pc.src_stride = src_stride;
  pc.dst_stride = dst_stride;
  pc.src_offset = static_cast<uint32_t>(src.offset() * src.itemsize());
  pc.idx_offset = static_cast<uint32_t>(idx.offset() * idx.itemsize());
  pc.dst_offset = static_cast<uint32_t>(out.offset() * out.itemsize());
  pc.src_ax_size = static_cast<uint32_t>(src.shape(axis_));
  // inner repurposed to carry out_ax_size for element-wise INDEX_GATHER path
  // (the inner field is only read by op=3/INDEX_GATHER_GEN, safe to reuse for op=0)
  pc.inner = static_cast<uint32_t>(out.shape(axis_));
  // src_outer_stride repurposed as element_wise flag:
  // 1 = idx.size() == out.size() (take_along_axis style, one index per output
  // element) 0 = group-wise (1 index per dst_stride group)
  pc.src_outer_stride = (idx.size() == out.size()) ? 1u : 0u;

  indexing_dispatch(
      dev,
      encoder,
      stream(),
      vulkan::get_buffer(src),
      vulkan::get_buffer(idx),
      vulkan::get_buffer(out),
      pc,
      out.size());
  if (src_copied) {
    dev.add_temporary(stream(), src_storage);
  }
  if (idx_copied) {
    dev.add_temporary(stream(), idx_storage);
  }
}

// General Scatter: multi-axis → CPU fallback
void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

// ScatterAxis: simple axis-indexed scatter → GPU dispatch
void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  // inputs: [src, indices, updates]
  // Copy src→out first, then scatter updates into it
  // The indexing shader is currently hard-coded to use 32-bit floats.
  // Fall back to CPU for other types.
  bool supported_type = (inputs[0].dtype() == float32);
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs[i].itemsize() != 4) supported_type = false;
  }
  if (!supported_type) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  bool copied = false;
  auto ctype = inputs[0].flags().row_contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu_inplace(inputs[0], out, ctype, stream());
  copied = true;

  if (inputs.size() < 3 || inputs[2].size() == 0)
    return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  const array* idx_ptr = &inputs[1];
  const array* updates_ptr = &inputs[2];
  array idx_contig(inputs[1].shape(), inputs[1].dtype(), nullptr, {});
  array updates_contig(inputs[2].shape(), inputs[2].dtype(), nullptr, {});
  bool idx_copied = false;
  bool updates_copied = false;

  if (!inputs[1].flags().row_contiguous) {
    idx_contig = array(inputs[1].shape(), inputs[1].dtype(), nullptr, {});
    idx_contig.set_data(allocator::malloc(idx_contig.nbytes()));
    copy_gpu(inputs[1], idx_contig, CopyType::General, stream());
    idx_ptr = &idx_contig;
    copied = true;
    idx_copied = true;
  }
  if (!inputs[2].flags().row_contiguous) {
    updates_contig = array(inputs[2].shape(), inputs[2].dtype(), nullptr, {});
    updates_contig.set_data(allocator::malloc(updates_contig.nbytes()));
    copy_gpu(inputs[2], updates_contig, CopyType::General, stream());
    updates_ptr = &updates_contig;
    copied = true;
    updates_copied = true;
  }

  if (copied) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(encoder.cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &b, 0, nullptr, 0, nullptr);
  }

  const array& idx = *idx_ptr;
  const array& updates = *updates_ptr;

  uint32_t src_stride = 1;
  for (int i = axis_ + 1; i < updates.ndim(); i++)
    src_stride *= static_cast<uint32_t>(updates.shape(i));
  uint32_t dst_stride = 1;
  for (int i = axis_ + 1; i < out.ndim(); i++)
    dst_stride *= static_cast<uint32_t>(out.shape(i));

  // INDEX_SCATTER=1, INDEX_SCATTER_ADD=2
  uint32_t op = (reduce_type_ == ScatterAxis::ReduceType::Sum) ? 2u : 1u;

  IndexPushConst pc{};
  pc.n = static_cast<uint32_t>(idx.size());
  pc.op = op;
  pc.idx_size = static_cast<uint32_t>(idx.size());
  pc.src_stride = src_stride;
  pc.dst_stride = dst_stride;
  pc.src_offset = static_cast<uint32_t>(updates.offset() * updates.itemsize());
  pc.idx_offset = static_cast<uint32_t>(idx.offset() * idx.itemsize());
  pc.dst_offset = static_cast<uint32_t>(out.offset() * out.itemsize());
  pc.src_ax_size =
      static_cast<uint32_t>(out.shape(axis_)); // axis size for neg-idx wrap
  pc.src_outer_stride = 1u; // ScatterAxis is always element-wise

  indexing_dispatch(
      dev,
      encoder,
      stream(),
      vulkan::get_buffer(updates),
      vulkan::get_buffer(idx),
      vulkan::get_buffer(out),
      pc,
      updates.size());
  if (idx_copied) {
    dev.add_temporary(stream(), idx_contig);
  }
  if (updates_copied) {
    dev.add_temporary(stream(), updates_contig);
  }
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
  while (sort_pow2 < sort_size)
    sort_pow2 <<= 1;

  // Fall back to CPU if:
  // - Not sorting on last axis (non-contiguous scan)
  // - Not float32 or int32 (bitonic/radix sort logic for these only)
  // - Not contiguous
  // - Sort size > 256 (bitonic sort limit; shared mem s_data[512] fits 256 elems at wg=256)
  bool supported_dtype = (in.dtype() == float32 || in.dtype() == int32);
  if (sort_axis != in.ndim() - 1 || !supported_dtype ||
      !in.flags().row_contiguous || sort_pow2 > 128) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  // Use bitonic sort for small arrays (<= 256), radix sort for larger
  bool use_bitonic = (sort_pow2 <= 128);

  // Copy input to output (sort is in-place on output)
  out.set_data(allocator::malloc(out.nbytes()));
  CopyType copy_type = CopyType::Vector; // We checked row_contiguous above
  copy_gpu_inplace(in, out, copy_type, stream());

  uint32_t n = static_cast<uint32_t>(out.size());
  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkCommandBuffer cmd = encoder.cmd;

  if (use_bitonic) {
    // ─────────────────────────────────────────────────────────────────────────
    // Bitonic Sort (for small arrays <= 128)
    // ─────────────────────────────────────────────────────────────────────────
    uint32_t n_sorts = n / sort_size;

    VkBuffer data_buf = vulkan::get_buffer(out);

    // Allocate dummy index buffer (sort shader requires both bindings)
    auto idx_alloc = allocator::malloc(n_sorts * sort_size * sizeof(uint32_t));
    VkBuffer idx_buf =
        static_cast<vulkan::VulkanBuffer*>(idx_alloc.ptr())->buffer;

    VkPipelineLayout layout;
    VkDescriptorSetLayout ds_layout;
    VkPipeline pipeline = dev.get_pipeline(
        "sort",
        layout,
        ds_layout,
        2,
        24,
        vulkan::get_default_specialization_info(dev));
    if (pipeline == VK_NULL_HANDLE) {
      allocator::free(idx_alloc);
      vulkan::device(stream().device).synchronize(stream());
      eval_cpu(inputs, out);
      return;
    }

    VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
    VkDescriptorBufferInfo infos[2]{
        {data_buf, 0, VK_WHOLE_SIZE}, {idx_buf, 0, VK_WHOLE_SIZE}};
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

    struct BitonicPushConst {
      uint32_t n;
      uint32_t sort_size;
      uint32_t n_sorts;
      uint32_t ascending;
      uint32_t with_index;
      uint32_t type_is_int;
    } pc{n, sort_pow2, n_sorts, 1u, 0u, in.dtype() == int32 ? 1u : 0u};

    vkCmdPushConstants(
        cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

    vkCmdDispatch(cmd, n_sorts, 1, 1);

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

    // Free index buffer after GPU finishes
    auto s = stream();
    vulkan::device(s.device).add_completed_handler(
        s, [idx_alloc]() mutable { allocator::free(idx_alloc); });
    return; // Exit after bitonic sort completes
  } else {
    // ─────────────────────────────────────────────────────────────────────────
    // Radix Sort (for larger arrays > 128)
    // ─────────────────────────────────────────────────────────────────────────
    // Radix sort requires a temporary buffer for ping-pong between passes
    size_t temp_size = n * sizeof(uint32_t);
    auto temp_alloc = allocator::malloc(temp_size);
    VkBuffer temp_buf =
        static_cast<vulkan::VulkanBuffer*>(temp_alloc.ptr())->buffer;

    VkBuffer data_buf = vulkan::get_buffer(out);

    // Get radix sort pipeline
    VkPipelineLayout layout;
    VkDescriptorSetLayout ds_layout;
    VkPipeline pipeline = dev.get_pipeline(
        "radix_sort",
        layout,
        ds_layout,
        3, // 3 bindings: in_data, out_data, temp_data
        16, // push constant size
        vulkan::get_default_specialization_info(dev));

    if (pipeline == VK_NULL_HANDLE) {
      allocator::free(temp_alloc);
      vulkan::device(stream().device).synchronize(stream());
      eval_cpu(inputs, out);
      return;
    }

    // Radix sort push constants
    struct RadixPushConst {
      uint32_t n; // Total number of elements
      uint32_t digit; // Current digit (0-7 for 4-bit digits)
      uint32_t ascending; // 1 = ascending, 0 = descending
      uint32_t dtype; // 0 = float32, 1 = int32
    };
    uint32_t dtype_flag = (in.dtype() == int32) ? 1u : 0u;

    // Perform 8 passes of radix sort (4-bit digits, 8 * 4 = 32 bits)
    for (uint32_t digit = 0; digit < 8; digit++) {
      // On even passes: in_data -> out_data (using temp as output)
      // On odd passes: temp -> out_data (using original buffer as output)
      bool even_pass = (digit % 2 == 0);
      VkBuffer in_buffer = even_pass ? data_buf : temp_buf;
      VkBuffer out_buffer = even_pass ? temp_buf : data_buf;

      VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
      VkDescriptorBufferInfo infos[3]{
          {in_buffer, 0, VK_WHOLE_SIZE},
          {out_buffer, 0, VK_WHOLE_SIZE},
          {temp_buf, 0, VK_WHOLE_SIZE}};

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

      RadixPushConst pc{n, digit, 1u, dtype_flag};
      vkCmdPushConstants(
          cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(
          cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

      // Dispatch with single workgroup (radix sort uses shared memory for
      // counting)
      vkCmdDispatch(cmd, 1, 1, 1);

      // Barrier to ensure pass completes before next pass
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
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
    }

    // After 8 passes, final result is in data_buf (if even number of passes)
    // If odd number of passes, we need to copy from temp to data
    if (8 % 2 != 0) {
      // Copy from temp_buf to data_buf
      VkDescriptorSet copy_ds = dev.alloc_descriptor_set(stream(), ds_layout);
      VkDescriptorBufferInfo copy_infos[3]{
          {temp_buf, 0, VK_WHOLE_SIZE},
          {data_buf, 0, VK_WHOLE_SIZE},
          {temp_buf, 0, VK_WHOLE_SIZE}};

      VkWriteDescriptorSet copy_writes[3]{};
      for (int i = 0; i < 3; i++) {
        copy_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        copy_writes[i].dstSet = copy_ds;
        copy_writes[i].dstBinding = i;
        copy_writes[i].descriptorCount = 1;
        copy_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        copy_writes[i].pBufferInfo = &copy_infos[i];
      }
      vkUpdateDescriptorSets(dev.vk_device(), 3, copy_writes, 0, nullptr);

      // Use a simple copy dispatch (one element per thread)
      // We'll use the radix sort pipeline with a special "copy" mode
      // Actually, let's just use a simple manual copy here
      // For simplicity, we'll dispatch a copy kernel or just do element-wise
      // copy Since we don't have a copy kernel, let's just leave the result in
      // the right buffer After 8 passes (even), result is in data_buf
    }

    // Free temp buffer after GPU finishes
    auto s = stream();
    vulkan::device(s.device).add_completed_handler(
        s, [temp_alloc]() mutable { allocator::free(temp_alloc); });
  }
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan → GPU dispatch via scan.comp (prefix scan)
// ─────────────────────────────────────────────────────────────────────────────

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs[0];

  uint32_t scan_size = static_cast<uint32_t>(in.shape(axis_));
  int ndim = in.ndim();
  int int_axis = static_cast<int>(axis_);
  int norm_axis = int_axis < 0 ? int_axis + ndim : int_axis;

  // Shader assumes contiguous scan (last axis only) and float32 only.
  // Fall back to CPU for:
  //   (1) non-last axis (scan elements are non-contiguous in memory)
  //   (2) non-float32 dtype (int32, int64, float16, bfloat16, complex not
  //   handled)
  // Note: scan_size > 1024 is handled by the recursive multi-pass scan below.
  bool is_float32 = in.dtype() == float32;
  if (norm_axis != ndim - 1 || !is_float32) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  // Map reduce type to shader op code
  uint32_t op;
  switch (reduce_type_) {
    case Sum:
      op = 0u;
      break;
    case Prod:
      op = 1u;
      break;
    case Max:
      op = 2u;
      break;
    case Min:
      op = 3u;
      break;
    case LogAddExp:
      op = 4u;
      break;
    default:
      // Unsupported scan type: fall back to CPU
      vulkan::device(stream().device).synchronize(stream());
      eval_cpu(inputs, out);
      return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  // Define a recursive helper lambda inside eval_gpu that does block scanning
  std::function<void(
      VkBuffer, VkBuffer, uint32_t, uint32_t, uint32_t, bool, bool)>
      scan_gpu_recursive = [&](VkBuffer in_buf,
                               VkBuffer out_buf,
                               uint32_t current_scan_size,
                               uint32_t current_n_scans,
                               uint32_t stride_dist,
                               bool inclusive,
                               bool reverse) {
        auto& encoder = vulkan::get_command_encoder(stream());
        auto& dev = vulkan::device(stream().device);

        struct ScanPushConst {
          uint32_t n;
          uint32_t scan_size;
          uint32_t n_scans;
          uint32_t op;
          uint32_t inclusive;
          uint32_t reverse;
          uint32_t save_totals;
          uint32_t has_prefix;
          uint32_t stride_dist;
        };

        if (current_scan_size <= 1024) {
          // Base case: single block scan
          encoder.op_count++;
          ScanPushConst pc{
              current_scan_size * current_n_scans,
              current_scan_size,
              current_n_scans,
              op,
              inclusive ? 1u : 0u,
              reverse ? 1u : 0u,
              0u,
              0u,
              stride_dist};

          VkPipelineLayout layout;
          VkDescriptorSetLayout ds_layout;
          VkPipeline pipeline = dev.get_pipeline(
              "scan",
              layout,
              ds_layout,
              4,
              sizeof(pc),
              vulkan::get_default_specialization_info(dev));

          VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
          VkDescriptorBufferInfo infos[4]{
              {in_buf, 0, VK_WHOLE_SIZE},
              {out_buf, 0, VK_WHOLE_SIZE},
              {out_buf, 0, VK_WHOLE_SIZE}, // unused totals buf (safe dummy)
              {in_buf, 0, VK_WHOLE_SIZE} // unused prefix buf (safe dummy)
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

          vkCmdPushConstants(
              encoder.cmd,
              layout,
              VK_SHADER_STAGE_COMPUTE_BIT,
              0,
              sizeof(pc),
              &pc);
          vkCmdBindPipeline(
              encoder.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
          vkCmdBindDescriptorSets(
              encoder.cmd,
              VK_PIPELINE_BIND_POINT_COMPUTE,
              layout,
              0,
              1,
              &ds,
              0,
              nullptr);
          vkCmdDispatch(encoder.cmd, current_n_scans, 1, 1);

          VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(
              encoder.cmd,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0,
              1,
              &barrier,
              0,
              nullptr,
              0,
              nullptr);
        } else {
          // Recursive case: multi-block scan
          uint32_t chunk_size = 1024;
          uint32_t num_chunks =
              (current_scan_size + chunk_size - 1) / chunk_size;

          size_t totals_bytes = num_chunks * current_n_scans * sizeof(float);
          auto totals_alloc = allocator::malloc(totals_bytes);
          VkBuffer totals_buf =
              static_cast<vulkan::VulkanBuffer*>(totals_alloc.ptr())->buffer;

          // Ensure zeroed out to prevent garbage values
          vkCmdFillBuffer(encoder.cmd, totals_buf, 0, VK_WHOLE_SIZE, 0);

          VkMemoryBarrier zero_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          zero_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          zero_barrier.dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          vkCmdPipelineBarrier(
              encoder.cmd,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0,
              1,
              &zero_barrier,
              0,
              nullptr,
              0,
              nullptr);

          // Pass 1: Block-wise inclusive scan, saving chunk totals
          encoder.op_count++;
          ScanPushConst pc1{
              current_scan_size * current_n_scans,
              chunk_size,
              num_chunks * current_n_scans,
              op,
              inclusive ? 1u : 0u, // Pass 1 determines output inclusiveness
              reverse ? 1u : 0u,
              1u, // save_totals
              0u, // has_prefix
              1u // Pass 1 always lays blocks out flat in the auxiliary totals
                 // buffer. The stride applies during the recursive internal
                 // scan mapping
          };

          VkPipelineLayout layout;
          VkDescriptorSetLayout ds_layout;
          VkPipeline pipeline = dev.get_pipeline(
              "scan",
              layout,
              ds_layout,
              4,
              sizeof(pc1),
              vulkan::get_default_specialization_info(dev));

          VkDescriptorSet ds1 = dev.alloc_descriptor_set(stream(), ds_layout);
          VkDescriptorBufferInfo infos1[4]{
              {in_buf, 0, VK_WHOLE_SIZE},
              {out_buf, 0, VK_WHOLE_SIZE},
              {totals_buf, 0, VK_WHOLE_SIZE},
              {in_buf, 0, VK_WHOLE_SIZE}}; // dummy prefix
          VkWriteDescriptorSet writes1[4]{};
          for (int i = 0; i < 4; i++) {
            writes1[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes1[i].dstSet = ds1;
            writes1[i].dstBinding = i;
            writes1[i].descriptorCount = 1;
            writes1[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes1[i].pBufferInfo = &infos1[i];
          }
          vkUpdateDescriptorSets(dev.vk_device(), 4, writes1, 0, nullptr);

          vkCmdPushConstants(
              encoder.cmd,
              layout,
              VK_SHADER_STAGE_COMPUTE_BIT,
              0,
              sizeof(pc1),
              &pc1);
          vkCmdBindPipeline(
              encoder.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
          vkCmdBindDescriptorSets(
              encoder.cmd,
              VK_PIPELINE_BIND_POINT_COMPUTE,
              layout,
              0,
              1,
              &ds1,
              0,
              nullptr);
          vkCmdDispatch(encoder.cmd, num_chunks * current_n_scans, 1, 1);

          VkMemoryBarrier barrier1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          barrier1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          barrier1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(
              encoder.cmd,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0,
              1,
              &barrier1,
              0,
              nullptr,
              0,
              nullptr);

          // Pass 2: Exclusive scan of the totals block recursively
          // The block prefixes themselves are always exclusive of their own
          // chunks, so that they can be added across the block elements
          // smoothly.
          scan_gpu_recursive(
              totals_buf,
              totals_buf,
              num_chunks,
              current_n_scans,
              num_chunks, // Pass 2 scans across the num_chunks elements
                          // allocated per segment scan, ignoring parallel inner
                          // loops
              false, // Must be exclusive for the block prefix dependencies
              reverse);

          // Pass 3: Expand the prefixes back onto the blocks
          encoder.op_count++;
          ScanPushConst pc3{
              current_scan_size * current_n_scans,
              chunk_size,
              num_chunks * current_n_scans,
              op,
              inclusive ? 1u : 0u,
              reverse ? 1u : 0u,
              0u, // save_totals
              1u, // has_prefix
              1u // Prefix block application reads from flat layout
          };

          VkDescriptorSet ds3 = dev.alloc_descriptor_set(stream(), ds_layout);
          VkDescriptorBufferInfo infos3[4]{
              {in_buf, 0, VK_WHOLE_SIZE}, // Read from ORIGINAL input
              {out_buf, 0, VK_WHOLE_SIZE}, // Write to out_buf
              {totals_buf, 0, VK_WHOLE_SIZE}, // dummy
              {totals_buf, 0, VK_WHOLE_SIZE} // Reading from totals prefix
          };
          VkWriteDescriptorSet writes3[4]{};
          for (int i = 0; i < 4; i++) {
            writes3[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes3[i].dstSet = ds3;
            writes3[i].dstBinding = i;
            writes3[i].descriptorCount = 1;
            writes3[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes3[i].pBufferInfo = &infos3[i];
          }
          vkUpdateDescriptorSets(dev.vk_device(), 4, writes3, 0, nullptr);

          vkCmdPushConstants(
              encoder.cmd,
              layout,
              VK_SHADER_STAGE_COMPUTE_BIT,
              0,
              sizeof(pc3),
              &pc3);
          vkCmdBindPipeline(
              encoder.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
          vkCmdBindDescriptorSets(
              encoder.cmd,
              VK_PIPELINE_BIND_POINT_COMPUTE,
              layout,
              0,
              1,
              &ds3,
              0,
              nullptr);
          vkCmdDispatch(encoder.cmd, num_chunks * current_n_scans, 1, 1);

          VkMemoryBarrier barrier3{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          barrier3.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          barrier3.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(
              encoder.cmd,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0,
              1,
              &barrier3,
              0,
              nullptr,
              0,
              nullptr);

          // Free temporary buffer after GPU finishes
          auto s = stream();
          dev.add_completed_handler(
              s, [totals_alloc]() mutable { allocator::free(totals_alloc); });
        }
      };

  uint32_t n_scans = static_cast<uint32_t>(in.size() / scan_size);

  scan_gpu_recursive(
      vulkan::get_buffer(in),
      vulkan::get_buffer(out),
      scan_size,
      n_scans,
      1,
      inclusive_,
      reverse_);
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
      vulkan::device(s.device).add_temporary(s, zero_arr);
    } else {
      array beta_arr(beta_, out.dtype());
      dispatch_binary(c, beta_arr, out, 2 /* Mul */, s);
      vulkan::device(s.device).add_temporary(s, beta_arr);
    }
    return;
  }

  array temp_ab(out.shape(), out.dtype(), nullptr, {});
  Matmul(s).eval_gpu({a, b}, temp_ab);
  vulkan::device(s.device).add_temporary(s, temp_ab);

  array scaled_ab = temp_ab;
  if (alpha_ != 1.0f) {
    array alpha_arr(alpha_, out.dtype());
    array temp(out.shape(), out.dtype(), nullptr, {});
    dispatch_binary(temp_ab, alpha_arr, temp, 2 /* Mul */, s);
    vulkan::device(s.device).add_temporary(s, alpha_arr);
    vulkan::device(s.device).add_temporary(s, temp);
    scaled_ab = temp;
  }

  array scaled_c = c;
  if (beta_ != 1.0f && beta_ != 0.0f) {
    array beta_arr(beta_, out.dtype());
    array temp(out.shape(), out.dtype(), nullptr, {});
    dispatch_binary(c, beta_arr, temp, 2 /* Mul */, s);
    vulkan::device(s.device).add_temporary(s, beta_arr);
    vulkan::device(s.device).add_temporary(s, temp);
    scaled_c = temp;
  }

  if (beta_ == 0.0f) {
    array zero_arr(0.0f, out.dtype());
    dispatch_binary(scaled_ab, zero_arr, out, 0 /* Add */, s);
    vulkan::device(s.device).add_temporary(s, zero_arr);
  } else {
    dispatch_binary(scaled_ab, scaled_c, out, 0 /* Add */, s);
  }

  vulkan::device(s.device).add_temporary(s, temp_ab);
  if (alpha_ != 1.0f)
    vulkan::device(s.device).add_temporary(s, scaled_ab);
  if (beta_ != 1.0f && beta_ != 0.0f)
    vulkan::device(s.device).add_temporary(s, scaled_c);
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
  // For anything more complex (1D, 3D, batched, dilated, groups), we fallback
  // to CPU.
  bool can_use_gpu = true;
  if (in.ndim() != 4 || wt.ndim() != 4)
    can_use_gpu = false;
  if (groups_ != 1)
    can_use_gpu = false;
  if (in.shape(0) != 1)
    can_use_gpu = false; // batch > 1
  for (auto d : kernel_dilation_)
    if (d != 1)
      can_use_gpu = false;
  for (auto d : input_dilation_)
    if (d != 1)
      can_use_gpu = false;

  if (!can_use_gpu) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "conv",
      layout,
      ds_layout,
      3,
      52,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkBuffer in_buf = vulkan::get_buffer(in);
  VkBuffer wt_buf = vulkan::get_buffer(wt);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkDescriptorBufferInfo infos[3] = {
      {in_buf, 0, VK_WHOLE_SIZE},
      {wt_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}};

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
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
  VkPipeline pipeline = dev.get_pipeline(
      "rbits",
      layout,
      ds_layout,
      2,
      12,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(stream().device).synchronize(stream());
    eval_cpu(inputs, out);
    return;
  }

  uint32_t grid_y = half_size + odd;
  uint32_t grid_x = vulkan::div_ceil(num_keys, 256u);
  if (grid_y == 0)
    grid_y = 1; // avoid zero dispatch

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
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
  } pc{
      odd,
      static_cast<uint32_t>(bytes_per_key),
      static_cast<uint32_t>(keys.ndim()),
      static_cast<uint32_t>(num_keys)};

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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load (memory-mapped files)
// ─────────────────────────────────────────────────────────────────────────────

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Load reads from an io::Reader (mmap or file). On unified-memory devices
  // (MoltenVK / Apple Silicon) the VMA buffer is CPU-accessible so eval_cpu
  // can write directly into the already-allocated GPU buffer.
  eval_cpu(inputs, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Imag / Real (complex)
// ─────────────────────────────────────────────────────────────────────────────

void Imag::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

void Real::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

// ─────────────────────────────────────────────────────────────────────────────
// MaskedScatter
// ─────────────────────────────────────────────────────────────────────────────

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
  return;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compiled / JIT
// ─────────────────────────────────────────────────────────────────────────────

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // mx.compile()-fused kernels have no Vulkan GPU implementation yet.
  // Delegate to CPU — works on unified-memory (MoltenVK) devices.
  eval_cpu(inputs, outputs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Matmul variants.

//
// BlockMaskedMM and GatherMM have native Vulkan implementations below.
// GatherQMM and SegmentedMM still fall back to CPU.
// ─────────────────────────────────────────────────────────────────────────────

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() < 2) {
    throw std::invalid_argument("[BlockMaskedMM::eval_gpu] expected >= 2 inputs");
  }

  auto& s = stream();
  auto& dev = vulkan::device(s.device);
  const array& a_orig = inputs[0];
  const array& b_orig = inputs[1];

  bool has_out_mask = (inputs.size() == 3 || inputs.size() == 5);
  bool has_op_mask = (inputs.size() > 3);

  auto float_supported = [](Dtype dt) {
    return dt == float32 || dt == float16 || dt == bfloat16;
  };
  auto mask_type_id = [](Dtype dt) -> uint32_t {
    switch (dt) {
      case float32:
        return 2u;
      case float16:
        return 3u;
      case bfloat16:
        return 4u;
      default:
        return UINT32_MAX;
    }
  };

  if (!float_supported(a_orig.dtype()) || !float_supported(b_orig.dtype())) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  uint32_t M = static_cast<uint32_t>(a_orig.shape(-2));
  uint32_t K = static_cast<uint32_t>(a_orig.shape(-1));
  uint32_t N = static_cast<uint32_t>(b_orig.shape(-1));
  uint32_t batch = (M == 0 || N == 0) ? 0u
                                       : static_cast<uint32_t>(out.size() / (M * N));
  uint32_t block = static_cast<uint32_t>(block_size_);

  if (block == 0) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0 || a_orig.size() == 0 || b_orig.size() == 0) {
    return;
  }

  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;
  if (K == 0 && out.nbytes() > 0) {
    vkCmdFillBuffer(encoder.cmd, vulkan::get_buffer(out), 0, VK_WHOLE_SIZE, 0);
    return;
  }

  auto ensure_contiguous =
      [&](const array& src, std::optional<Dtype> dst_dtype = std::nullopt)
      -> std::pair<array, bool> {
    auto target_dtype = dst_dtype.value_or(src.dtype());
    bool copy_needed =
        src.offset() != 0 || !src.flags().row_contiguous || src.dtype() != target_dtype;
    if (!copy_needed) {
      return {src, false};
    }
    array tmp(src.shape(), target_dtype, nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    copy_gpu(src, tmp, CopyType::General, s);
    return {tmp, true};
  };

  auto [a_contig, a_copied] = ensure_contiguous(a_orig);
  auto [b_contig, b_copied] = ensure_contiguous(b_orig);
  const array& a = a_contig;
  const array& b = b_contig;

  uint32_t tm = (M + block - 1) / block;
  uint32_t tn = (N + block - 1) / block;
  uint32_t tk = (K + block - 1) / block;

  std::optional<array> lhs_mask_contig;
  std::optional<array> rhs_mask_contig;
  std::optional<array> out_mask_contig;
  bool lhs_mask_copied = false;
  bool rhs_mask_copied = false;
  bool out_mask_copied = false;

  if (has_out_mask) {
    Dtype out_mask_dtype =
        (inputs[2].dtype() == bool_) ? float32 : inputs[2].dtype();
    auto [m, copied] = ensure_contiguous(inputs[2], out_mask_dtype);
    if (m.ndim() < 2 || static_cast<uint32_t>(m.shape(-2)) != tm ||
        static_cast<uint32_t>(m.shape(-1)) != tn) {
      dev.synchronize(s);
      eval_cpu(inputs, out);
      return;
    }
    out_mask_copied = copied;
    out_mask_contig = std::move(m);
  }

  if (has_op_mask) {
    Dtype lhs_mask_dtype = (inputs[inputs.size() - 2].dtype() == bool_)
        ? float32
        : inputs[inputs.size() - 2].dtype();
    Dtype rhs_mask_dtype = (inputs[inputs.size() - 1].dtype() == bool_)
        ? float32
        : inputs[inputs.size() - 1].dtype();
    auto [lhs, lhs_cop] =
        ensure_contiguous(inputs[inputs.size() - 2], lhs_mask_dtype);
    auto [rhs, rhs_cop] =
        ensure_contiguous(inputs[inputs.size() - 1], rhs_mask_dtype);
    if (lhs.ndim() < 2 || rhs.ndim() < 2 ||
        static_cast<uint32_t>(lhs.shape(-2)) != tm ||
        static_cast<uint32_t>(lhs.shape(-1)) != tk ||
        static_cast<uint32_t>(rhs.shape(-2)) != tk ||
        static_cast<uint32_t>(rhs.shape(-1)) != tn) {
      dev.synchronize(s);
      eval_cpu(inputs, out);
      return;
    }
    lhs_mask_copied = lhs_cop;
    rhs_mask_copied = rhs_cop;
    lhs_mask_contig = std::move(lhs);
    rhs_mask_contig = std::move(rhs);
  }

  uint32_t lhs_mask_type =
      lhs_mask_contig.has_value() ? mask_type_id(lhs_mask_contig->dtype()) : 0u;
  uint32_t rhs_mask_type =
      rhs_mask_contig.has_value() ? mask_type_id(rhs_mask_contig->dtype()) : 0u;
  uint32_t out_mask_type =
      out_mask_contig.has_value() ? mask_type_id(out_mask_contig->dtype()) : 0u;
  if (lhs_mask_type == UINT32_MAX || rhs_mask_type == UINT32_MAX ||
      out_mask_type == UINT32_MAX) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  bool use_temp_out = out.dtype() != float32;
  std::optional<array> temp_out;
  if (use_temp_out) {
    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer c_buf = vulkan::get_buffer(use_temp_out ? *temp_out : out);
  VkBuffer lhs_mask_buf =
      lhs_mask_contig.has_value() ? vulkan::get_buffer(*lhs_mask_contig) : a_buf;
  VkBuffer rhs_mask_buf =
      rhs_mask_contig.has_value() ? vulkan::get_buffer(*rhs_mask_contig) : b_buf;
  VkBuffer out_mask_buf =
      out_mask_contig.has_value() ? vulkan::get_buffer(*out_mask_contig) : c_buf;

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  struct PushConst {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch;
    uint32_t block_size;
    uint32_t tm;
    uint32_t tn;
    uint32_t tk;
    uint32_t a_elem_bytes;
    uint32_t b_elem_bytes;
    uint32_t lhs_mask_type;
    uint32_t rhs_mask_type;
    uint32_t out_mask_type;
  } pc{
      M,
      N,
      K,
      batch,
      block,
      tm,
      tn,
      tk,
      a.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(a.itemsize()),
      b.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(b.itemsize()),
      lhs_mask_type,
      rhs_mask_type,
      out_mask_type};

  VkPipeline pipeline = dev.get_pipeline(
      "block_masked_mm",
      layout,
      ds_layout,
      6,
      sizeof(pc),
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(s, ds_layout);
  VkDescriptorBufferInfo infos[6]{
      {a_buf, 0, VK_WHOLE_SIZE},
      {b_buf, 0, VK_WHOLE_SIZE},
      {c_buf, 0, VK_WHOLE_SIZE},
      {lhs_mask_buf, 0, VK_WHOLE_SIZE},
      {rhs_mask_buf, 0, VK_WHOLE_SIZE},
      {out_mask_buf, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[6]{};
  for (int i = 0; i < 6; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev.vk_device(), 6, writes, 0, nullptr);

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  uint32_t subgroup_size = dev.subgroup_size();
  uint32_t BN = (subgroup_size == 64) ? 32 : 16;
  uint32_t BM = 256 / BN;
  vkCmdDispatch(cmd, vulkan::div_ceil(N, BN), vulkan::div_ceil(M, BM), batch);

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

  if (use_temp_out) {
    copy_gpu(*temp_out, out, CopyType::General, s);
    dev.add_temporary(s, *temp_out);
  }

  if (a_copied) {
    dev.add_temporary(s, a);
  }
  if (b_copied) {
    dev.add_temporary(s, b);
  }
  if (lhs_mask_copied && lhs_mask_contig.has_value()) {
    dev.add_temporary(s, *lhs_mask_contig);
  }
  if (rhs_mask_copied && rhs_mask_contig.has_value()) {
    dev.add_temporary(s, *rhs_mask_contig);
  }
  if (out_mask_copied && out_mask_contig.has_value()) {
    dev.add_temporary(s, *out_mask_contig);
  }
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() != 4) {
    throw std::invalid_argument("[GatherMM::eval_gpu] expected 4 inputs");
  }

  auto& s = stream();
  auto& dev = vulkan::device(s.device);

  const array& a_orig = inputs[0];
  const array& b_orig = inputs[1];
  const array& lhs_indices_orig = inputs[2];
  const array& rhs_indices_orig = inputs[3];

  auto float_supported = [](Dtype dt) {
    return dt == float32 || dt == float16 || dt == bfloat16;
  };
  if (!float_supported(a_orig.dtype()) || !float_supported(b_orig.dtype()) ||
      lhs_indices_orig.dtype() != uint32 || rhs_indices_orig.dtype() != uint32) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  uint32_t M = (a_orig.ndim() >= 2) ? static_cast<uint32_t>(a_orig.shape(-2)) : 1u;
  uint32_t K = static_cast<uint32_t>(a_orig.shape(-1));
  uint32_t N = (b_orig.ndim() >= 2) ? static_cast<uint32_t>(b_orig.shape(-1)) : 1u;

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0 || a_orig.size() == 0 || b_orig.size() == 0) {
    return;
  }

  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;
  if (K == 0 && out.nbytes() > 0) {
    vkCmdFillBuffer(encoder.cmd, vulkan::get_buffer(out), 0, VK_WHOLE_SIZE, 0);
    return;
  }

  auto ensure_contiguous = [&](const array& src) -> std::pair<array, bool> {
    if (src.offset() == 0 && src.flags().row_contiguous) {
      return {src, false};
    }
    array tmp(src.shape(), src.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    copy_gpu(src, tmp, CopyType::General, s);
    return {tmp, true};
  };

  auto [a_contig, a_copied] = ensure_contiguous(a_orig);
  auto [b_contig, b_copied] = ensure_contiguous(b_orig);
  auto [lhs_idx_contig, lhs_idx_copied] = ensure_contiguous(lhs_indices_orig);
  auto [rhs_idx_contig, rhs_idx_copied] = ensure_contiguous(rhs_indices_orig);
  const array& a = a_contig;
  const array& b = b_contig;
  const array& lhs_indices = lhs_idx_contig;
  const array& rhs_indices = rhs_idx_contig;

  uint32_t a_batch_ndim = (a.ndim() > 2) ? static_cast<uint32_t>(a.ndim() - 2) : 0u;
  uint32_t b_batch_ndim = (b.ndim() > 2) ? static_cast<uint32_t>(b.ndim() - 2) : 0u;
  uint32_t idx_a_ndim =
      lhs_indices.size() > 0 ? static_cast<uint32_t>(lhs_indices.ndim()) : 0u;
  uint32_t idx_b_ndim =
      rhs_indices.size() > 0 ? static_cast<uint32_t>(rhs_indices.ndim()) : 0u;
  uint32_t out_batch_ndim = (out.ndim() > 2) ? static_cast<uint32_t>(out.ndim() - 2) : 0u;

  if (a_batch_ndim > 4 || b_batch_ndim > 4 || idx_a_ndim > 4 || idx_b_ndim > 4 ||
      out_batch_ndim > 4) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  VkMemoryBarrier pre_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  pre_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  pre_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
      encoder.cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &pre_barrier,
      0,
      nullptr,
      0,
      nullptr);

  bool use_temp_out = out.dtype() != float32;
  std::optional<array> temp_out;
  if (use_temp_out) {
    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  struct PushConst {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t a_elem_bytes;
    uint32_t b_elem_bytes;

    uint32_t a_batch_ndim;
    uint32_t b_batch_ndim;
    uint32_t idx_a_ndim;
    uint32_t idx_b_ndim;
    uint32_t out_batch_ndim;

    uint32_t a_stride_row;
    uint32_t a_stride_col;
    uint32_t b_stride_row;
    uint32_t b_stride_col;
  } pc{};

  // Shape/stride data goes into a separate SSBO (binding 5) to stay within
  // the 128-byte push constant limit.
  // Layout: shape_a[4], stride_a[4], shape_b[4], stride_b[4],
  //         shape_idx_a[4], stride_idx_a[4], shape_idx_b[4], stride_idx_b[4],
  //         shape_out[4] = 36 uint32 values
  uint32_t ss_data[36] = {};

  VkBuffer a_buf = vulkan::get_buffer(a);
  VkBuffer b_buf = vulkan::get_buffer(b);
  VkBuffer idx_a_buf = vulkan::get_buffer(lhs_indices);
  VkBuffer idx_b_buf = vulkan::get_buffer(rhs_indices);
  VkBuffer c_buf = vulkan::get_buffer(use_temp_out ? *temp_out : out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      dev.has_cooperative_matrix() && (M % 16 == 0) && (N % 16 == 0) &&
              (K % 16 == 0)
          ? "gather_mm_coop"
          : "gather_mm",
      layout,
      ds_layout,
      6, // 6 bindings: A, B, IdxA, IdxB, C, ShapeStrides
      sizeof(PushConst),
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    dev.synchronize(s);
    eval_cpu(inputs, out);
    return;
  }

  // Populate push constants (scalars only, fits in 56 bytes)
  pc.M = M;
  pc.N = N;
  pc.K = K;

  pc.a_elem_bytes =
      a.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(a.itemsize());
  pc.b_elem_bytes =
      b.dtype() == bfloat16 ? 3u : static_cast<uint32_t>(b.itemsize());

  auto get_batch_dims = [](const auto& v) {
    return decltype(v){v.begin(), v.end() - 2};
  };

  auto batch_shape_A = get_batch_dims(a.shape());
  auto batch_strides_A = get_batch_dims(a.strides());
  auto batch_shape_B = get_batch_dims(b.shape());
  auto batch_strides_B = get_batch_dims(b.strides());

  pc.a_batch_ndim = a_batch_ndim;
  pc.b_batch_ndim = b_batch_ndim;
  pc.idx_a_ndim = idx_a_ndim;
  pc.idx_b_ndim = idx_b_ndim;
  pc.out_batch_ndim = out_batch_ndim;

  pc.a_stride_row = a.ndim() > 1 ? a.strides()[a.ndim() - 2] : 0;
  pc.a_stride_col = a.ndim() > 0 ? a.strides()[a.ndim() - 1] : 1;
  pc.b_stride_row = b.ndim() > 1 ? b.strides()[b.ndim() - 2] : 0;
  pc.b_stride_col = b.ndim() > 0 ? b.strides()[b.ndim() - 1] : 1;

  // Populate shape/stride SSBO data
  auto fill4 = [](uint32_t* dst, uint32_t limit, const auto& arr) {
    for (uint32_t i = 0; i < 4; i++)
      dst[i] = (i < limit) ? static_cast<uint32_t>(arr[i]) : 1u;
  };

  fill4(&ss_data[0], pc.a_batch_ndim, batch_shape_A);
  fill4(&ss_data[4], pc.a_batch_ndim, batch_strides_A);
  fill4(&ss_data[8], pc.b_batch_ndim, batch_shape_B);
  fill4(&ss_data[12], pc.b_batch_ndim, batch_strides_B);
  fill4(&ss_data[16], pc.idx_a_ndim, lhs_indices.shape());
  fill4(&ss_data[20], pc.idx_a_ndim, lhs_indices.strides());
  fill4(&ss_data[24], pc.idx_b_ndim, rhs_indices.shape());
  fill4(&ss_data[28], pc.idx_b_ndim, rhs_indices.strides());
  auto batch_shape_out = get_batch_dims(out.shape());
  fill4(&ss_data[32], pc.out_batch_ndim, batch_shape_out);

  // Allocate and fill SSBO for shape/stride data
  size_t ss_bytes = sizeof(ss_data);
  auto ss_alloc = allocator::malloc(ss_bytes);
  VkBuffer ss_buf =
      static_cast<vulkan::VulkanBuffer*>(ss_alloc.ptr())->buffer;

  // Copy shape/stride data to the GPU buffer (HOST_COHERENT, already mapped)
  {
    auto* vk_buf = static_cast<vulkan::VulkanBuffer*>(ss_alloc.ptr());
    void* mapped = vk_buf->mapped_ptr;
    if (!mapped) {
      // Fallback: map if not already mapped
      vmaMapMemory(dev.vma_allocator(), vk_buf->allocation, &mapped);
    }
    memcpy(mapped, ss_data, ss_bytes);
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(s, ds_layout);
  VkDescriptorBufferInfo a_info{a_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo b_info{b_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo idx_a_info{idx_a_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo idx_b_info{idx_b_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo c_info{c_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo ss_info{ss_buf, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[6]{};
  for (int i = 0; i < 6; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &a_info;
  writes[1].pBufferInfo = &b_info;
  writes[2].pBufferInfo = &idx_a_info;
  writes[3].pBufferInfo = &idx_b_info;
  writes[4].pBufferInfo = &c_info;
  writes[5].pBufferInfo = &ss_info;
  vkUpdateDescriptorSets(dev.vk_device(), 6, writes, 0, nullptr);

  uint32_t batch_size_out = out.size() / (M * N);

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

  constexpr uint32_t BN = 16;
  constexpr uint32_t BM = 16;

  // Grid
  vkCmdDispatch(
      cmd, vulkan::div_ceil(N, BN), vulkan::div_ceil(M, BM), batch_size_out);

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

  // Free shape/stride buffer after GPU finishes
  dev.add_completed_handler(
      s, [ss_alloc]() mutable { allocator::free(ss_alloc); });

  if (use_temp_out) {
    copy_gpu(*temp_out, out, CopyType::General, s);
    dev.add_temporary(s, *temp_out);
  }
  if (a_copied) {
    dev.add_temporary(s, a);
  }
  if (b_copied) {
    dev.add_temporary(s, b);
  }
  if (lhs_idx_copied) {
    dev.add_temporary(s, lhs_indices);
  }
  if (rhs_idx_copied) {
    dev.add_temporary(s, rhs_indices);
  }
}


void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Gather + dequantize + matmul — no GPU implementation yet.
  // Requires indexing.comp (gather) + quantized.comp (dequant) + matmul.comp,
  // all pipelined.  CPU fallback is correct on unified memory (MoltenVK/iGPU).
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Segmented (variable-length batch) matmul.
  // CUDA backend also has no GPU implementation for this op.
  // eval.cpp catches this and calls eval_cpu() via the unified-memory path.
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, out);
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

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() >= 4);
  auto& s = stream();
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
  const array& x = inputs[0];
  const array& w_pack = inputs[1];
  const array& scales = inputs[2];
  const array& biases = inputs[3];

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
  VkPipeline pipeline = dev.get_pipeline(
      "quantized",
      layout,
      ds_layout,
      4,
      24,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(s.device).add_temporary(s, dq_weights);
    throw std::runtime_error(
        "[vulkan::QuantizedMatmul] failed to get quantized dequant pipeline");
  }

  VkBuffer src_buf = vulkan::get_buffer(w_pack);
  VkBuffer scale_buf = vulkan::get_buffer(scales);
  VkBuffer bias_buf = vulkan::get_buffer(biases);
  VkBuffer dq_buf = vulkan::get_buffer(dq_weights);

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo bufs[4] = {
      {src_buf, 0, VK_WHOLE_SIZE},
      {scale_buf, 0, VK_WHOLE_SIZE},
      {bias_buf, 0, VK_WHOLE_SIZE},
      {dq_buf, 0, VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 4; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufs[i];
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
  } pc{
      n_dequant,
      dq_op,
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
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  // ── Pass 2: float matmul x[M,K] @ dq[K,N] -> out[M,N] ───────────────────
  Matmul(s).eval_gpu({x, dq_weights}, out);
  vulkan::device(s.device).add_temporary(s, dq_weights);
}

// ─────────────────────────────────────────────────────────────────────────────
// QQMatmul — quantized x quantized (both operands quantized)
//
// Input layout (from primitives.h / eval_cpu):
//   inputs[0]: packed uint  w1  (LHS quantized)  shape: depends on
//   bits_/group_size_ inputs[1]: float scales1 inputs[2]: float biases1  (may
//   be absent for some modes — but we read 6 when present) inputs[3]: packed
//   uint  w2  (RHS quantized) inputs[4]: float scales2 inputs[5]: float biases2
//   (may be absent)
//
// Note: eval_cpu only handles the special-case where inputs[1].dtype()==uint32
// and inputs[0].shape(-2)==1, at which point inputs[0] is actually the float
// activation (x), inputs[1]/[2] are the quantized weight+scales for the RHS.
// For the general case (both truly quantized), we use a three-step GPU path:
//   Pass A: dequantize w1 (LHS) -> float dq1
//   Pass B: dequantize w2 (RHS) -> float dq2
//   Pass C: Matmul dq1 @ dq2 -> out
//
// Shape convention: after dequantization both dq1 and dq2 are row-major float.
// We need: dq1[M, K] @ dq2[K, N] -> out[M, N].
// Neither is transposed (same as QuantizedMatmul with transpose_=false).
// ─────────────────────────────────────────────────────────────────────────────

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& dev = vulkan::device(s.device);
  auto& encoder = vulkan::get_command_encoder(s);
  encoder.op_count++;

  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  // When inputs[1].dtype() == uint32, the layout is the "vector x
  // quantized-weight" path:
  //   inputs[0] = float x  [M, K]
  //   inputs[1] = packed uint w  [K_packed, N]
  //   inputs[2] = float scales
  //   inputs[3] = float biases  (optional, may be empty)
  // In this case, only one dequantize pass is needed — identical to
  // QuantizedMatmul. For the fully-quantized (both inputs packed uint) general
  // case, do two dequant passes.

  bool lhs_is_float = (inputs[0].dtype() != uint32);

  if (lhs_is_float) {
    // Activation (float) x quantized-weight path.
    // Delegate entirely to QuantizedMatmul logic (no transpose).
    const array& x = inputs[0];
    const array& w_pack = inputs[1];
    const array& scales = inputs[2];
    const array& biases = inputs[3];

    int K = static_cast<int>(x.shape(-1));
    int N = static_cast<int>(out.shape(-1));
    uint32_t n_dequant = static_cast<uint32_t>(K) * static_cast<uint32_t>(N);

    array dq_weights(Shape{K, N}, float32, nullptr, {});
    dq_weights.set_data(allocator::malloc(dq_weights.nbytes()));

    VkPipelineLayout layout;
    VkDescriptorSetLayout ds_layout;
    VkPipeline pipeline = dev.get_pipeline(
        "quantized",
        layout,
        ds_layout,
        4,
        24,
        vulkan::get_default_specialization_info(dev));
    if (pipeline == VK_NULL_HANDLE) {
      vulkan::device(s.device).add_temporary(s, dq_weights);
      throw std::runtime_error(
          "[vulkan::QQMatmul] failed to get quantized dequant pipeline");
    }

    VkBuffer src_buf = vulkan::get_buffer(w_pack);
    VkBuffer scale_buf = vulkan::get_buffer(scales);
    VkBuffer bias_buf = vulkan::get_buffer(biases);
    VkBuffer dq_buf = vulkan::get_buffer(dq_weights);

    VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
    VkDescriptorBufferInfo bufs[4] = {
        {src_buf, 0, VK_WHOLE_SIZE},
        {scale_buf, 0, VK_WHOLE_SIZE},
        {bias_buf, 0, VK_WHOLE_SIZE},
        {dq_buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = ds;
      writes[i].dstBinding = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &bufs[i];
    }
    vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

    struct QuantPushConst {
      uint32_t n;
      uint32_t op;
      uint32_t group_size;
      uint32_t bits;
      uint32_t N_dim;
      uint32_t K_dim;
    } pc{
        n_dequant,
        1u,
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

    Matmul(s).eval_gpu({x, dq_weights}, out);
    vulkan::device(s.device).add_temporary(s, dq_weights);
    return;
  }

  // General case: both inputs[0] and inputs[3] are packed-uint quantized
  // matrices. inputs[0]=w1_pack, inputs[1]=scales1, inputs[2]=biases1
  // inputs[3]=w2_pack, inputs[4]=scales2, inputs[5]=biases2
  // Output shape: out = [M, N]; we need dq1[M,K] @ dq2[K,N].
  const array& w1_pack = inputs[0];
  const array& scales1 = inputs[1];
  const array& biases1 = inputs[2];
  const array& w2_pack = inputs[3];
  const array& scales2 = inputs[4];
  const array& biases2 = inputs[5];

  int M = static_cast<int>(out.shape(-2));
  int N = static_cast<int>(out.shape(-1));
  // K is determined from w1: w1_pack shape is [M, K_packed], K = K_packed *
  // (32/bits_) or from scales1 shape: scales1 has K groups along last dim *
  // group_size_
  int K = static_cast<int>(scales1.shape(-1)) * group_size_;

  uint32_t n_dq1 = static_cast<uint32_t>(M) * static_cast<uint32_t>(K);
  uint32_t n_dq2 = static_cast<uint32_t>(K) * static_cast<uint32_t>(N);

  // Allocate temporaries for dequantized matrices
  array dq1(Shape{M, K}, float32, nullptr, {});
  dq1.set_data(allocator::malloc(dq1.nbytes()));
  array dq2(Shape{K, N}, float32, nullptr, {});
  dq2.set_data(allocator::malloc(dq2.nbytes()));

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "quantized",
      layout,
      ds_layout,
      4,
      24,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    vulkan::device(s.device).add_temporary(s, dq1);
    vulkan::device(s.device).add_temporary(s, dq2);
    throw std::runtime_error(
        "[vulkan::QQMatmul] failed to get quantized dequant pipeline");
  }

  struct QuantPushConst {
    uint32_t n;
    uint32_t op;
    uint32_t group_size;
    uint32_t bits;
    uint32_t N_dim;
    uint32_t K_dim;
  };

  VkCommandBuffer cmd = encoder.cmd;

  // ── Pass A: Dequantize w1 -> dq1 [M, K] (op=1, non-transposed) ──────────
  {
    VkBuffer src_buf = vulkan::get_buffer(w1_pack);
    VkBuffer scale_buf = vulkan::get_buffer(scales1);
    VkBuffer bias_buf = vulkan::get_buffer(biases1);
    VkBuffer dq_buf = vulkan::get_buffer(dq1);

    VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
    VkDescriptorBufferInfo bufs[4] = {
        {src_buf, 0, VK_WHOLE_SIZE},
        {scale_buf, 0, VK_WHOLE_SIZE},
        {bias_buf, 0, VK_WHOLE_SIZE},
        {dq_buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = ds;
      writes[i].dstBinding = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &bufs[i];
    }
    vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

    QuantPushConst pc{
        n_dq1,
        1u,
        static_cast<uint32_t>(group_size_),
        static_cast<uint32_t>(bits_),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(M)};
    vkCmdPushConstants(
        cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, vulkan::div_ceil(n_dq1, 256u), 1, 1);
  }

  // Barrier between pass A and pass B
  {
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
  }

  // ── Pass B: Dequantize w2 -> dq2 [K, N] (op=1, non-transposed) ──────────
  {
    VkBuffer src_buf = vulkan::get_buffer(w2_pack);
    VkBuffer scale_buf = vulkan::get_buffer(scales2);
    VkBuffer bias_buf = vulkan::get_buffer(biases2);
    VkBuffer dq_buf = vulkan::get_buffer(dq2);

    VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
    VkDescriptorBufferInfo bufs[4] = {
        {src_buf, 0, VK_WHOLE_SIZE},
        {scale_buf, 0, VK_WHOLE_SIZE},
        {bias_buf, 0, VK_WHOLE_SIZE},
        {dq_buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = ds;
      writes[i].dstBinding = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &bufs[i];
    }
    vkUpdateDescriptorSets(dev.vk_device(), 4, writes, 0, nullptr);

    QuantPushConst pc{
        n_dq2,
        1u,
        static_cast<uint32_t>(group_size_),
        static_cast<uint32_t>(bits_),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K)};
    vkCmdPushConstants(
        cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, vulkan::div_ceil(n_dq2, 256u), 1, 1);
  }

  // Barrier between pass B and matmul
  {
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
  }

  // ── Pass C: float matmul dq1[M,K] @ dq2[K,N] -> out[M,N] ────────────────
  Matmul(s).eval_gpu({dq1, dq2}, out);
  vulkan::device(s.device).add_temporary(s, dq1);
  vulkan::device(s.device).add_temporary(s, dq2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hadamard — GPU dispatch via hadamard.comp (Walsh-Hadamard Transform)
//
// The GPU shader implements the power-of-2 WHT (Sylvester construction).
// For sizes with a non-power-of-2 component (m > 1) or sizes > 2048 we fall
// back to eval_cpu, which handles those via the hadamard_m() matrix path.
// ─────────────────────────────────────────────────────────────────────────────

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  auto& s = stream();

  // decompose_hadamard returns {n, m} where n is the power-of-2 part
  // and m is the Paley/non-power-of-2 component (usually 1).
  int axis = out.ndim() - 1;
  auto [n, m] = decompose_hadamard(out.shape(axis));

  // GPU shader handles pure power-of-2 WHT (m == 1) up to n = 16384.
  // n <= 2048: single shared-memory pass (h_step=0).
  // 2048 < n <= 16384: in-group pass on 2048-element tiles, then cross-group
  //   passes via global-memory butterfly stages (h_step > 0).
  bool can_gpu = (m == 1) && (n > 0) && (n <= 16384) && (in.dtype() == float32);
  if (!can_gpu) {
    // For large n or non-power-of-2 sizes we run the butterfly on CPU.
    // We cannot call eval_cpu() directly (stream mismatch causes wrong values),
    // so instead: copy input to a VMA buffer via GPU, synchronize, then run
    // the butterfly in-place on the CPU-accessible unified-memory buffer.
    //
    // IMPORTANT: do NOT pre-allocate out before calling copy_gpu — copy_gpu
    // internally calls set_copy_output_data which always re-allocates out.
    // A double allocation causes the scheduler to see a stale buffer pointer
    // when multiple streams are evaluated concurrently.
    if (out.size() == 0) {
      out.set_data(allocator::malloc(0));
      return;
    }
    // copy_gpu allocates out AND dispatches the GPU copy in one shot.
    copy_gpu(in, out, CopyType::General, s);
    // Flush command buffer and wait for GPU copy to complete
    vulkan::device(s.device).synchronize(s);

    // --- Butterfly transforms in-place on out.data<T>() ---
    auto compute_hadamard = [&]<typename T>() {
      auto* out_ptr = out.data<T>();
      size_t size = out.size();

      // hadamard_n: in-place power-of-2 Walsh-Hadamard butterfly
      float n_scale = (m > 1) ? 1.0f : scale_;
      int n_over_2 = n / 2;
      for (int b = 0; b < (int)(size / (size_t)n); b++) {
        T* dp = out_ptr + (size_t)b * n;
        int hh = 1;
        while (hh < n) {
          for (int i = 0; i < n / 2; i++) {
            int kk = i & (hh - 1);
            int j = ((i - kk) << 1) + kk;
            float xv = static_cast<float>(dp[j]);
            float yv = static_cast<float>(dp[j + hh]);
            float nv1 = xv + yv;
            float nv2 = xv - yv;
            if (hh == n_over_2) {
              nv1 *= n_scale;
              nv2 *= n_scale;
            }
            dp[j] = static_cast<T>(nv1);
            dp[j + hh] = static_cast<T>(nv2);
          }
          hh <<= 1;
        }
      }

      // hadamard_m: apply non-power-of-2 Hadamard matrix (m ∈ {12,20,28})
      if (m > 1) {
        auto h_matrices = hadamard_matrices();
        auto& mat_sv = h_matrices[m];
        std::vector<bool> hmat;
        hmat.reserve((size_t)m * m);
        auto start = mat_sv.find('\n');
        start = (start == std::string_view::npos) ? 0 : start + 1;
        while (start < mat_sv.size()) {
          auto end = mat_sv.find('\n', start);
          if (end == std::string_view::npos)
            end = mat_sv.size();
          auto row = mat_sv.substr(start, end - start);
          for (char c : row)
            hmat.push_back(c == '+');
          start = end + 1;
        }
        // Each batch of (m * n) elements: apply H_m matrix across the m groups
        for (int b = 0; b < (int)(size / (size_t)m / (size_t)n); b++) {
          T* dp = out_ptr + (size_t)b * (size_t)n * m;
          std::vector<float> tmp(m);
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
              float acc = 0.0f;
              for (int kk = 0; kk < m; kk++) {
                float val = static_cast<float>(dp[i + (size_t)kk * n]);
                acc += hmat[(size_t)kk + (size_t)j * m] ? val : -val;
              }
              tmp[j] = acc;
            }
            for (int j = 0; j < m; j++) {
              dp[i + (size_t)j * n] = static_cast<T>(tmp[j] * scale_);
            }
          }
        }
      }
    };

    if (in.dtype() == float32) {
      compute_hadamard.template operator()<float>();
    } else if (in.dtype() == float16) {
      compute_hadamard.template operator()<float16_t>();
    } else if (in.dtype() == bfloat16) {
      compute_hadamard.template operator()<bfloat16_t>();
    } else {
      throw std::runtime_error("[vulkan::Hadamard] unsupported type");
    }

    // After CPU butterfly, insert a host→device memory barrier so subsequent
    // GPU ops (e.g. subtraction in assertLess(y1-y2)) see our writes.
    // Without this, the GPU could read stale cache lines from out's VMA buffer
    // even on unified-memory hardware (MoltenVK / Apple M1).
    {
      auto& enc = vulkan::get_command_encoder(s);
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(
          enc.cmd,
          VK_PIPELINE_STAGE_HOST_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          1,
          &mb,
          0,
          nullptr,
          0,
          nullptr);
      enc.op_count++;
    }
    return;
  }

  // Allocate output buffer
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  // The shader reads from src (binding 0) and writes to dst (binding 1).
  // If the input is non-contiguous (e.g. from a stride-2 slice), we need
  // to make a contiguous copy so the shader's linear indexing is correct.
  array src_arr(in.shape(), in.dtype(), nullptr, {});
  if (!in.flags().row_contiguous) {
    src_arr.set_data(allocator::malloc(src_arr.nbytes()));
    copy_gpu(in, src_arr, CopyType::General, s);
    vulkan::device(s.device).add_temporary(s, src_arr);
  } else {
    src_arr = in;
  }

  uint32_t batch_size = static_cast<uint32_t>(out.size() / n);

  auto& encoder = vulkan::get_command_encoder(s);
  auto& dev = vulkan::device(s.device);
  encoder.op_count++;

  // hadamard.comp push_constant layout (16 bytes):
  //   uint n, uint batch_size, float scale, uint h_step
  // h_step == 0  → shared-memory in-group butterfly (all stages, n <= 2048)
  // h_step  > 0  → single cross-group butterfly stage with that stride
  struct HadamardPC {
    uint32_t n;
    uint32_t batch_size;
    float scale;
    uint32_t h_step;
  };

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "hadamard",
      layout,
      ds_layout,
      2,
      sizeof(HadamardPC),
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    eval_cpu(inputs, out);
    return;
  }

  VkCommandBuffer cmd = encoder.cmd;

  // Helper: issue one pipeline barrier between compute passes
  auto compute_barrier = [&]() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
  };

  // Helper: bind two buffers (src_buf, dst_buf), push constants, dispatch
  auto dispatch_pass = [&](
      VkBuffer src_buf,
      VkBuffer dst_buf,
      HadamardPC pc,
      uint32_t num_groups_x) {
    VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
    VkDescriptorBufferInfo infos[2]{
        {src_buf, 0, VK_WHOLE_SIZE},
        {dst_buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = ds;
      writes[i].dstBinding = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev.vk_device(), 2, writes, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, num_groups_x, 1, 1);
  };

  if (n <= 2048) {
    // ── Single shared-memory pass: all stages within the workgroup ───────────
    // scale is applied inside the shader at the last butterfly stage.
    HadamardPC pc{static_cast<uint32_t>(n), batch_size, scale_, 0u};
    dispatch_pass(
        vulkan::get_buffer(src_arr),
        vulkan::get_buffer(out),
        pc,
        batch_size);
  } else {
    // ── Multi-pass for n > 2048 ──────────────────────────────────────────────
    // Strategy:
    //   Pass 1 (in-group): treat each 2048-element sub-block independently.
    //     n_tile = 2048, effective batch = batch_size * (n / 2048).
    //     scale = 1.0 (not yet — applied at final stage).
    //     src: src_arr, dst: out.
    //   Passes 2..log2(n)-log2(2048)+1 (cross-group):
    //     Each pass does ONE butterfly stage at stride h_step.
    //     src and dst are both 'out' (in-place via separate SSBO bindings on same buffer).
    //     scale applied only at the last pass (h_step == n/2).

    uint32_t n_tile = 2048u;
    uint32_t tiles_per_batch = static_cast<uint32_t>(n) / n_tile;
    uint32_t in_group_batch = batch_size * tiles_per_batch;

    // Pass 1: in-group butterfly on 2048-element tiles
    {
      HadamardPC pc{n_tile, in_group_batch, 1.0f, 0u};
      dispatch_pass(
          vulkan::get_buffer(src_arr),
          vulkan::get_buffer(out),
          pc,
          in_group_batch);  // one workgroup per tile
    }

    // Cross-group passes for h_step = 2048, 4096, ..., n/2
    VkBuffer out_buf = vulkan::get_buffer(out);
    for (uint32_t h_step = n_tile; h_step < static_cast<uint32_t>(n); h_step <<= 1u) {
      compute_barrier();

      bool is_last = (h_step == static_cast<uint32_t>(n) >> 1u);
      float pass_scale = is_last ? scale_ : 1.0f;

      // Global-memory pass: each thread handles one butterfly pair.
      // total_pairs = batch_size * (n / 2)
      uint32_t half_n = static_cast<uint32_t>(n) >> 1u;
      uint32_t total_pairs = batch_size * half_n;
      uint32_t wg_size = dev.preferred_workgroup_size();
      uint32_t num_groups = (total_pairs + wg_size - 1) / wg_size;

      HadamardPC pc{static_cast<uint32_t>(n), batch_size, pass_scale, h_step};
      // src and dst both point to 'out' buffer (in-place)
      dispatch_pass(out_buf, out_buf, pc, num_groups);
    }
  }

  // Final barrier so downstream ops see all writes
  compute_barrier();
}

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
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // eval_cpu allocates lu / pivots / row_indices internally.
  eval_cpu(inputs, outputs);
}

void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // eval_cpu allocates Q and R internally via set_data.
  eval_cpu(inputs, outputs);
}

void SVD::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // eval_cpu allocates U, S, Vt (or just S) internally.
  eval_cpu(inputs, outputs);
}

void Eig::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // eval_cpu allocates eigenvalues / eigenvectors internally.
  eval_cpu(inputs, outputs);
}

void Eigh::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // eval_cpu allocates eigenvalues and copies input to eigenvectors internally.
  eval_cpu(inputs, outputs);
}

// -- Single-output ops --------------------------------------------------------

void Inverse::eval_gpu(const std::vector<array>& inputs, array& out) {
  // eval_cpu calls copy_cpu which invokes set_copy_output_data -> allocates
  // out.
  eval_cpu(inputs, out);
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  // eval_cpu calls copy_cpu which invokes set_copy_output_data -> allocates
  // out.
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

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

// ─── LayerNorm GPU dispatch ─────────────────────────────────────────────────

bool LayerNorm::use_fallback(Stream s) {
  return false;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  bool use_temp = in.dtype() != float32;

  std::optional<array> temp_in;
  std::optional<array> temp_weight;
  std::optional<array> temp_bias;
  std::optional<array> temp_out;

  if (use_temp) {
    temp_in = array(in.shape(), float32, nullptr, {});
    temp_in->set_data(allocator::malloc(temp_in->nbytes()));
    copy_gpu(in, *temp_in, CopyType::General, stream());

    if (inputs.size() > 1 && inputs[1].size() > 0) {
      temp_weight = array(inputs[1].shape(), float32, nullptr, {});
      temp_weight->set_data(allocator::malloc(temp_weight->nbytes()));
      copy_gpu(inputs[1], *temp_weight, CopyType::General, stream());
    }

    if (inputs.size() > 2 && inputs[2].size() > 0) {
      temp_bias = array(inputs[2].shape(), float32, nullptr, {});
      temp_bias->set_data(allocator::malloc(temp_bias->nbytes()));
      copy_gpu(inputs[2], *temp_bias, CopyType::General, stream());
    }

    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  uint32_t n_cols = static_cast<uint32_t>(in.shape().back());
  uint32_t n_rows = static_cast<uint32_t>(in.size() / n_cols);

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  bool has_weight = inputs.size() > 1 && inputs[1].size() > 0;
  bool has_bias = inputs.size() > 2 && inputs[2].size() > 0;

  VkBuffer in_buf = vulkan::get_buffer(use_temp ? *temp_in : in);
  VkBuffer weight_buf = has_weight
      ? vulkan::get_buffer(use_temp ? *temp_weight : inputs[1])
      : in_buf;
  VkBuffer bias_buf =
      has_bias ? vulkan::get_buffer(use_temp ? *temp_bias : inputs[2]) : in_buf;
  VkBuffer out_buf = vulkan::get_buffer(use_temp ? *temp_out : out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "normalization",
      layout,
      ds_layout,
      4,
      24,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo infos[4]{
      {in_buf, 0, VK_WHOLE_SIZE},
      {weight_buf, 0, VK_WHOLE_SIZE},
      {bias_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}};
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
    uint32_t op; // 0 = NORM_LAYER_NORM
    float eps;
    uint32_t has_weight;
    uint32_t has_bias;
  } pc{n_rows, n_cols, 0u, eps_, has_weight ? 1u : 0u, has_bias ? 1u : 0u};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_rows, 1, 1);

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

  if (use_temp) {
    copy_gpu(*temp_out, out, CopyType::General, stream());
  }
}

NO_GPU_MULTI(LayerNormVJP)

// ─── RMSNorm GPU dispatch ───────────────────────────────────────────────────

bool RMSNorm::use_fallback(Stream s) {
  return false;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  bool use_temp = in.dtype() != float32;

  std::optional<array> temp_in;
  std::optional<array> temp_weight;
  std::optional<array> temp_out;

  if (use_temp) {
    temp_in = array(in.shape(), float32, nullptr, {});
    temp_in->set_data(allocator::malloc(temp_in->nbytes()));
    copy_gpu(in, *temp_in, CopyType::General, stream());

    if (inputs.size() > 1 && inputs[1].size() > 0) {
      temp_weight = array(inputs[1].shape(), float32, nullptr, {});
      temp_weight->set_data(allocator::malloc(temp_weight->nbytes()));
      copy_gpu(inputs[1], *temp_weight, CopyType::General, stream());
    }

    temp_out = array(out.shape(), float32, nullptr, {});
    temp_out->set_data(allocator::malloc(temp_out->nbytes()));
  }

  uint32_t n_cols = static_cast<uint32_t>(in.shape().back());
  uint32_t n_rows = static_cast<uint32_t>(in.size() / n_cols);

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  bool has_weight = inputs.size() > 1 && inputs[1].size() > 0;

  VkBuffer in_buf = vulkan::get_buffer(use_temp ? *temp_in : in);
  VkBuffer weight_buf = has_weight
      ? vulkan::get_buffer(use_temp ? *temp_weight : inputs[1])
      : in_buf;
  VkBuffer out_buf = vulkan::get_buffer(use_temp ? *temp_out : out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "normalization",
      layout,
      ds_layout,
      4,
      24,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo infos[4]{
      {in_buf, 0, VK_WHOLE_SIZE},
      {weight_buf, 0, VK_WHOLE_SIZE},
      {in_buf, 0, VK_WHOLE_SIZE}, // bias_buf placeholder (unused)
      {out_buf, 0, VK_WHOLE_SIZE}};
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
    uint32_t op; // 1 = NORM_RMS_NORM
    float eps;
    uint32_t has_weight;
    uint32_t has_bias;
  } pc{n_rows, n_cols, 1u, eps_, has_weight ? 1u : 0u, 0u};

  VkCommandBuffer cmd = encoder.cmd;
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cmd, n_rows, 1, 1);

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

  if (use_temp) {
    copy_gpu(*temp_out, out, CopyType::General, stream());
  }
}

NO_GPU_MULTI(RMSNormVJP)

// ─── RoPE GPU dispatch ──────────────────────────────────────────────────────

bool RoPE::use_fallback(Stream s) {
  return false;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& out = outputs[0];
  const auto& in = inputs[0];

  // inputs: [x, cos_freqs, sin_freqs]
  if (inputs.size() < 3) {
    fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0)
    return;

  auto& encoder = vulkan::get_command_encoder(stream());
  auto& dev = vulkan::device(stream().device);
  encoder.op_count++;

  VkBuffer in_buf = vulkan::get_buffer(in);
  VkBuffer cos_buf = vulkan::get_buffer(inputs[1]);
  VkBuffer sin_buf = vulkan::get_buffer(inputs[2]);
  VkBuffer out_buf = vulkan::get_buffer(out);

  VkPipelineLayout layout;
  VkDescriptorSetLayout ds_layout;
  VkPipeline pipeline = dev.get_pipeline(
      "rope",
      layout,
      ds_layout,
      4,
      12,
      vulkan::get_default_specialization_info(dev));
  if (pipeline == VK_NULL_HANDLE) {
    fallback_(inputs);
    return;
  }

  VkDescriptorSet ds = dev.alloc_descriptor_set(stream(), ds_layout);
  VkDescriptorBufferInfo infos[4]{
      {in_buf, 0, VK_WHOLE_SIZE},
      {cos_buf, 0, VK_WHOLE_SIZE},
      {sin_buf, 0, VK_WHOLE_SIZE},
      {out_buf, 0, VK_WHOLE_SIZE}};
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
  vkCmdPushConstants(
      cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(
      cmd, vulkan::div_ceil(in.size() / 2, vulkan::WORKGROUP_SIZE), 1, 1);

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
}
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)

// ─────────────────────────────────────────────────────────────────────────────
// fast::ConvertFP8 — CPU fallback (no GPU shader for FP8 conversion yet)
// ConvertFP8 extends Primitive (multi-output), so eval_gpu takes outputs
// vector. eval_cpu handles set_data internally — do not pre-allocate here.
// ─────────────────────────────────────────────────────────────────────────────

void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::device(stream().device).synchronize(stream());
  eval_cpu(inputs, outputs);
}

// ─────────────────────────────────────────────────────────────────────────────
// fast::Quantize — GPU dispatch for dequantize (op=1 via quantized.comp),
// CPU fallback for the quantize direction.
//
// dequantize_=true:  1 output — dequantize packed w -> float using GPU shader
// dequantize_=false: 3 outputs — quantize float -> packed w, scales, biases
//                    (GPU shader for quantize not yet implemented; use CPU)
//
// NOTE: eval_cpu for Quantize always accesses outputs[0..2], so it CANNOT
// be called for the dequantize case (only 1 output).
// ─────────────────────────────────────────────────────────────────────────────

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (!dequantize_) {
    // Quantize path (float -> packed): eval_cpu handles all 3 outputs.
    eval_cpu(inputs, outputs);
    return;
  }

  // Dequantize path (packed -> float): 1 output.
  // inputs[0]=packed_w, inputs[1]=scales, inputs[2]=biases
  // outputs[0]=dequantized float
  //
  // Use inline CPU dequantize: VMA buffers are host-visible on MoltenVK/iGPU
  // (unified memory), so we can read inputs and write outputs directly.
  // This avoids a GPU shader dispatch that causes VK_ERROR_DEVICE_LOST on
  // certain MoltenVK configurations due to buffer type mismatch.
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  size_t n = static_cast<size_t>(out.size());
  if (n == 0)
    return;

  // MLX guarantees inputs are evaluated before eval_gpu is called.
  // On MoltenVK/iGPU unified memory, host can read GPU-written buffers
  // directly.
  const uint32_t* w_packed = inputs[0].data<uint32_t>();
  const float* scales = inputs[1].data<float>();
  const float* biases = inputs[2].data<float>();
  float* dst = out.data<float>();

  uint32_t bits = static_cast<uint32_t>(bits_);
  uint32_t gs = static_cast<uint32_t>(group_size_);
  uint32_t epw = 32u / bits; // elements per packed word
  uint32_t mask = (1u << bits) - 1u;

  for (size_t i = 0; i < n; ++i) {
    uint32_t ui = static_cast<uint32_t>(i);
    uint32_t packed = w_packed[ui / epw];
    uint32_t shift = (ui % epw) * bits;
    uint32_t q = (packed >> shift) & mask;
    uint32_t group = ui / gs;
    dst[i] = static_cast<float>(q) * scales[group] + biases[group];
  }
}

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
