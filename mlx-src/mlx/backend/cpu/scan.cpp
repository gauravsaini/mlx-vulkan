// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename Op>
void contiguous_scan(
    const T* input,
    U* output,
    int count,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  const int64_t block_span = static_cast<int64_t>(stride);
  for (int i = 0; i < count; i++) {
    const int64_t base = static_cast<int64_t>(i) * block_span;
    U acc = init;
    if (!reverse) {
      if (inclusive) {
        acc = static_cast<U>(input[base]);
        output[base] = acc;
        for (int j = 1; j < stride; j++) {
          auto idx = base + j;
          acc = op(acc, input[idx]);
          output[idx] = acc;
        }
      } else {
        for (int j = 0; j < stride; j++) {
          auto idx = base + j;
          output[idx] = acc;
          acc = op(acc, input[idx]);
        }
      }
    } else {
      if (inclusive) {
        auto idx_last = base + stride - 1;
        acc = static_cast<U>(input[idx_last]);
        output[idx_last] = acc;
        for (int j = stride - 2; j >= 0; j--) {
          auto idx = base + j;
          acc = op(acc, input[idx]);
          output[idx] = acc;
        }
      } else {
        for (int j = stride - 1; j >= 0; j--) {
          auto idx = base + j;
          output[idx] = acc;
          acc = op(acc, input[idx]);
        }
      }
    }
  }
};

template <typename T, typename U, typename Op>
void strided_scan(
    const T* input,
    U* output,
    int count,
    int size,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  // Index-based implementation avoids pointer drift and out-of-bounds writes
  // in reverse/exclusive paths.
  const int64_t block_span = static_cast<int64_t>(size) * stride;

  for (int i = 0; i < count; i++) {
    const int64_t block_base = static_cast<int64_t>(i) * block_span;

    for (int k = 0; k < stride; k++) {
      U acc = init;
      if (!reverse) {
        if (inclusive) {
          auto idx0 = block_base + k;
          acc = static_cast<U>(input[idx0]);
          output[idx0] = acc;
          for (int j = 1; j < size; j++) {
            auto idx = block_base + static_cast<int64_t>(j) * stride + k;
            acc = op(acc, input[idx]);
            output[idx] = acc;
          }
        } else {
          for (int j = 0; j < size; j++) {
            auto idx = block_base + static_cast<int64_t>(j) * stride + k;
            output[idx] = acc;
            acc = op(acc, input[idx]);
          }
        }
      } else {
        if (inclusive) {
          auto idx_last =
              block_base + static_cast<int64_t>(size - 1) * stride + k;
          acc = static_cast<U>(input[idx_last]);
          output[idx_last] = acc;
          for (int j = size - 2; j >= 0; j--) {
            auto idx = block_base + static_cast<int64_t>(j) * stride + k;
            acc = op(acc, input[idx]);
            output[idx] = acc;
          }
        } else {
          for (int j = size - 1; j >= 0; j--) {
            auto idx = block_base + static_cast<int64_t>(j) * stride + k;
            output[idx] = acc;
            acc = op(acc, input[idx]);
          }
        }
      }
    }
  }
};

template <typename T, typename U, typename Op>
void scan_op(
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  if (in.size() == 0 || in.shape(axis) == 0) {
    return;
  }

  if (in.flags().row_contiguous) {
    if (in.strides()[axis] == 1) {
      contiguous_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis),
          in.shape(axis),
          reverse,
          inclusive,
          op,
          init);
    } else {
      strided_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis) / in.strides()[axis],
          in.shape(axis),
          in.strides()[axis],
          reverse,
          inclusive,
          op,
          init);
    }
  } else {
    throw std::runtime_error("Scan op supports only contiguous inputs");
  }
}

template <typename T, typename U>
void scan_dispatch(
    Scan::ReduceType rtype,
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive) {
  switch (rtype) {
    case Scan::Sum: {
      auto op = [](U y, T x) { return y + x; };
      auto init = static_cast<U>(0);
      scan_op<T, U>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Prod: {
      auto op = [](U y, T x) { return y * x; };
      auto init = static_cast<U>(1);
      scan_op<T, U>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Min: {
      auto op = [](U y, T x) { return x < y ? x : y; };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::max();
      scan_op<T, U>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::Max: {
      auto op = [](U y, T x) { return x < y ? y : x; };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::min();
      scan_op<T, U>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
    case Scan::LogAddExp: {
      auto op = [](U a, T b) {
        return detail::LogAddExp{}(a, static_cast<U>(b));
      };
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::min();
      scan_op<T, U>(in, out, axis, reverse, inclusive, op, init);
      break;
    }
  }
}

} // namespace

void Scan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& encoder = cpu::get_command_encoder(stream());

  // Ensure contiguity
  auto in = inputs[0];
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_cpu(in, stream());
    encoder.add_temporary(in);
  }
  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    reduce_type_ = reduce_type_,
                    reverse_ = reverse_,
                    inclusive_ = inclusive_]() mutable {
    switch (in.dtype()) {
      case bool_: {
        // We could do a full dtype x dtype switch but this is the only case
        // where we accumulate in a different type, for now.
        //
        // TODO: If we add the option to accumulate floats in higher precision
        //       floats perhaps we should add the full all-to-all dispatch.
        if (reduce_type_ == Scan::Sum && out.dtype() == int32) {
          scan_dispatch<bool, int32_t>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        } else {
          scan_dispatch<bool, bool>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        }
        break;
      }
      case uint8:
        scan_dispatch<uint8_t, uint8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint16:
        scan_dispatch<uint16_t, uint16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint32:
        scan_dispatch<uint32_t, uint32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint64:
        scan_dispatch<uint64_t, uint64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int8:
        scan_dispatch<int8_t, int8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int16:
        scan_dispatch<int16_t, int16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int32:
        scan_dispatch<int32_t, int32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int64:
        scan_dispatch<int64_t, int64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float16:
        scan_dispatch<float16_t, float16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float32:
        scan_dispatch<float, float>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float64:
        scan_dispatch<double, double>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case bfloat16:
        scan_dispatch<bfloat16_t, bfloat16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case complex64:
        scan_dispatch<complex64_t, complex64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
    }
  });
}

} // namespace mlx::core
