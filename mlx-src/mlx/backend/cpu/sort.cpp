// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// NaN-aware comparator that places NaNs at the end
template <typename T>
bool nan_aware_less(T a, T b) {
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, complex64_t>) {
    if (std::isnan(a))
      return false;
    if (std::isnan(b))
      return true;
  }
  return a < b;
}

template <typename T>
struct StridedIterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int32_t;
  using value_type = T;
  using reference = value_type&;
  using pointer = value_type*;

  // Constructors
  StridedIterator() = default;

  explicit StridedIterator(T* ptr, int64_t stride, difference_type offset = 0)
      : stride_(stride), ptr_(ptr + offset * stride) {}

  explicit StridedIterator(array& arr, int axis, difference_type offset = 0)
      : StridedIterator(arr.data<T>(), arr.strides()[axis], offset) {}

  // Accessors
  reference operator*() const {
    return ptr_[0];
  }

  reference operator[](difference_type idx) const {
    return ptr_[idx * stride_];
  }

  // Comparisons
  bool operator==(const StridedIterator& other) const {
    return ptr_ == other.ptr_ && stride_ == other.stride_;
  }

  bool operator!=(const StridedIterator& other) const {
    return ptr_ != other.ptr_;
  }

  bool operator<(const StridedIterator& other) const {
    return ptr_ < other.ptr_;
  }

  bool operator>(const StridedIterator& other) const {
    return ptr_ > other.ptr_;
  }

  bool operator<=(const StridedIterator& other) const {
    return ptr_ <= other.ptr_;
  }

  bool operator>=(const StridedIterator& other) const {
    return ptr_ >= other.ptr_;
  }

  difference_type operator-(const StridedIterator& other) const {
    return (ptr_ - other.ptr_) / stride_;
  }

  // Moving
  StridedIterator& operator++() {
    ptr_ += stride_;
    return *this;
  }

  StridedIterator& operator--() {
    ptr_ -= stride_;
    return *this;
  }

  StridedIterator& operator+=(difference_type diff) {
    ptr_ += diff * stride_;
    return *this;
  }

  StridedIterator& operator-=(difference_type diff) {
    ptr_ -= diff * stride_;
    return *this;
  }

  StridedIterator operator+(difference_type diff) {
    return StridedIterator(ptr_, stride_, diff);
  }

  StridedIterator operator-(difference_type diff) {
    return StridedIterator(ptr_, stride_, -diff);
  }

 private:
  int64_t stride_;
  T* ptr_;
};

template <typename T>
void sort_buffer(T* out_ptr, const Shape& shape, const Strides& strides, int axis) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + shape.size() : axis;
  size_t in_size =
      std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
  size_t n_rows = in_size / shape[axis];

  auto remaining_shape = shape;
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = strides;
  remaining_strides.erase(remaining_strides.begin() + axis);

  auto axis_stride = strides[axis];
  auto axis_size = shape[axis];

  // Perform sorting in place
  ContiguousIterator src_it(
      remaining_shape, remaining_strides, remaining_shape.size());
  for (int i = 0; i < n_rows; i++) {
    T* data_ptr = out_ptr + src_it.loc;

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::stable_sort(st, ed, nan_aware_less<T>);
    src_it.step();
  }
}

template <typename T, typename IdxT = uint32_t>
void argsort_buffer(
    const T* in_ptr,
    const Shape& in_shape,
    const Strides& in_strides,
    IdxT* out_ptr,
    const Shape& out_shape,
    const Strides& out_strides,
    int axis) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in_shape.size() : axis;
  size_t n_rows = std::accumulate(
                      in_shape.begin(),
                      in_shape.end(),
                      size_t{1},
                      std::multiplies<size_t>()) /
      in_shape[axis];

  auto in_remaining_shape = in_shape;
  in_remaining_shape.erase(in_remaining_shape.begin() + axis);

  auto in_remaining_strides = in_strides;
  in_remaining_strides.erase(in_remaining_strides.begin() + axis);

  auto out_remaining_shape = out_shape;
  out_remaining_shape.erase(out_remaining_shape.begin() + axis);

  auto out_remaining_strides = out_strides;
  out_remaining_strides.erase(out_remaining_strides.begin() + axis);

  auto in_stride = in_strides[axis];
  auto out_stride = out_strides[axis];
  auto axis_size = in_shape[axis];

  // Perform sorting
  ContiguousIterator in_it(
      in_remaining_shape, in_remaining_strides, in_remaining_shape.size());
  ContiguousIterator out_it(
      out_remaining_shape, out_remaining_strides, out_remaining_shape.size());
  for (int i = 0; i < n_rows; i++) {
    const T* data_ptr = in_ptr + in_it.loc;
    IdxT* idx_ptr = out_ptr + out_it.loc;

    in_it.step();
    out_it.step();

    StridedIterator st_(idx_ptr, out_stride, 0);
    StridedIterator ed_(idx_ptr, out_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, out_stride, 0);
    StridedIterator ed(idx_ptr, out_stride, axis_size);

    std::stable_sort(st, ed, [data_ptr, in_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * in_stride];
      auto v2 = data_ptr[b * in_stride];

      // Handle NaNs (place them at the end)
      if (std::is_floating_point<T>::value) {
        if (std::isnan(v1))
          return false;
        if (std::isnan(v2))
          return true;
      }

      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

template <typename T>
void partition_buffer(
    T* out_ptr,
    const Shape& shape,
    const Strides& strides,
    int axis,
    int kth) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + shape.size() : axis;
  size_t in_size =
      std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
  size_t n_rows = in_size / shape[axis];

  auto remaining_shape = shape;
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = strides;
  remaining_strides.erase(remaining_strides.begin() + axis);

  auto axis_stride = strides[axis];
  int axis_size = shape[axis];

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition in place
  ContiguousIterator src_it(
      remaining_shape, remaining_strides, remaining_shape.size());
  for (int i = 0; i < n_rows; i++) {
    T* data_ptr = out_ptr + src_it.loc;
    src_it.step();

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator md(data_ptr, axis_stride, kth);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::nth_element(st, md, ed, nan_aware_less<T>);
  }
}

template <typename T, typename IdxT = uint32_t>
void argpartition_buffer(
    const T* in_ptr,
    const Shape& in_shape,
    const Strides& in_strides,
    IdxT* out_ptr,
    const Shape& out_shape,
    const Strides& out_strides,
    int axis,
    int kth) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in_shape.size() : axis;
  size_t n_rows = std::accumulate(
                      in_shape.begin(),
                      in_shape.end(),
                      size_t{1},
                      std::multiplies<size_t>()) /
      in_shape[axis];

  auto in_remaining_shape = in_shape;
  in_remaining_shape.erase(in_remaining_shape.begin() + axis);

  auto in_remaining_strides = in_strides;
  in_remaining_strides.erase(in_remaining_strides.begin() + axis);

  auto out_remaining_shape = out_shape;
  out_remaining_shape.erase(out_remaining_shape.begin() + axis);

  auto out_remaining_strides = out_strides;
  out_remaining_strides.erase(out_remaining_strides.begin() + axis);

  auto in_stride = in_strides[axis];
  auto out_stride = out_strides[axis];
  auto axis_size = in_shape[axis];

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition
  ContiguousIterator in_it(
      in_remaining_shape, in_remaining_strides, in_remaining_shape.size());
  ContiguousIterator out_it(
      out_remaining_shape, out_remaining_strides, out_remaining_shape.size());

  for (int i = 0; i < n_rows; i++) {
    const T* data_ptr = in_ptr + in_it.loc;
    IdxT* idx_ptr = out_ptr + out_it.loc;
    in_it.step();
    out_it.step();

    StridedIterator st_(idx_ptr, out_stride, 0);
    StridedIterator ed_(idx_ptr, out_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, out_stride, 0);
    StridedIterator md(idx_ptr, out_stride, kth);
    StridedIterator ed(idx_ptr, out_stride, axis_size);

    std::nth_element(st, md, ed, [data_ptr, in_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * in_stride];
      auto v2 = data_ptr[b * in_stride];

      // Handle NaNs (place them at the end)
      if (std::is_floating_point<T>::value) {
        if (std::isnan(v1))
          return false;
        if (std::isnan(v2))
          return true;
      }

      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

} // namespace

void ArgSort::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_]() mutable {
    std::vector<uint32_t> host_out(out.size());
    switch (in.dtype()) {
      case bool_:
        argsort_buffer<bool>(
            in.data<bool>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case uint8:
        argsort_buffer<uint8_t>(
            in.data<uint8_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case uint16:
        argsort_buffer<uint16_t>(
            in.data<uint16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case uint32:
        argsort_buffer<uint32_t>(
            in.data<uint32_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case uint64:
        argsort_buffer<uint64_t>(
            in.data<uint64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case int8:
        argsort_buffer<int8_t>(
            in.data<int8_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case int16:
        argsort_buffer<int16_t>(
            in.data<int16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case int32:
        argsort_buffer<int32_t>(
            in.data<int32_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case int64:
        argsort_buffer<int64_t>(
            in.data<int64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case float32:
        argsort_buffer<float>(
            in.data<float>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case float64:
        argsort_buffer<double>(
            in.data<double>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case float16:
        argsort_buffer<float16_t>(
            in.data<float16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case bfloat16:
        argsort_buffer<bfloat16_t>(
            in.data<bfloat16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
      case complex64:
        argsort_buffer<complex64_t>(
            in.data<complex64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_);
        break;
    }
    allocator::copy_from_host(
        out.buffer(), host_out.data(), host_out.size() * sizeof(uint32_t));
  });
}

void Sort::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  int axis = axis_;
  if (axis < 0) {
    axis += in.ndim();
  }

  // Copy input to output
  CopyType ctype = (in.flags().contiguous && in.strides()[axis] != 0)
      ? CopyType::Vector
      : CopyType::General;
  copy_cpu(in, out, ctype, stream());

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out), axis]() mutable {
    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      std::vector<uint8_t> host_bytes(out.size() * sizeof(T));
      auto* host_out = reinterpret_cast<T*>(host_bytes.data());
      allocator::copy_to_host(
          out.buffer(), host_out, host_bytes.size());
      sort_buffer<T>(host_out, out.shape(), out.strides(), axis);
      allocator::copy_from_host(
          out.buffer(), host_out, host_bytes.size());
    });
  });
}

void ArgPartition::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    kth_ = kth_]() mutable {
    std::vector<uint32_t> host_out(out.size());
    switch (in.dtype()) {
      case bool_:
        argpartition_buffer<bool>(
            in.data<bool>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case uint8:
        argpartition_buffer<uint8_t>(
            in.data<uint8_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case uint16:
        argpartition_buffer<uint16_t>(
            in.data<uint16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case uint32:
        argpartition_buffer<uint32_t>(
            in.data<uint32_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case uint64:
        argpartition_buffer<uint64_t>(
            in.data<uint64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case int8:
        argpartition_buffer<int8_t>(
            in.data<int8_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case int16:
        argpartition_buffer<int16_t>(
            in.data<int16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case int32:
        argpartition_buffer<int32_t>(
            in.data<int32_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case int64:
        argpartition_buffer<int64_t>(
            in.data<int64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case float32:
        argpartition_buffer<float>(
            in.data<float>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case float64:
        argpartition_buffer<double>(
            in.data<double>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case float16:
        argpartition_buffer<float16_t>(
            in.data<float16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case bfloat16:
        argpartition_buffer<bfloat16_t>(
            in.data<bfloat16_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
      case complex64:
        argpartition_buffer<complex64_t>(
            in.data<complex64_t>(),
            in.shape(),
            in.strides(),
            host_out.data(),
            out.shape(),
            out.strides(),
            axis_,
            kth_);
        break;
    }
    allocator::copy_from_host(
        out.buffer(), host_out.data(), host_out.size() * sizeof(uint32_t));
  });
}

void Partition::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Copy input to output
  CopyType ctype = (in.flags().contiguous && in.strides()[axis_] != 0)
      ? CopyType::Vector
      : CopyType::General;
  copy_cpu(in, out, ctype, stream());

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    kth_ = kth_]() mutable {
    switch (out.dtype()) {
      case bool_:
        {
          std::vector<uint8_t> host_bytes(out.size() * sizeof(bool));
          auto* host_out = reinterpret_cast<bool*>(host_bytes.data());
          allocator::copy_to_host(out.buffer(), host_out, host_bytes.size());
          partition_buffer<bool>(host_out, out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(out.buffer(), host_out, host_bytes.size());
          return;
        }
      case uint8:
        {
          std::vector<uint8_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint8_t));
          partition_buffer<uint8_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint8_t));
          return;
        }
      case uint16:
        {
          std::vector<uint16_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint16_t));
          partition_buffer<uint16_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint16_t));
          return;
        }
      case uint32:
        {
          std::vector<uint32_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint32_t));
          partition_buffer<uint32_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint32_t));
          return;
        }
      case uint64:
        {
          std::vector<uint64_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint64_t));
          partition_buffer<uint64_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(uint64_t));
          return;
        }
      case int8:
        {
          std::vector<int8_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int8_t));
          partition_buffer<int8_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int8_t));
          return;
        }
      case int16:
        {
          std::vector<int16_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int16_t));
          partition_buffer<int16_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int16_t));
          return;
        }
      case int32:
        {
          std::vector<int32_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int32_t));
          partition_buffer<int32_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int32_t));
          return;
        }
      case int64:
        {
          std::vector<int64_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int64_t));
          partition_buffer<int64_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(int64_t));
          return;
        }
      case float32:
        {
          std::vector<float> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(float));
          partition_buffer<float>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(float));
          return;
        }
      case float64:
        {
          std::vector<double> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(double));
          partition_buffer<double>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(), host_out.data(), host_out.size() * sizeof(double));
          return;
        }
      case float16:
        {
          std::vector<float16_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(float16_t));
          partition_buffer<float16_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(float16_t));
          return;
        }
      case bfloat16:
        {
          std::vector<bfloat16_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(bfloat16_t));
          partition_buffer<bfloat16_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(bfloat16_t));
          return;
        }
      case complex64:
        {
          std::vector<complex64_t> host_out(out.size());
          allocator::copy_to_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(complex64_t));
          partition_buffer<complex64_t>(
              host_out.data(), out.shape(), out.strides(), axis_, kth_);
          allocator::copy_from_host(
              out.buffer(),
              host_out.data(),
              host_out.size() * sizeof(complex64_t));
          return;
        }
    }
  });
}

} // namespace mlx::core
