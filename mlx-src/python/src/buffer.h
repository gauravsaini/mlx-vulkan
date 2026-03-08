// Copyright © 2024 Apple Inc.
#pragma once
#include <cstring>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

// Only defined in >= Python 3.9
// https://github.com/python/cpython/blob/f6cdc6b4a191b75027de342aa8b5d344fb31313e/Include/typeslots.h#L2-L3
#ifndef Py_bf_getbuffer
#define Py_bf_getbuffer 1
#define Py_bf_releasebuffer 2
#endif

namespace mx = mlx::core;
namespace nb = nanobind;

namespace {

struct buffer_snapshot {
  std::vector<char> host_data;
  size_t logical_byte_offset{0};
};

inline buffer_snapshot snapshot_array(const mx::array& a) {
  buffer_snapshot snapshot;
  if (a.nbytes() == 0) {
    return snapshot;
  }

  int64_t min_elem_offset = 0;
  int64_t max_elem_offset = 0;
  for (int i = 0; i < a.ndim(); ++i) {
    auto extent = static_cast<int64_t>(a.shape(i) - 1) * a.strides(i);
    if (extent < 0) {
      min_elem_offset += extent;
    } else {
      max_elem_offset += extent;
    }
  }

  auto itemsize = static_cast<int64_t>(a.itemsize());
  auto span_elems = max_elem_offset - min_elem_offset + 1;
  auto span_bytes = static_cast<size_t>(span_elems * itemsize);
  auto base_offset = static_cast<size_t>(a.offset() + min_elem_offset * itemsize);

  snapshot.host_data.resize(span_bytes);
  mx::allocator::copy_to_host(
      a.buffer(), snapshot.host_data.data(), span_bytes, base_offset);
  snapshot.logical_byte_offset =
      static_cast<size_t>(-min_elem_offset * itemsize);
  return snapshot;
}

} // namespace

std::string buffer_format(const mx::array& a) {
  // https://docs.python.org/3.10/library/struct.html#format-characters
  switch (a.dtype()) {
    case mx::bool_:
      return "?";
    case mx::uint8:
      return "B";
    case mx::uint16:
      return "H";
    case mx::uint32:
      return "I";
    case mx::uint64:
      return "Q";
    case mx::int8:
      return "b";
    case mx::int16:
      return "h";
    case mx::int32:
      return "i";
    case mx::int64:
      return "q";
    case mx::float16:
      return "e";
    case mx::float32:
      return "f";
    case mx::bfloat16:
      return "B";
    case mx::float64:
      return "d";
    case mx::complex64:
      return "Zf\0";
    default: {
      std::ostringstream os;
      os << "bad dtype: " << a.dtype();
      throw std::runtime_error(os.str());
    }
  }
}

struct buffer_info {
  std::string format;
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;
  std::vector<char> host_data;

  buffer_info(
      std::string format,
      std::vector<Py_ssize_t> shape_in,
      std::vector<Py_ssize_t> strides_in,
      std::vector<char> host_data_in)
      : format(std::move(format)),
        shape(std::move(shape_in)),
        strides(std::move(strides_in)),
        host_data(std::move(host_data_in)) {}

  buffer_info(const buffer_info&) = delete;
  buffer_info& operator=(const buffer_info&) = delete;

  buffer_info(buffer_info&& other) noexcept {
    (*this) = std::move(other);
  }

  buffer_info& operator=(buffer_info&& rhs) noexcept {
    format = std::move(rhs.format);
    shape = std::move(rhs.shape);
    strides = std::move(rhs.strides);
    host_data = std::move(rhs.host_data);
    return *this;
  }
};

extern "C" inline int getbuffer(PyObject* obj, Py_buffer* view, int flags) {
  std::memset(view, 0, sizeof(Py_buffer));
  auto a = nb::cast<mx::array>(nb::handle(obj));
  auto value = a;

  {
    nb::gil_scoped_release nogil;
    value.eval();
  }

  if (value.has_primitive() && !value.is_tracer()) {
    value.detach();
  }

  std::vector<Py_ssize_t> shape(value.shape().begin(), value.shape().end());
  std::vector<Py_ssize_t> strides(value.strides().begin(), value.strides().end());
  for (auto& s : strides) {
    s *= value.itemsize();
  }
  auto snapshot = snapshot_array(value);
  buffer_info* info = new buffer_info(
      buffer_format(value),
      std::move(shape),
      std::move(strides),
      std::move(snapshot.host_data));

  view->obj = obj;
  view->ndim = value.ndim();
  view->internal = info;
  view->buf = info->host_data.empty()
      ? nullptr
      : (info->host_data.data() + snapshot.logical_byte_offset);
  view->itemsize = value.itemsize();
  view->len = value.nbytes();
  view->readonly = true;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    view->format = const_cast<char*>(info->format.c_str());
  }
  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
    view->strides = info->strides.data();
    view->shape = info->shape.data();
  }
  Py_INCREF(view->obj);
  return 0;
}

extern "C" inline void releasebuffer(PyObject*, Py_buffer* view) {
  delete (buffer_info*)view->internal;
}
