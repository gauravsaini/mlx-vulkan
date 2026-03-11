// Copyright © 2024 Apple Inc.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdlib>
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

struct buffer_owner {
  mx::array array_ref;
  std::string format;
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;
  void* data{nullptr};

  buffer_owner(
      mx::array array_ref,
      std::string format,
      std::vector<Py_ssize_t> shape_in,
      std::vector<Py_ssize_t> strides_in,
      void* data_ptr)
      : array_ref(std::move(array_ref)),
        format(std::move(format)),
        shape(std::move(shape_in)),
        strides(std::move(strides_in)),
        data(data_ptr) {}

  buffer_owner(const buffer_owner&) = delete;
  buffer_owner& operator=(const buffer_owner&) = delete;
};

extern "C" inline int getbuffer(PyObject* obj, Py_buffer* view, int flags) {
  std::memset(view, 0, sizeof(Py_buffer));
  if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
    PyErr_SetString(PyExc_BufferError, "mlx arrays expose a read-only buffer");
    return -1;
  }
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
  auto* info = new buffer_owner(
      value,
      buffer_format(value),
      std::move(shape),
      std::move(strides),
      value.nbytes() == 0 ? nullptr : value.data<void>());
  PyObject* owner = PyCapsule_New(
      info,
      "mlx.buffer_owner",
      [](PyObject* capsule) {
        auto* owner = static_cast<buffer_owner*>(
            PyCapsule_GetPointer(capsule, "mlx.buffer_owner"));
        delete owner;
      });
  if (owner == nullptr) {
    delete info;
    return -1;
  }

  view->obj = owner;
  view->ndim = value.ndim();
  view->internal = info;
  view->buf = info->data;
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
  return 0;
}

extern "C" inline void releasebuffer(PyObject*, Py_buffer* view) {
  view->internal = nullptr;
}
