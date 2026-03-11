// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include <cstdlib>
#include <cstdio>

namespace mlx::core {

namespace {
bool compiled_debug_enabled() {
  const char* env = std::getenv("MLX_COMPILE_DEBUG");
  return env && env[0] != '\0' && env[0] != '0';
}

bool compiled_input_is_donatable(const array& in) {
  if (!in.is_donatable()) {
    return false;
  }
  // Compiled functions must not donate live user-visible leaf inputs. If the
  // array is a leaf, only donate once the compiled output is its sole owner.
  return in.has_primitive() || in.desc_use_count() == 1;
}
} // namespace

void print_constant(std::ostream& os, const array& x) {
  switch (x.dtype()) {
    case float32:
      return print_float_constant<float>(os, x);
    case float16:
      return print_float_constant<float16_t>(os, x);
    case bfloat16:
      return print_float_constant<bfloat16_t>(os, x);
    case float64:
      return print_float_constant<double>(os, x);
    case complex64:
      return print_complex_constant<complex64_t>(os, x);
    case int8:
      os << static_cast<int32_t>(x.item<int8_t>());
      return;
    case int16:
      return print_int_constant<int16_t>(os, x);
    case int32:
      return print_int_constant<int32_t>(os, x);
    case int64:
      return print_int_constant<int64_t>(os, x);
    case uint8:
      os << static_cast<uint32_t>(x.item<uint8_t>());
      return;
    case uint16:
      return print_int_constant<uint16_t>(os, x);
    case uint32:
      return print_int_constant<uint32_t>(os, x);
    case uint64:
      return print_int_constant<uint64_t>(os, x);
    case bool_:
      os << std::boolalpha << x.item<bool>();
      return;
    default:
      throw std::runtime_error("Unsupported constant type");
  }
}

std::string get_type_string(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "bfloat16_t";
    case float64:
      return "double";
    case complex64:
      return "complex64_t";
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    default: {
      std::ostringstream msg;
      msg << "Unsupported compilation type " << d;
      throw std::runtime_error(msg.str());
    }
  }
}

bool compiled_check_contiguity(
    const std::vector<array>& inputs,
    const Shape& shape) {
  bool contiguous = true;
  bool all_contig = true;
  bool all_row_contig = true;
  bool all_col_contig = true;
  int non_scalar_inputs = 0;
  for (const auto& x : inputs) {
    if (is_scalar(x)) {
      continue;
    }
    non_scalar_inputs++;
    bool shape_eq = x.shape() == shape;
    all_contig &= (x.flags().contiguous && shape_eq);
    all_row_contig &= (x.flags().row_contiguous && shape_eq);
    all_col_contig &= (x.flags().col_contiguous && shape_eq);
  }
  if (non_scalar_inputs > 1 && !all_row_contig && !all_col_contig) {
    contiguous = false;
  } else if (non_scalar_inputs == 1 && !all_contig) {
    contiguous = false;
  } else if (non_scalar_inputs == 0 && !shape.empty()) {
    contiguous = false;
  }
  return contiguous;
}

void compiled_allocate_outputs(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    const std::function<allocator::Buffer(size_t)>&
        mallocfn /* = allocator::malloc */) {
  if (contiguous) {
    int o = 0;
    Strides strides;
    size_t data_size;
    array::Flags flags;
    for (int i = 0; i < inputs.size() && o < outputs.size(); ++i) {
      auto& in = inputs[i];
      // Conditions for donation
      // - Correct size
      // - Not a scalar
      // - Donatable
      // - Not a constant
      if (in.itemsize() == outputs[o].itemsize() && !is_scalar(in) &&
          compiled_input_is_donatable(in) && !is_constant(i)) {
        if (compiled_debug_enabled()) {
          std::fprintf(
              stderr,
              "[compiled allocate] donate contiguous input=%d output=%d "
              "desc=%zu data=%zu in_ptr=%p out_size=%zu\n",
              i,
              o,
              in.desc_use_count(),
              in.data_use_count(),
              in.buffer().ptr(),
              outputs[o].size());
        }
        outputs[o++].copy_shared_buffer(in);
      } else if (compiled_debug_enabled()) {
        std::fprintf(
            stderr,
            "[compiled allocate] skip contiguous input=%d output=%d "
            "same_itemsize=%d scalar=%d donatable=%d constant=%d "
            "desc=%zu data=%zu in_size=%zu out_size=%zu\n",
            i,
            o,
            in.itemsize() == outputs[o].itemsize(),
            is_scalar(in),
            compiled_input_is_donatable(in),
            is_constant(i),
            in.desc_use_count(),
            in.data_use_count(),
            in.size(),
            outputs[o].size());
      }
      // Get representative input flags to properly set non-donated outputs
      if (strides.empty() && in.size() == outputs[0].size()) {
        strides = in.strides();
        flags = in.flags();
        data_size = in.data_size();
      }
    }
    for (; o < outputs.size(); ++o) {
      outputs[o].set_data(
          mallocfn(data_size * outputs[o].itemsize()),
          data_size,
          strides,
          flags);
    }
  } else {
    int o = 0;
    for (int i = 0; i < inputs.size() && o < outputs.size(); ++i) {
      auto& in = inputs[i];
      // Conditions for donation
      // - Row contiguous
      // - Donatable
      // - Correct size
      // - Not a constant
      if (
          in.flags().row_contiguous && in.size() == outputs[o].size() &&
          in.itemsize() == outputs[o].itemsize() &&
          compiled_input_is_donatable(in) && !is_constant(i)) {
        if (compiled_debug_enabled()) {
          std::fprintf(
              stderr,
              "[compiled allocate] donate strided input=%d output=%d "
              "desc=%zu data=%zu in_ptr=%p out_size=%zu\n",
              i,
              o,
              in.desc_use_count(),
              in.data_use_count(),
              in.buffer().ptr(),
              outputs[o].size());
        }
        outputs[o].copy_shared_buffer(
            in, outputs[o].strides(), in.flags(), in.data_size());
        o++;
      } else if (compiled_debug_enabled()) {
        std::fprintf(
            stderr,
            "[compiled allocate] skip strided input=%d output=%d "
            "row_contig=%d same_size=%d same_itemsize=%d donatable=%d "
            "constant=%d desc=%zu data=%zu in_size=%zu out_size=%zu\n",
            i,
            o,
            in.flags().row_contiguous,
            in.size() == outputs[o].size(),
            in.itemsize() == outputs[o].itemsize(),
            compiled_input_is_donatable(in),
            is_constant(i),
            in.desc_use_count(),
            in.data_use_count(),
            in.size(),
            outputs[o].size());
      }
    }
    for (; o < outputs.size(); ++o) {
      outputs[o].set_data(mallocfn(outputs[o].nbytes()));
    }
  }
}

std::tuple<bool, Shape, std::vector<Strides>> compiled_collapse_contiguous_dims(
    const std::vector<array>& inputs,
    const array& out,
    const std::function<bool(size_t)>& is_constant) {
  const Shape& shape = out.shape();
  bool contiguous = compiled_check_contiguity(inputs, shape);
  if (contiguous) {
    return {true, shape, {}};
  }

  std::vector<Strides> strides_vec{out.strides()};
  for (size_t i = 0; i < inputs.size(); ++i) {
    // Skip constants.
    if (is_constant(i)) {
      continue;
    }

    // Skip scalar inputs.
    const auto& x = inputs[i];
    if (is_scalar(x)) {
      continue;
    }

    // Broadcast the inputs to the output shape.
    Strides xstrides;
    size_t j = 0;
    for (; j < shape.size() - x.ndim(); ++j) {
      if (shape[j] == 1) {
        xstrides.push_back(out.strides()[j]);
      } else {
        xstrides.push_back(0);
      }
    }
    for (size_t i = 0; i < x.ndim(); ++i, ++j) {
      if (x.shape(i) == 1) {
        if (shape[j] == 1) {
          xstrides.push_back(out.strides()[j]);
        } else {
          xstrides.push_back(0);
        }
      } else {
        xstrides.push_back(x.strides()[i]);
      }
    }
    strides_vec.push_back(std::move(xstrides));
  }

  auto tup = collapse_contiguous_dims(shape, strides_vec, INT32_MAX);
  return {false, std::move(std::get<0>(tup)), std::move(std::get<1>(tup))};
}

bool compiled_use_large_index(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    bool contiguous) {
  if (contiguous) {
    size_t max_size = 0;
    for (const auto& in : inputs) {
      max_size = std::max(max_size, in.data_size());
    }
    return max_size > UINT32_MAX;
  } else {
    size_t max_size = 0;
    for (const auto& o : outputs) {
      max_size = std::max(max_size, o.size());
    }
    return max_size > UINT32_MAX;
  }
}

} // namespace mlx::core
