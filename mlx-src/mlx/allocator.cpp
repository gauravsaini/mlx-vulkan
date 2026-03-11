// Copyright © 2026 Apple Inc.

#include "mlx/allocator.h"

namespace mlx::core::allocator {

namespace {
thread_local int cpu_allocator_override_depth = 0;
}

void push_cpu_allocator_override() {
  cpu_allocator_override_depth++;
}

void pop_cpu_allocator_override() {
  if (cpu_allocator_override_depth > 0) {
    cpu_allocator_override_depth--;
  }
}

bool cpu_allocator_override_enabled() {
  return cpu_allocator_override_depth > 0;
}

} // namespace mlx::core::allocator
