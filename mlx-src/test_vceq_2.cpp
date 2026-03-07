#include <arm_neon.h>
#include <iostream>
#include "mlx/backend/cpu/simd/simd.h"

int main() {
  float16x8_t a = vdupq_n_f16(1.0);
  float16x8_t b = vdupq_n_f16(0.0);
  // Set first lane to be equal!
  b = vsetq_lane_f16(1.0, b, 0);

  uint16x8_t out = vceqq_f16(a, b);

  // Method 1: reinterpret_cast and copy what MLX does
  mlx::core::simd::Simd<uint16_t, 8> original(*(uint16_t*)(&out));

  // Method 2: correct reinterpret_cast?
  mlx::core::simd::Simd<uint16_t, 8> new_way(
      *reinterpret_cast<uint16x8_t*>(&out));

  // Method 3: direct constructor?
  // mlx::core::simd::Simd<uint16_t, 8> direct(out); // might not compile if
  // unsupported

  std::cout << "original lane 0: " << std::hex << original[0] << std::endl;
  std::cout << "original lane 1: " << std::hex << original[1] << std::endl;
  std::cout << "new_way lane 0: " << std::hex << new_way[0] << std::endl;
  std::cout << "new_way lane 1: " << std::hex << new_way[1] << std::endl;

  return 0;
}
