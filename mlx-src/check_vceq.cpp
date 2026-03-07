#include <iostream>
#include <arm_neon.h>
#include "mlx/backend/cpu/simd/simd.h"

int main() {
    float16x8_t a = vdupq_n_f16(1.0);
    float16x8_t b = vdupq_n_f16(0.0);
    uint16x8_t out = vceqq_f16(a, b);
    
    // How to properly construct a Simd<uint16_t, 8> out of uint16x8_t?
    mlx::core::simd::Simd<uint16_t, 8> simdout(out);
    
    // Let's print out what *(uint16_t*)&out gave us
    mlx::core::simd::Simd<uint16_t, 8> buggy(*(uint16_t*)(&out));
    std::cout << "Done" << std::endl;
    return 0;
}
