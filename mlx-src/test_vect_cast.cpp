#include <iostream>
#include <arm_neon.h>

int main() {
    float16x8_t a = vdupq_n_f16(1.0);
    float16x8_t b = vdupq_n_f16(0.0);
    
    // Set first lane to be equal!
    b = vsetq_lane_f16(1.0, b, 0);

    uint16x8_t out = vceqq_f16(a, b);
    
    uint16_t first_lane = *(uint16_t*)(&out);
    std::cout << "First lane cast value: " << std::hex << first_lane << std::endl;
    return 0;
}
