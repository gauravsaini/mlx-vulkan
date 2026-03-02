// bfloat16 helpers - stored as uint16, converted to/from float32 for math

uint bf16_to_bits(float f) {
    if (isnan(f)) return 0x7FC0u; // standard quiet NaN for bfloat16
    uint bits = floatBitsToUint(f);
    // Round to nearest even
    uint rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    return (bits + rounding_bias) >> 16u;
}

float bf16_to_float(uint b) {
    return uintBitsToFloat(b << 16u);
}

vec2 unpackBfloat2x16(uint val) {
    float f0 = bf16_to_float(val & 0xFFFFu);
    float f1 = bf16_to_float(val >> 16u);
    return vec2(f0, f1);
}

uint packBfloat2x16(vec2 v) {
    uint lo = bf16_to_bits(v.x);
    uint hi = bf16_to_bits(v.y);
    return lo | (hi << 16u);
}
