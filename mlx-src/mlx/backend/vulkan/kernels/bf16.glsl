// bfloat16 helpers - stored as uint16, converted to/from float32 for math

uint bf16_to_bits(float f) {
    uint bits = floatBitsToUint(f);
    // Round to nearest even
    uint rounding_bias = ((bits >> 16) & 1) + 0x7FFFu;
    return (bits + rounding_bias) >> 16;
}

float bf16_to_float(uint b) {
    return uintBitsToFloat(b << 16);
}
