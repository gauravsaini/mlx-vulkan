import re
with open('/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/mlx/backend/vulkan/kernels/unary.comp', 'r') as f:
    text = f.read()
# Replace val == 0.0 with bitwise check for 0.0 or -0.0
new_text = re.sub(r'case UNARY_LOGNOT:\s*return \(val == 0\.0\) \? 1\.0 : 0\.0;', 
                  'case UNARY_LOGNOT: { uint u = floatBitsToUint(val); return ((u & 0x7FFFFFFFu) == 0u) ? 1.0 : 0.0; }', text)
with open('/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/mlx/backend/vulkan/kernels/unary.comp', 'w') as f:
    f.write(new_text)
