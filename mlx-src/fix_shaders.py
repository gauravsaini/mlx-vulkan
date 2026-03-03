import os
import re

files = [
    "reduce.comp",
    "arg_reduce.comp",
    "softmax.comp",
    "normalization.comp",
    "logsumexp.comp",
    "scan.comp",
    "hadamard.comp"
]

for f in files:
    path = f"mlx/backend/vulkan/kernels/{f}"
    with open(path, 'r') as file:
        content = file.read()

    # Add WORKGROUP_SIZE constant
    if "WORKGROUP_SIZE" not in content:
        content = content.replace("layout(local_size_x_id = 0", "layout(constant_id = 0) const uint WORKGROUP_SIZE = 256;\nlayout(local_size_x_id = 0")
    
    # Replace hardcoded strides
    content = content.replace("i += 256", "i += WORKGROUP_SIZE")
    content = content.replace("stride = 128", "stride = WORKGROUP_SIZE / 2")
    content = content.replace("= 256u", "= WORKGROUP_SIZE")
    content = content.replace("/ 256u", "/ WORKGROUP_SIZE")
    content = content.replace("+ 255u", "+ (WORKGROUP_SIZE - 1)")
    
    # Replace shared array allocations safely
    content = re.sub(r'shared (\w+) (\w+)\[256\]', r'shared \1 \2[WORKGROUP_SIZE]', content)

    with open(path, 'w') as file:
        file.write(content)

print("Shaders fixed!")
