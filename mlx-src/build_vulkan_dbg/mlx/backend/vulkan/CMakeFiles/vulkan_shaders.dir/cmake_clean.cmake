file(REMOVE_RECURSE
  "CMakeFiles/vulkan_shaders"
  "kernels/arange.spv"
  "kernels/arg_reduce.spv"
  "kernels/binary.spv"
  "kernels/binary_two.spv"
  "kernels/block_masked_mm.spv"
  "kernels/conv.spv"
  "kernels/copy.spv"
  "kernels/fft_bluestein.spv"
  "kernels/fft_four_step_0.spv"
  "kernels/fft_four_step_1.spv"
  "kernels/fft_rader.spv"
  "kernels/fft_stockham.spv"
  "kernels/gather_mm.spv"
  "kernels/gather_mm_coop.spv"
  "kernels/hadamard.spv"
  "kernels/indexing.spv"
  "kernels/logsumexp.spv"
  "kernels/matmul.spv"
  "kernels/matmul_coop.spv"
  "kernels/normalization.spv"
  "kernels/quantized.spv"
  "kernels/radix_sort.spv"
  "kernels/random.spv"
  "kernels/rbits.spv"
  "kernels/reduce.spv"
  "kernels/rope.spv"
  "kernels/scan.spv"
  "kernels/softmax.spv"
  "kernels/sort.spv"
  "kernels/ternary.spv"
  "kernels/unary.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/vulkan_shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
