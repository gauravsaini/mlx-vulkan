# MLX Vulkan Backend CMakeLists.txt

# Find glslc for AOT SPIR-V shader compilation
find_program(GLSLC_PROGRAM glslc REQUIRED
  HINTS $ENV{VULKAN_SDK}/bin)
message(STATUS "Found glslc: ${GLSLC_PROGRAM}")

# Create output directory for compiled shaders
set(VULKAN_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernels)
file(MAKE_DIRECTORY ${VULKAN_SHADER_OUTPUT_DIR})

# Helper to compile a GLSL compute shader to SPIR-V
set(SPIRV_OUTPUTS "")
function(compile_shader SHADER_NAME)
  set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${SHADER_NAME}.comp)
  set(SHADER_SPV ${VULKAN_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
  add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_PROGRAM}
      --target-env=vulkan1.2
      -fshader-stage=compute
      -I${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -o ${SHADER_SPV}
      ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling shader ${SHADER_NAME}.comp -> ${SHADER_NAME}.spv"
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${SHADER_SPV})
  set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} PARENT_SCOPE)
endfunction()

# Compile all shaders
compile_shader(copy)
compile_shader(unary)
compile_shader(binary)
compile_shader(binary_two)
compile_shader(ternary)
compile_shader(reduce)
compile_shader(arg_reduce)
compile_shader(matmul)
compile_shader(softmax)
compile_shader(arange)
compile_shader(indexing)
compile_shader(scan)
compile_shader(sort)
compile_shader(radix_sort)
compile_shader(normalization)
compile_shader(rope)
compile_shader(logsumexp)
compile_shader(random)
compile_shader(quantized)
compile_shader(conv)
compile_shader(rbits)
compile_shader(gather_mm)
compile_shader(block_masked_mm)

# Compile fft.comp into the various C++ dispatch names
add_custom_command(
  OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/fft_stockham.spv
  COMMAND ${GLSLC_PROGRAM} --target-env=vulkan1.2 -fshader-stage=compute -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -o ${VULKAN_SHADER_OUTPUT_DIR}/fft_stockham.spv ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  VERBATIM)
list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/fft_stockham.spv)

# Compile matmul.comp into matmul_coop.spv using cooperative matrix extension
# Only compile on non-Apple platforms since MoltenVK doesn't support VK_KHR_cooperative_matrix
if(NOT APPLE)
  add_custom_command(
    OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv
    COMMAND ${GLSLC_PROGRAM} --target-env=vulkan1.2 -fshader-stage=compute -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -DMLX_USE_COOP_MAT=1 -o ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matmul.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matmul.comp
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv)

  add_custom_command(
    OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv
    COMMAND ${GLSLC_PROGRAM} --target-env=vulkan1.2 -fshader-stage=compute -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -DMLX_USE_COOP_MAT=1 -o ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather_mm.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather_mm.comp
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv)
else()
  # Create empty dummy files to satisfy the vulkan_shaders ALL DEPENDS target without breaking build
  add_custom_command(
    OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv
    COMMAND ${CMAKE_COMMAND} -E touch ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/matmul_coop.spv)
  
  add_custom_command(
    OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv
    COMMAND ${CMAKE_COMMAND} -E touch ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/gather_mm_coop.spv)
endif()

add_custom_command(
  OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/fft_rader.spv
  COMMAND ${GLSLC_PROGRAM} --target-env=vulkan1.2 -fshader-stage=compute -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -o ${VULKAN_SHADER_OUTPUT_DIR}/fft_rader.spv ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  VERBATIM)
list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/fft_rader.spv)

add_custom_command(
  OUTPUT ${VULKAN_SHADER_OUTPUT_DIR}/fft_bluestein.spv
  COMMAND ${GLSLC_PROGRAM} --target-env=vulkan1.2 -fshader-stage=compute -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -o ${VULKAN_SHADER_OUTPUT_DIR}/fft_bluestein.spv ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft.comp
  VERBATIM)
list(APPEND SPIRV_OUTPUTS ${VULKAN_SHADER_OUTPUT_DIR}/fft_bluestein.spv)

compile_shader(hadamard)

add_custom_target(vulkan_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(mlx vulkan_shaders)

# Add Vulkan backend sources to mlx target
target_sources(
  mlx
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

# Link Vulkan and VMA
target_link_libraries(mlx PRIVATE Vulkan::Vulkan)

# VMA is header-only — include path only, never link as an exported target
find_package(VulkanMemoryAllocator QUIET)
if(VulkanMemoryAllocator_FOUND)
  # Use include dir from the found package, not the CMake target (avoids export issues)
  get_target_property(_vma_include GPUOpen::VulkanMemoryAllocator INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(mlx PRIVATE ${_vma_include})
else()
  message(STATUS "VulkanMemoryAllocator not found via find_package, using FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(VulkanMemoryAllocator)
  # VMA is header-only — include directly rather than link, avoids export set issues
  target_include_directories(mlx PRIVATE ${vulkanmemoryallocator_SOURCE_DIR}/include)
endif()

# Set compile definitions
target_compile_definitions(mlx PRIVATE
  MLX_USE_VULKAN
  VULKAN_KERNELS_PATH="${VULKAN_SHADER_OUTPUT_DIR}/")

target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
