# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-src")
  file(MAKE_DIRECTORY "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-src")
endif()
file(MAKE_DIRECTORY
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-build"
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix"
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/tmp"
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/src"
  "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg311/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
