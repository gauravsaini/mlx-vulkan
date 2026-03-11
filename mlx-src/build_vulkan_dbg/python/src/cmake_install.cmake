# Install script for directory: /Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_stub")
  set(CMD "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14;/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg/_deps/nanobind-src/src/stubgen.py;-q;-r;-i;/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg/..;-i;/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python/src/..;-p;/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python/src/../mlx/_stub_patterns.txt;-m;mlx.core;-O;/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python/src/../mlx")
execute_process(
 COMMAND ${CMD}
 WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}"
)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/build_vulkan_dbg/python/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
