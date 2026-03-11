#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mlx" for configuration "Debug"
set_property(TARGET mlx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(mlx PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmlx.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libmlx.dylib"
  )

list(APPEND _cmake_import_check_targets mlx )
list(APPEND _cmake_import_check_files_for_mlx "${_IMPORT_PREFIX}/lib/libmlx.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
