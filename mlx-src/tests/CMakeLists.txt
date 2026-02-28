add_executable(test_gather_cpu test_gather_cpu.cpp)
target_link_libraries(test_gather_cpu PRIVATE mlx)
add_test(NAME test_gather_cpu COMMAND test_gather_cpu)
