add_executable(test_gather_cpu test_gather_cpu.cpp)
target_link_libraries(test_gather_cpu PRIVATE mlx)
add_test(NAME test_gather_cpu COMMAND test_gather_cpu)

add_executable(test_mem ../test_mem.cpp)
target_link_libraries(test_mem PRIVATE mlx)
add_test(NAME test_mem COMMAND test_mem)
