#include <nanobind/nanobind.h>
#include "mlx/array.h"
#include <unistd.h>

namespace nb = nanobind;

void init_dummy_ptr(nb::module_& m) {
    m.def("get_data_ptr", [](mlx::core::array& a) {
        float* ptr = a.data<float>();
        if (a.dtype() == mlx::core::float32) {
            volatile float* vptr = ptr;
            vptr[0] = 999.0f;
            printf("[get_data_ptr] Wrote 999.0 to %p, read back: %f\n", ptr, vptr[0]);
            fflush(stdout);
        }
        return (uintptr_t)ptr;
    });
    m.def("print_value", [](mlx::core::array& a) {
        if (a.dtype() == mlx::core::float32) {
            float* p = a.data<float>();
            printf("[print_value] ptr=%p, val=%f\n", p, p[0]);
            fflush(stdout);
        }
    }); // close print_value
    
    m.def("test_write_read", [](mlx::core::array& a) {
        volatile float* ptr = a.data<float>();
        if (a.dtype() == mlx::core::float32) {
            ptr[0] = 777.0f;
            printf("[test_write_read] Read1: %f\n", ptr[0]);
            fflush(stdout);
            usleep(1000 * 1000); // 1 second
            printf("[test_write_read] Read2: %f\n", ptr[0]);
            fflush(stdout);
        }
    });
}
