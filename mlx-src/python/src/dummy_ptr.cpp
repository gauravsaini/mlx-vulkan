#include <nanobind/nanobind.h>
#include "mlx/array.h"
#include "mlx/primitives.h"
#include <cstdint>
#include <iostream>
#include <unistd.h>

namespace nb = nanobind;

void init_dummy_ptr(nb::module_& m) {
    m.def("dump_graph", [](mlx::core::array& a, int max_depth) {
        std::function<void(const mlx::core::array&, int, int)> walk =
            [&](const mlx::core::array& node, int depth, int index) {
                for (int i = 0; i < depth; ++i) {
                    std::cout << "  ";
                }
                std::cout
                    << "[" << index << "]"
                    << " status=" << static_cast<int>(node.status())
                    << " has_primitive=" << node.has_primitive()
                    << " primitive="
                    << (node.has_primitive() ? node.primitive().name() : "leaf")
                    << " data=" << (node.data_shared_ptr() != nullptr)
                    << " ndim=" << node.ndim()
                    << " size=" << node.size()
                    << " dtype=" << static_cast<int>(node.dtype().val())
                    << " inputs=" << node.inputs().size()
                    << "\n";
                if (depth >= max_depth) {
                    return;
                }
                int i = 0;
                for (const auto& in : node.inputs()) {
                    walk(in, depth + 1, i++);
                }
            };
        walk(a, 0, 0);
        std::cout.flush();
    });

    m.def("debug_array_info", [](mlx::core::array& a) {
        return nb::make_tuple(
            static_cast<unsigned long long>(a.id()),
            a.has_primitive(),
            a.siblings().size(),
            a.sibling_position(),
            a.inputs().size());
    });

    m.def("debug_array_counts", [](mlx::core::array& a) {
        return nb::make_tuple(
            a.desc_use_count(),
            a.data_use_count(),
            a.is_donatable(),
            a.has_primitive(),
            a.inputs().size());
    });

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

    m.def("print_buffer_debug", [](mlx::core::array& a) {
        auto raw_buffer = a.buffer().ptr();
        auto raw_ptr = a.buffer().raw_ptr();
        uint64_t tag = raw_buffer ? *reinterpret_cast<const uint64_t*>(raw_buffer) : 0;
        printf(
            "[print_buffer_debug] buffer=%p raw=%p tag=0x%016llx\n",
            raw_buffer,
            raw_ptr,
            static_cast<unsigned long long>(tag));
        fflush(stdout);
    });
    
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
