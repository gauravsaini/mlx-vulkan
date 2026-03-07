#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/mlx.h"
#include <iostream>
using namespace mlx::core;

int main() {
    set_default_device(Device::gpu);
    auto a = array({0.0f, 1.0f});
    auto b = astype(a, bool_);
    mlx::core::eval({b});
    auto b_data = b.data<bool>();
    std::cout << "astype bool (0.0): " << b_data[0] << " (1.0): " << b_data[1] << std::endl;
    
    auto res = logical_not(b);
    mlx::core::eval({res});
    
    // Evaluate logical_not with float input
    auto res2 = logical_not(a);
    mlx::core::eval({res2});
    
    auto data1 = res.data<bool>();
    auto data2 = res2.data<bool>();
    std::cout << "logical_not on bool (0.0): " << data1[0] << " (1.0): " << data1[1] << std::endl;
    std::cout << "logical_not on float (0.0): " << data2[0] << " (1.0): " << data2[1] << std::endl;
    return 0;
}
