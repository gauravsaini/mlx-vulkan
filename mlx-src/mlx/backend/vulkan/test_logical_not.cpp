#include "mlx/array.h"
#include "mlx/ops.h"
#include <iostream>

using namespace mlx::core;

int main() {
    set_default_device(Device::gpu);
    auto a = array({-1.0f, 1.0f, 0.0f, 1.0f, -2.0f, 3.0f});
    auto res = logical_not(a);
    eval(res);
    auto data = res.data<bool>();
    std::cout << "logical_not: " << data[0] << " " << data[1] << " " << data[2] << std::endl;
    return 0;
}
