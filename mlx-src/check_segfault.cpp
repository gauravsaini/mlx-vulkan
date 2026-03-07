#include "mlx/array.h"
#include "mlx/ops.h"
#include <iostream>

using namespace mlx::core;

int main() {
    set_default_device(Device::cpu);
    // Trigger Gather
    auto a = array({1.0f, 2.0f, 3.0f});
    auto idx = array({0, 2}, int32);
    auto g = take(a, idx);
    eval(g);
    std::cout << "Take GPU passed\n";
    
    // Trigger Logical Not
    auto b = array({1.0f, 0.0f}, float32);
    auto n = logical_not(b);
    eval(n);
    std::cout << "LogicalNot GPU passed\n";
    
    return 0;
}
