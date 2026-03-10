#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include <iostream>
#include <unistd.h>

using namespace mlx::core;

int main() {
    array x(1.0f);
    array y(2.0f);
    array a = add(x, y, default_stream(Device::cpu));
    eval({a});

    float* ptr = a.data<float>();
    std::cout << "[CPP] ptr1 addr: " << ptr << std::endl;
    
    std::cout << "[CPP] Sleeping 1 sec BEFORE writing..." << std::endl;
    usleep(1000 * 1000); // 1 sec
    
    ptr[0] = 777.0f;
    std::cout << "[CPP] Read1: " << ptr[0] << std::endl;

    usleep(1000 * 1000); // 1 sec
    std::cout << "[CPP] Read2: " << ptr[0] << std::endl;
    
    float* ptr2 = a.data<float>();
    std::cout << "[CPP] ptr2 addr: " << ptr2 << std::endl;
    std::cout << "[CPP] Read3: " << ptr2[0] << std::endl;
    return 0;
}
