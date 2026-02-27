#include "mlx/mlx.h"
#include <iostream>
using namespace mlx::core;
int main() {
    array a({1, 2, 3, 4}, {2, 2});
    eval(a);
    std::cout << "Pointer: " << (void*)a.data<int>() << std::endl;
    std::cout << "Data: " << a.data<int>()[0] << std::endl;
    return 0;
}
