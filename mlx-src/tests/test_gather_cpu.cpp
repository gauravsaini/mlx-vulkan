#include "mlx/mlx.h"
#include <iostream>
using namespace mlx::core;
int main() {
    array a({1, 2, 3, 4}, {2, 2});
    array diag = diagonal(a);
    eval(diag);
    std::cout << "diag evaluated successfully.\n";
    return 0;
}
