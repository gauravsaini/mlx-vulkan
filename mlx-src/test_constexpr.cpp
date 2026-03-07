#include <iostream>
#include "mlx/types/half_types.h"
#include "mlx/types/limits.h"

int main() {
    constexpr mlx::core::float16_t inf = mlx::core::numeric_limits<mlx::core::float16_t>::infinity();
    std::cout << "Compiled fine." << std::endl;
    return 0;
}
