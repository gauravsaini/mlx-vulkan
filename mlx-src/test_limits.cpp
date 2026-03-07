#include <iostream>
#include "mlx/types/half_types.h"
#include "mlx/types/limits.h"

int main() {
    mlx::core::float16_t inf = mlx::core::numeric_limits<mlx::core::float16_t>::infinity();
    mlx::core::float16_t zero = mlx::core::float16_t(0.0f);
    
    std::cout << "inf bytes: " << std::hex << ((uint16_t*)&inf)[0] << std::endl;
    std::cout << "zero bytes: " << std::hex << ((uint16_t*)&zero)[0] << std::endl;
    std::cout << "inf == zero: " << (inf == zero) << std::endl;
    return 0;
}
