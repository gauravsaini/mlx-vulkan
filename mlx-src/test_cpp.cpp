#include <iostream>
#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  array a = matmul(ones({4, 4}), ones({4, 4}), StreamOrDevice(Device::gpu));
  eval(a);
  std::cout << "first: " << a.item<float>() << std::endl;
  return 0;
}
