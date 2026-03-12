#include "mlx/mlx.h"
#include <iostream>

using namespace mlx::core;

int main() {
  Stream s = default_stream(Device::gpu);
  auto imported = import_function("fn.mlxfn");
  std::cout << "Imported successfully" << std::endl;
  auto out = imported({});
  std::cout << "Executed successfully" << std::endl;
  eval(out);
  std::cout << "Evaluated successfully" << std::endl;
  return 0;
}
