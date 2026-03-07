#include "mlx/mlx.h"

#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace mlx::core;

namespace {

using ScanFn = std::function<array(const array&, int, bool, bool)>;

bool check_equal(const array& x, const array& y, const std::string& label) {
  bool ok = array_equal(x, y).item<bool>();
  if (!ok) {
    std::cerr << "FAIL: " << label << "\n";
  }
  return ok;
}

} // namespace

int main() {
  set_default_device(Device::cpu);

  std::vector<std::pair<std::string, ScanFn>> ops = {
      {"cumsum",
       [](const array& a, int axis, bool reverse, bool inclusive) {
         return cumsum(a, axis, reverse, inclusive);
       }},
      {"cumprod",
       [](const array& a, int axis, bool reverse, bool inclusive) {
         return cumprod(a, axis, reverse, inclusive);
       }},
      {"cummax",
       [](const array& a, int axis, bool reverse, bool inclusive) {
         return cummax(a, axis, reverse, inclusive);
       }},
      {"cummin",
       [](const array& a, int axis, bool reverse, bool inclusive) {
         return cummin(a, axis, reverse, inclusive);
       }},
  };

  auto rev_idx = arange(31, -1, -1, int32);

  constexpr int kIters = 50;
  for (int iter = 0; iter < kIters; ++iter) {
    auto a = random::randint(-100, 100, {32, 32, 32}, int32);
    auto shape = a.shape();

    for (const auto& [name, op] : ops) {
      for (int axis = 0; axis < 3; ++axis) {
        auto c_inclusive = op(a, axis, false, true);
        auto c_exclusive = op(a, axis, false, false);

        Shape c1_start{0, 0, 0};
        Shape c1_stop = shape;
        c1_stop[axis] -= 1;
        Shape c2_start{0, 0, 0};
        c2_start[axis] = 1;
        Shape c2_stop = shape;
        if (!check_equal(
                slice(c_inclusive, c1_start, c1_stop),
                slice(c_exclusive, c2_start, c2_stop),
                name + " forward exclusive axis=" + std::to_string(axis))) {
          return 1;
        }

        auto rev = take(a, rev_idx, axis);
        auto c1 = take(op(rev, axis, false, true), rev_idx, axis);
        auto c2 = op(a, axis, true, true);
        if (!check_equal(
                c1, c2, name + " reverse inclusive axis=" + std::to_string(axis))) {
          return 1;
        }

        auto c1_excl = c1;
        Shape c1e_start{0, 0, 0};
        Shape c1e_stop = shape;
        c1e_start[axis] = 1;
        Shape c2e_start{0, 0, 0};
        Shape c2e_stop = shape;
        c2e_stop[axis] -= 1;

        auto c2_excl = op(a, axis, true, false);
        if (!check_equal(
                slice(c1_excl, c1e_start, c1e_stop),
                slice(c2_excl, c2e_start, c2e_stop),
                name + " reverse exclusive axis=" + std::to_string(axis))) {
          return 1;
        }
      }
    }
  }

  std::cout << "scan regression checks passed.\n";
  return 0;
}
