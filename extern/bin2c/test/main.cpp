#include <iostream>
#include <string>
#include "embeded.h"

int main() {
  // Here we show how we can decode a embeded file
  const auto n = target_size;
  const auto p = target;
  std::string a((char*)p);
  std::cout << p << std::endl;
  return 0;
};
