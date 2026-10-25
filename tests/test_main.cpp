#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "test_framework.h"

int main(int argc, char** argv) {
  std::string filter;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--filter") == 0) filter = argv[i + 1];
  int passed = 0, failed = 0;
  for (const auto& t : testing::registry()) {
    const std::string full = t.group + "." + t.name;
    if (!filter.empty() && t.group != filter && full != filter) continue;
    const auto start = std::chrono::steady_clock::now();
    try {
      t.fn();
      ++passed;
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      std::printf("[ PASS ] %-45s %8.1f ms\n", full.c_str(), ms);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[ FAIL ] %s\n         %s\n", full.c_str(), e.what());
    }
    std::fflush(stdout);
  }
  std::printf("%d passed, %d failed\n", passed, failed);
  return failed ? 1 : 0;
}
