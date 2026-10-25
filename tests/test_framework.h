// Tiny self-contained test harness (no external dependencies to fetch).
#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testing {

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TestCase {
  std::string group, name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* group, const char* name, std::function<void()> fn) {
    registry().push_back({group, name, std::move(fn)});
  }
};

template <class A, class B>
void checkEq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
  if (!(a == b)) {
    std::ostringstream os;
    os << file << ":" << line << ": CHECK_EQ(" << ea << ", " << eb << ") failed: " << a << " vs " << b;
    throw Failure(os.str());
  }
}

}  // namespace testing

#define TEST(group, name)                                                                   \
  static void test_##group##_##name();                                                      \
  static ::testing::Registrar registrar_##group##_##name(#group, #name, test_##group##_##name); \
  static void test_##group##_##name()

#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::ostringstream os_;                                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";                             \
      throw ::testing::Failure(os_.str());                                                           \
    }                                                                                                \
  } while (0)

#define CHECK_MSG(cond, msg)                                                                         \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::ostringstream os_;                                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed: " << msg;                    \
      throw ::testing::Failure(os_.str());                                                           \
    }                                                                                                \
  } while (0)

#define CHECK_EQ(a, b) ::testing::checkEq((a), (b), #a, #b, __FILE__, __LINE__)
