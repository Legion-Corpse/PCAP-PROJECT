#include "common/cli.h"

#include <sstream>
#include <stdexcept>

#include "common/dataset.h"

namespace ooc {

Args::Args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + a);
    a = a.substr(2);
    const auto eq = a.find('=');
    if (eq != std::string::npos) {
      values_[a.substr(0, eq)] = a.substr(eq + 1);
    } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      values_[a] = argv[++i];
    } else {
      values_[a] = "1";
    }
  }
}

std::string Args::str(const std::string& key, const std::string& fallback) const {
  used_[key] = true;
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : it->second;
}

double Args::num(const std::string& key, double fallback) const {
  const std::string s = str(key);
  return s.empty() ? fallback : std::stod(s);
}

long long Args::integer(const std::string& key, long long fallback) const {
  const std::string s = str(key);
  return s.empty() ? fallback : (long long)parseCount(s);
}

std::vector<std::string> Args::list(const std::string& key, const std::string& fallback) const {
  std::vector<std::string> out;
  std::stringstream ss(str(key, fallback));
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

std::vector<std::string> Args::unused() const {
  std::vector<std::string> out;
  for (const auto& kv : values_)
    if (!used_.count(kv.first)) out.push_back(kv.first);
  return out;
}

}  // namespace ooc
