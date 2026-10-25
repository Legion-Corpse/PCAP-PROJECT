// Minimal "--key value" / "--flag" command-line parser.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ooc {

class Args {
 public:
  Args(int argc, char** argv);

  bool has(const std::string& key) const { return values_.count(key) > 0; }
  std::string str(const std::string& key, const std::string& fallback = "") const;
  double num(const std::string& key, double fallback) const;
  long long integer(const std::string& key, long long fallback) const;  // accepts K/M/G suffixes
  std::vector<std::string> list(const std::string& key, const std::string& fallback) const;  // comma separated

  // Keys that were given but never queried (typos).
  std::vector<std::string> unused() const;

 private:
  std::map<std::string, std::string> values_;
  mutable std::map<std::string, bool> used_;
};

}  // namespace ooc
