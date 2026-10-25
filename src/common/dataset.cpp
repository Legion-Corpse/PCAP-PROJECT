#include "common/dataset.h"

#include "common/host_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ooc {
namespace {

constexpr double kPi = 3.14159265358979323846;

uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// splitmix64: tiny, fast and good enough for synthetic data. Every 64K-point
// chunk gets its own stream, so output is identical for any thread count. The
// start state is hashed: splitmix streams seeded with nearby states are the
// same sequence shifted by a few draws, which would correlate the chunks.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(mix64(seed)) {}
  uint64_t next() { return mix64(s += 0x9e3779b97f4a7c15ULL); }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
  }
};

constexpr int64_t kChunk = 1 << 16;

template <class Fn>
void generateChunked(std::vector<float3>& out, int64_t n, uint64_t seed, Fn&& fn) {
  out.resize(size_t(n));
  parallelFor(n, kChunk, [&](int64_t begin, int64_t end) {
    Rng rng(mix64(seed) ^ mix64(uint64_t(begin / kChunk) + 0x51ED270B27u));
    for (int64_t i = begin; i < end; ++i) out[size_t(i)] = fn(rng);
  });
}

float clamp01(double v) { return float(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v)); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) parts.push_back(item);
  return parts;
}

std::string extensionOf(const std::string& path) {
  const auto dot = path.find_last_of('.');
  return dot == std::string::npos ? std::string() : lower(path.substr(dot + 1));
}

PointSet uniformCloud(int64_t n, uint64_t seed) {
  PointSet ps;
  ps.name = "uniform";
  generateChunked(ps.points, n, seed, [](Rng& r) {
    return make_f3(float(r.uniform()), float(r.uniform()), float(r.uniform()));
  });
  return ps;
}

PointSet clusteredCloud(int64_t n, int clusters, uint64_t seed) {
  struct Cluster { double cx, cy, cz, sigma, cdf; };
  std::vector<Cluster> cs(size_t(std::max(1, clusters)));
  Rng setup(seed ^ 0xC1u);
  double total = 0.0;
  for (size_t i = 0; i < cs.size(); ++i) {
    auto& c = cs[i];
    c.cx = 0.1 + 0.8 * setup.uniform();
    c.cy = 0.1 + 0.8 * setup.uniform();
    c.cz = 0.1 + 0.8 * setup.uniform();
    c.sigma = 0.005 * std::pow(10.0, setup.uniform());  // log-uniform in [0.005, 0.05]
    total += 1.0 / double(i + 1);                        // Zipf-like cluster weights
    c.cdf = total;
  }
  for (auto& c : cs) c.cdf /= total;
  PointSet ps;
  ps.name = "clustered";
  generateChunked(ps.points, n, seed, [&](Rng& r) {
    if (r.uniform() < 0.05) return make_f3(float(r.uniform()), float(r.uniform()), float(r.uniform()));
    const double u = r.uniform();
    const auto it = std::lower_bound(cs.begin(), cs.end(), u, [](const Cluster& c, double v) { return c.cdf < v; });
    const Cluster& c = it == cs.end() ? cs.back() : *it;
    return make_f3(clamp01(c.cx + c.sigma * r.normal()), clamp01(c.cy + c.sigma * r.normal()),
                   clamp01(c.cz + c.sigma * r.normal()));
  });
  return ps;
}

// Points concentrated on 2D surfaces, the typical shape of LiDAR / photogrammetry
// data: density per volume is very uneven and most of space is empty.
PointSet sceneCloud(int64_t n, uint64_t seed) {
  struct Box { double x, y, w, d, h; };
  struct Sphere { double x, y, z, r; };
  Rng setup(seed ^ 0x5Cu);
  std::vector<Box> boxes(24);
  for (auto& b : boxes) {
    b.w = 0.04 + 0.08 * setup.uniform();
    b.d = 0.04 + 0.08 * setup.uniform();
    b.h = 0.05 + 0.35 * setup.uniform();
    b.x = setup.uniform() * (1.0 - b.w);
    b.y = setup.uniform() * (1.0 - b.d);
  }
  std::vector<Sphere> spheres(8);
  for (auto& s : spheres) {
    s.r = 0.02 + 0.05 * setup.uniform();
    s.x = s.r + setup.uniform() * (1.0 - 2 * s.r);
    s.y = s.r + setup.uniform() * (1.0 - 2 * s.r);
    s.z = s.r + 0.3 * setup.uniform();
  }
  PointSet ps;
  ps.name = "scene";
  generateChunked(ps.points, n, seed, [&](Rng& r) {
    const double kind = r.uniform();
    const double jitter = 2e-4;
    double x, y, z;
    if (kind < 0.40) {  // ground
      x = r.uniform(); y = r.uniform(); z = 0.0;
    } else if (kind < 0.85) {  // building facades and roofs
      const Box& b = boxes[size_t(r.next() % boxes.size())];
      const int face = int(r.next() % 5);
      const double u = r.uniform(), v = r.uniform();
      switch (face) {
        case 0: x = b.x + u * b.w; y = b.y; z = v * b.h; break;
        case 1: x = b.x + u * b.w; y = b.y + b.d; z = v * b.h; break;
        case 2: x = b.x; y = b.y + u * b.d; z = v * b.h; break;
        case 3: x = b.x + b.w; y = b.y + u * b.d; z = v * b.h; break;
        default: x = b.x + u * b.w; y = b.y + v * b.d; z = b.h; break;
      }
    } else {  // spheres (trees, domes)
      const Sphere& s = spheres[size_t(r.next() % spheres.size())];
      const double zz = 2.0 * r.uniform() - 1.0, t = 2.0 * kPi * r.uniform();
      const double rr = std::sqrt(std::max(0.0, 1.0 - zz * zz));
      x = s.x + s.r * rr * std::cos(t); y = s.y + s.r * rr * std::sin(t); z = s.z + s.r * zz;
    }
    return make_f3(clamp01(x + jitter * r.normal()), clamp01(y + jitter * r.normal()),
                   clamp01(z + jitter * r.normal()));
  });
  return ps;
}

PointSet loadPly(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line, format;
  size_t vertexCount = 0;
  bool inVertex = false;
  struct Prop { std::string type, name; };
  std::vector<Prop> props;
  std::getline(in, line);
  if (line.rfind("ply", 0) != 0) throw std::runtime_error(path + ": not a PLY file");
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ls(line);
    std::string tok;
    ls >> tok;
    if (tok == "format") ls >> format;
    else if (tok == "element") {
      std::string what;
      ls >> what;
      inVertex = what == "vertex";
      if (inVertex) ls >> vertexCount;
    } else if (tok == "property" && inVertex) {
      Prop p;
      ls >> p.type;
      if (p.type == "list") throw std::runtime_error(path + ": list properties on vertices are unsupported");
      ls >> p.name;
      props.push_back(p);
    } else if (tok == "end_header") break;
  }
  auto sizeOf = [](const std::string& t) -> size_t {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    throw std::runtime_error("unknown PLY type " + t);
  };
  int ix = -1, iy = -1, iz = -1;
  for (int i = 0; i < int(props.size()); ++i) {
    if (props[size_t(i)].name == "x") ix = i;
    if (props[size_t(i)].name == "y") iy = i;
    if (props[size_t(i)].name == "z") iz = i;
  }
  if (ix < 0 || iy < 0 || iz < 0) throw std::runtime_error(path + ": vertex x/y/z not found");
  PointSet ps;
  ps.name = path;
  ps.points.resize(vertexCount);
  if (format == "ascii") {
    for (size_t v = 0; v < vertexCount; ++v) {
      std::vector<double> vals(props.size());
      for (auto& d : vals) in >> d;
      ps.points[v] = make_f3(float(vals[size_t(ix)]), float(vals[size_t(iy)]), float(vals[size_t(iz)]));
    }
  } else if (format == "binary_little_endian") {
    std::vector<size_t> offs(props.size());
    size_t stride = 0;
    for (size_t i = 0; i < props.size(); ++i) { offs[i] = stride; stride += sizeOf(props[i].type); }
    std::vector<char> buf(stride * vertexCount);
    in.read(buf.data(), std::streamsize(buf.size()));
    auto read = [&](size_t v, int i) -> float {
      const char* p = buf.data() + v * stride + offs[size_t(i)];
      if (props[size_t(i)].type == "double" || props[size_t(i)].type == "float64") {
        double d; std::memcpy(&d, p, 8); return float(d);
      }
      float f; std::memcpy(&f, p, 4); return f;
    };
    for (size_t v = 0; v < vertexCount; ++v) ps.points[v] = make_f3(read(v, ix), read(v, iy), read(v, iz));
  } else {
    throw std::runtime_error(path + ": unsupported PLY format " + format);
  }
  return ps;
}

PointSet loadRawFloats(const std::string& path, int stride) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const auto bytes = size_t(in.tellg());
  in.seekg(0);
  std::vector<float> raw(bytes / sizeof(float));
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size() * sizeof(float)));
  PointSet ps;
  ps.name = path;
  const size_t n = raw.size() / size_t(stride);
  ps.points.resize(n);
  for (size_t i = 0; i < n; ++i) ps.points[i] = make_f3(raw[i * stride], raw[i * stride + 1], raw[i * stride + 2]);
  return ps;
}

PointSet loadText(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  PointSet ps;
  ps.name = path;
  std::string line;
  while (std::getline(in, line)) {
    for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
    std::istringstream ls(line);
    float x, y, z;
    if (ls >> x >> y >> z) ps.points.push_back(make_f3(x, y, z));
  }
  return ps;
}

}  // namespace

Aabb PointSet::bounds() const { return computeBounds(points); }

Aabb computeBounds(const std::vector<float3>& points) {
  const int64_t n = int64_t(points.size());
  constexpr int64_t grain = 1 << 16;
  std::vector<Aabb> partial(static_cast<size_t>((n + grain - 1) / grain));
  parallelFor(n, grain, [&](int64_t begin, int64_t end) {
    Aabb local;
    for (int64_t i = begin; i < end; ++i) local.extend(points[size_t(i)]);
    partial[size_t(begin / grain)] = local;
  });
  Aabb box;
  for (const auto& b : partial) if (b.valid()) box.extend(b);
  return box;
}

uint64_t parseCount(const std::string& text) {
  if (text.empty()) throw std::invalid_argument("empty count");
  double mult = 1.0;
  std::string num = text;
  switch (std::toupper(static_cast<unsigned char>(text.back()))) {
    case 'K': mult = 1e3; num.pop_back(); break;
    case 'M': mult = 1e6; num.pop_back(); break;
    case 'G': mult = 1e9; num.pop_back(); break;
    default: break;
  }
  return uint64_t(std::llround(std::stod(num) * mult));
}

PointSet makeDataset(const std::string& spec, uint64_t seed) {
  const auto colon = spec.find(':');
  const std::string kind = lower(spec.substr(0, colon));
  const std::string rest = colon == std::string::npos ? std::string() : spec.substr(colon + 1);
  if (kind == "file") return loadPointFile(rest);
  const auto args = split(rest, ':');
  if (args.empty()) throw std::invalid_argument("dataset spec needs a size: " + spec);
  const auto n = int64_t(parseCount(args[0]));
  PointSet ps;
  if (kind == "uniform") ps = uniformCloud(n, seed);
  else if (kind == "clustered") ps = clusteredCloud(n, args.size() > 1 ? std::stoi(args[1]) : 64, seed);
  else if (kind == "scene") ps = sceneCloud(n, seed);
  else throw std::invalid_argument("unknown dataset kind: " + kind);
  ps.name = spec;
  return ps;
}

PointSet loadPointFile(const std::string& path) {
  const std::string ext = extensionOf(path);
  if (ext == "ply") return loadPly(path);
  if (ext == "bin") return loadRawFloats(path, 3);
  if (ext == "kitti") return loadRawFloats(path, 4);
  if (ext == "xyz" || ext == "txt" || ext == "csv" || ext == "pts") return loadText(path);
  throw std::invalid_argument("unsupported point file extension: ." + ext);
}

void savePointsBin(const PointSet& ps, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(reinterpret_cast<const char*>(ps.points.data()), std::streamsize(ps.points.size() * sizeof(float3)));
}

PointSet makeQueries(const std::string& spec, const PointSet& points, uint64_t seed) {
  if (spec.empty() || lower(spec) == "same") {
    PointSet q;
    q.name = "same";
    q.points = points.points;
    return q;
  }
  return makeDataset(spec, seed + 7919);
}

float radiusForExpectedNeighbors(const Aabb& box, size_t n, double k) {
  const float3 e = box.extent();
  const double vol = std::max(1e-30, double(e.x) * double(e.y) * double(e.z));
  return float(std::cbrt(3.0 * k * vol / (4.0 * kPi * double(n))));
}

float conservativeRadius(float r, const Aabb& box) {
  float maxAbs = 0.f;
  if (box.valid()) {
    for (float v : {box.lo.x, box.lo.y, box.lo.z, box.hi.x, box.hi.y, box.hi.z}) maxAbs = std::max(maxAbs, std::fabs(v));
  }
  return r * (1.0f + 1e-5f) + 4.0f * FLT_EPSILON * maxAbs;
}

}  // namespace ooc
