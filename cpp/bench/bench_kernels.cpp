// Standalone micro-benchmark for the native kernels: naive reference vs optimized (NEON) path.
// Usage: bench_kernels [--json out.json] [--threads N] [--quick]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "neuralprobe/kernels.h"

namespace {

using Clock = std::chrono::steady_clock;

double median_seconds(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> t(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    fn();
    t[static_cast<size_t>(i)] = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  std::nth_element(t.begin(), t.begin() + iters / 2, t.end());
  return t[static_cast<size_t>(iters / 2)];
}

std::vector<float> rand_vec(size_t n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

struct Row {
  std::string name;
  double flops, ref_s, opt_s;
};

}  // namespace

int main(int argc, char** argv) {
  const char* json_path = nullptr;
  int threads = 1;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--quick")) quick = true;
  }
  const int iters = quick ? 5 : 20;
  std::vector<Row> rows;

  struct G { int64_t M, N, K; };
  std::vector<G> gemms = {{64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {1, 768, 768}, {128, 768, 768}};
  if (!quick) gemms.push_back({512, 512, 512});
  for (const auto& g : gemms) {
    auto A = rand_vec(g.M * g.K), B = rand_vec(g.K * g.N);
    std::vector<float> C(g.M * g.N);
    const double ref = median_seconds([&] { np_matmul_f32_ref(A.data(), B.data(), C.data(), g.M, g.N, g.K); },
                                      1, std::max(3, iters / 4));
    const double opt = median_seconds(
        [&] { np_matmul_f32_opt(A.data(), B.data(), C.data(), g.M, g.N, g.K, threads); }, 3, iters);
    char name[96];
    std::snprintf(name, sizeof(name), "matmul_%lldx%lldx%lld", (long long)g.M, (long long)g.N, (long long)g.K);
    rows.push_back({name, 2.0 * g.M * g.N * g.K, ref, opt});
  }

  struct D { int64_t C, H, W, K, S; };
  const std::vector<D> dws = {{32, 112, 112, 3, 1}, {144, 56, 56, 3, 2}, {192, 28, 28, 3, 1},
                              {576, 14, 14, 3, 1},  {960, 7, 7, 5, 1}};
  for (const auto& d : dws) {
    const int64_t p = d.K / 2;
    const int64_t OH = np_conv_out_dim(d.H, d.K, d.S, p, p), OW = np_conv_out_dim(d.W, d.K, d.S, p, p);
    auto x = rand_vec(d.C * d.H * d.W), w = rand_vec(d.C * d.K * d.K), b = rand_vec(d.C);
    std::vector<float> y(d.C * OH * OW);
    auto run_ref = [&] {
      np_dwconv2d_f32_ref(x.data(), w.data(), b.data(), y.data(), 1, d.C, d.H, d.W, d.K, d.K, d.S, d.S, p, p, p, p);
    };
    auto run_opt = [&] {
      np_dwconv2d_f32_opt(x.data(), w.data(), b.data(), y.data(), 1, d.C, d.H, d.W, d.K, d.K, d.S, d.S, p, p, p,
                          p, threads);
    };
    char name[96];
    std::snprintf(name, sizeof(name), "dwconv_c%lld_%lldx%lld_k%lld_s%lld", (long long)d.C, (long long)d.H,
                  (long long)d.W, (long long)d.K, (long long)d.S);
    rows.push_back({name, 2.0 * d.C * OH * OW * d.K * d.K, median_seconds(run_ref, 1, iters),
                    median_seconds(run_opt, 2, iters)});
  }

  std::printf("ISA: %s, threads: %d\n", np_isa_name(), threads);
  std::printf("%-34s %12s %12s %10s %10s %8s\n", "kernel", "ref_us", "opt_us", "ref_GF/s", "opt_GF/s", "speedup");
  for (const auto& r : rows)
    std::printf("%-34s %12.1f %12.1f %10.2f %10.2f %7.2fx\n", r.name.c_str(), r.ref_s * 1e6, r.opt_s * 1e6,
                r.flops / r.ref_s * 1e-9, r.flops / r.opt_s * 1e-9, r.ref_s / r.opt_s);

  if (json_path) {
    FILE* f = std::fopen(json_path, "w");
    if (!f) return 1;
    std::fprintf(f, "{\"isa\": \"%s\", \"threads\": %d, \"results\": [\n", np_isa_name(), threads);
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& r = rows[i];
      std::fprintf(f, "  {\"name\": \"%s\", \"ref_us\": %.3f, \"opt_us\": %.3f, \"opt_gflops\": %.3f, \"speedup\": %.3f}%s\n",
                   r.name.c_str(), r.ref_s * 1e6, r.opt_s * 1e6, r.flops / r.opt_s * 1e-9, r.ref_s / r.opt_s,
                   i + 1 < rows.size() ? "," : "");
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
  }
  return 0;
}
