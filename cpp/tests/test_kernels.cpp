#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "neuralprobe/kernels.h"

namespace {

std::vector<float> random_vec(size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// fp64 matmul plus the |A|*|B| magnitude used to derive a forward-error bound.
void matmul_f64(const std::vector<float>& A, const std::vector<float>& B, int64_t M, int64_t N, int64_t K,
                std::vector<double>& C, std::vector<double>& mag) {
  C.assign(static_cast<size_t>(M * N), 0.0);
  mag.assign(static_cast<size_t>(M * N), 0.0);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k)
      for (int64_t j = 0; j < N; ++j) {
        const double p = static_cast<double>(A[i * K + k]) * B[k * N + j];
        C[i * N + j] += p;
        mag[i * N + j] += std::fabs(p);
      }
}

void expect_within_gemm_bound(const std::vector<float>& got, const std::vector<double>& want,
                              const std::vector<double>& mag, int64_t K) {
  const double eps = std::numeric_limits<float>::epsilon();
  for (size_t i = 0; i < got.size(); ++i) {
    const double bound = 2.0 * (K + 1) * eps * mag[i] + 1e-30;
    ASSERT_LE(std::fabs(got[i] - want[i]), bound) << "index " << i << " got " << got[i] << " want " << want[i];
  }
}

using GemmParam = std::tuple<int64_t, int64_t, int64_t, int>;  // M, N, K, threads

class GemmTest : public ::testing::TestWithParam<GemmParam> {};

TEST_P(GemmTest, MatchesFp64Oracle) {
  const auto [M, N, K, threads] = GetParam();
  const auto A = random_vec(static_cast<size_t>(M * K), 1234 + static_cast<uint32_t>(M));
  const auto B = random_vec(static_cast<size_t>(K * N), 99 + static_cast<uint32_t>(N));
  std::vector<double> want, mag;
  matmul_f64(A, B, M, N, K, want, mag);

  std::vector<float> ref(static_cast<size_t>(M * N), -7.f), opt(static_cast<size_t>(M * N), -7.f);
  np_matmul_f32_ref(A.data(), B.data(), ref.data(), M, N, K);
  np_matmul_f32_opt(A.data(), B.data(), opt.data(), M, N, K, threads);
  expect_within_gemm_bound(ref, want, mag, K);
  expect_within_gemm_bound(opt, want, mag, K);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, GemmTest,
    ::testing::Values(GemmParam{1, 1, 1, 1}, GemmParam{1, 768, 768, 1}, GemmParam{4, 70, 33, 3},
                      GemmParam{2, 31, 9, 1}, GemmParam{5, 40, 40, 1}, GemmParam{7, 13, 5, 1},
                      GemmParam{8, 8, 8, 1}, GemmParam{9, 9, 9, 1}, GemmParam{33, 65, 17, 1},
                      GemmParam{64, 64, 300, 1},      // crosses one KC boundary
                      GemmParam{129, 200, 513, 1},    // crosses MC and multiple KC boundaries
                      GemmParam{257, 31, 64, 4},      // threads + ragged M
                      GemmParam{100, 2100, 16, 2},    // crosses NC boundary
                      GemmParam{3, 5, 1000, 8}));     // more threads than row tiles

TEST(Gemm, ZeroKProducesZeros) {
  std::vector<float> C(12, 5.f);
  np_matmul_f32_opt(nullptr, nullptr, C.data(), 3, 4, 0, 1);
  for (float v : C) EXPECT_EQ(v, 0.f);
}

TEST(Gemm, PropagatesNaNOnlyToAffectedRow) {
  const int64_t M = 9, N = 11, K = 17;
  auto A = random_vec(M * K, 1);
  const auto B = random_vec(K * N, 2);
  A[4 * K + 3] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> C(M * N);
  np_matmul_f32_opt(A.data(), B.data(), C.data(), M, N, K, 1);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) EXPECT_EQ(std::isnan(C[i * N + j]), i == 4) << i << "," << j;
}

TEST(Gemm, IdentityIsExact) {
  const int64_t n = 37;
  std::vector<float> I(n * n, 0.f);
  for (int64_t i = 0; i < n; ++i) I[i * n + i] = 1.f;
  const auto B = random_vec(n * n, 7);
  std::vector<float> C(n * n);
  np_matmul_f32_opt(I.data(), B.data(), C.data(), n, n, n, 1);
  for (int64_t i = 0; i < n * n; ++i) EXPECT_EQ(C[i], B[i]);
}

// N, C, H, W, KH, KW, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, use_bias, threads
using DwParam = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, bool, int>;

class DwConvTest : public ::testing::TestWithParam<DwParam> {};

TEST_P(DwConvTest, OptMatchesReference) {
  const auto [N, C, H, W, KH, KW, sh, sw, pt, pl, pb, pr, use_bias, threads] = GetParam();
  const int64_t OH = np_conv_out_dim(H, KH, sh, pt, pb);
  const int64_t OW = np_conv_out_dim(W, KW, sw, pl, pr);
  ASSERT_GT(OH, 0);
  ASSERT_GT(OW, 0);
  const auto x = random_vec(N * C * H * W, 11);
  const auto w = random_vec(C * KH * KW, 12);
  const auto b = random_vec(C, 13);
  const float* bias = use_bias ? b.data() : nullptr;
  std::vector<float> ref(N * C * OH * OW, -1.f), opt(N * C * OH * OW, -2.f);
  np_dwconv2d_f32_ref(x.data(), w.data(), bias, ref.data(), N, C, H, W, KH, KW, sh, sw, pt, pl, pb, pr);
  np_dwconv2d_f32_opt(x.data(), w.data(), bias, opt.data(), N, C, H, W, KH, KW, sh, sw, pt, pl, pb, pr,
                      threads);
  const float tol = 4.f * (KH * KW + 1) * std::numeric_limits<float>::epsilon();
  for (size_t i = 0; i < ref.size(); ++i) ASSERT_NEAR(opt[i], ref[i], tol * (1.f + std::fabs(ref[i]))) << i;
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, DwConvTest,
    ::testing::Values(DwParam{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, 1},
                      DwParam{1, 3, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true, 1},
                      DwParam{2, 5, 13, 17, 3, 3, 1, 1, 1, 1, 1, 1, true, 3},
                      DwParam{1, 4, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, false, 1},
                      DwParam{1, 4, 16, 33, 3, 3, 2, 2, 0, 0, 1, 1, true, 2},   // asymmetric (TF "SAME")
                      DwParam{1, 6, 20, 21, 5, 5, 1, 1, 2, 2, 2, 2, true, 1},
                      DwParam{1, 6, 20, 21, 5, 5, 2, 2, 2, 2, 2, 2, true, 1},
                      DwParam{1, 2, 11, 9, 7, 7, 1, 1, 3, 3, 3, 3, false, 1},   // generic path
                      DwParam{1, 2, 12, 12, 3, 1, 1, 1, 1, 0, 1, 0, true, 1},   // non-square
                      DwParam{1, 3, 19, 23, 3, 3, 3, 3, 1, 1, 1, 1, true, 1},   // stride 3 -> scalar
                      DwParam{1, 3, 10, 10, 3, 3, 1, 2, 0, 3, 2, 0, true, 1},   // mixed strides
                      DwParam{1, 32, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, true, 4}));

TEST(DwConv, OutDim) {
  EXPECT_EQ(np_conv_out_dim(5, 3, 1, 1, 1), 5);
  EXPECT_EQ(np_conv_out_dim(5, 3, 2, 1, 1), 3);
  EXPECT_EQ(np_conv_out_dim(2, 5, 1, 0, 0), 0);
}

}  // namespace
