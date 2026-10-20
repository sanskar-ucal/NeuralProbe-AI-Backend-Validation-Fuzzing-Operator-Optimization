// GEMM kernels.
//
// The optimized path follows the classic GotoBLAS/BLIS structure:
//   - B is packed into KC x NR column panels, A into MC x KC row panels (MR rows interleaved),
//     so the micro-kernel streams both operands contiguously from L1/L2.
//   - An 8x8 register-blocked micro-kernel keeps 16 float32x4 accumulators live and issues
//     FMA-by-lane (vfmaq_laneq_f32), i.e. 64 FMAs per 4 vector loads.
//   - Partial edge tiles are computed into a scratch tile and copied out, so the hot loop is branch-free.
//   - Multi-threading partitions M across threads (each thread packs its own panels).
//   - M <= 4 dispatches to a packing-free streaming kernel (see gemm_skinny).
#include <algorithm>
#include <cstring>
#include <vector>

#include "neuralprobe/kernels.h"
#include "parallel.h"

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(NP_FORCE_SCALAR)
#include <arm_neon.h>
#define NP_NEON 1
#else
#define NP_NEON 0
#endif

namespace {

constexpr int64_t MR = 8;
constexpr int64_t NR = 8;
constexpr int64_t KC = 256;   // A panel (MR x KC) + B panel (KC x NR) ~= 16 KB, fits L1
constexpr int64_t MC = 128;   // packed A block (MC x KC) = 128 KB, targets L2
constexpr int64_t NC = 2048;  // packed B block (KC x NC) = 2 MB

inline int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Packs B[0:kc, 0:nc] (row stride ldb) into ceil(nc/NR) panels laid out as [panel][k][NR].
void pack_b(const float* B, int64_t ldb, int64_t kc, int64_t nc, float* Bp) {
  for (int64_t j = 0; j < nc; j += NR) {
    const int64_t nr = std::min(NR, nc - j);
    float* dst = Bp + (j / NR) * kc * NR;
    for (int64_t k = 0; k < kc; ++k) {
      const float* src = B + k * ldb + j;
      if (nr == NR) {
        std::memcpy(dst + k * NR, src, NR * sizeof(float));
      } else {
        int64_t c = 0;
        for (; c < nr; ++c) dst[k * NR + c] = src[c];
        for (; c < NR; ++c) dst[k * NR + c] = 0.0f;
      }
    }
  }
}

// Packs A[0:mc, 0:kc] (row stride lda) into ceil(mc/MR) panels laid out as [panel][k][MR].
void pack_a(const float* A, int64_t lda, int64_t mc, int64_t kc, float* Ap) {
  for (int64_t i = 0; i < mc; i += MR) {
    const int64_t mr = std::min(MR, mc - i);
    float* dst = Ap + (i / MR) * kc * MR;
    for (int64_t k = 0; k < kc; ++k) {
      int64_t r = 0;
      for (; r < mr; ++r) dst[k * MR + r] = A[(i + r) * lda + k];
      for (; r < MR; ++r) dst[k * MR + r] = 0.0f;
    }
  }
}

// c[0:MR, 0:NR] (row stride ldc) = (accumulate ? c : 0) + a_panel * b_panel
#if NP_NEON
void micro_kernel_8x8(int64_t kc, const float* __restrict a, const float* __restrict b, float* __restrict c,
                      int64_t ldc, bool accumulate) {
  float32x4_t c0l = vdupq_n_f32(0.f), c0h = c0l, c1l = c0l, c1h = c0l, c2l = c0l, c2h = c0l, c3l = c0l,
              c3h = c0l, c4l = c0l, c4h = c0l, c5l = c0l, c5h = c0l, c6l = c0l, c6h = c0l, c7l = c0l, c7h = c0l;
  for (int64_t k = 0; k < kc; ++k) {
    const float32x4_t a0 = vld1q_f32(a);
    const float32x4_t a1 = vld1q_f32(a + 4);
    const float32x4_t b0 = vld1q_f32(b);
    const float32x4_t b1 = vld1q_f32(b + 4);
    c0l = vfmaq_laneq_f32(c0l, b0, a0, 0);
    c0h = vfmaq_laneq_f32(c0h, b1, a0, 0);
    c1l = vfmaq_laneq_f32(c1l, b0, a0, 1);
    c1h = vfmaq_laneq_f32(c1h, b1, a0, 1);
    c2l = vfmaq_laneq_f32(c2l, b0, a0, 2);
    c2h = vfmaq_laneq_f32(c2h, b1, a0, 2);
    c3l = vfmaq_laneq_f32(c3l, b0, a0, 3);
    c3h = vfmaq_laneq_f32(c3h, b1, a0, 3);
    c4l = vfmaq_laneq_f32(c4l, b0, a1, 0);
    c4h = vfmaq_laneq_f32(c4h, b1, a1, 0);
    c5l = vfmaq_laneq_f32(c5l, b0, a1, 1);
    c5h = vfmaq_laneq_f32(c5h, b1, a1, 1);
    c6l = vfmaq_laneq_f32(c6l, b0, a1, 2);
    c6h = vfmaq_laneq_f32(c6h, b1, a1, 2);
    c7l = vfmaq_laneq_f32(c7l, b0, a1, 3);
    c7h = vfmaq_laneq_f32(c7h, b1, a1, 3);
    a += MR;
    b += NR;
  }
#define NP_STORE_ROW(r, lo, hi)                                  \
  do {                                                           \
    float* row = c + (r) * ldc;                                  \
    if (accumulate) {                                            \
      vst1q_f32(row, vaddq_f32(vld1q_f32(row), lo));             \
      vst1q_f32(row + 4, vaddq_f32(vld1q_f32(row + 4), hi));     \
    } else {                                                     \
      vst1q_f32(row, lo);                                        \
      vst1q_f32(row + 4, hi);                                    \
    }                                                            \
  } while (0)
  NP_STORE_ROW(0, c0l, c0h);
  NP_STORE_ROW(1, c1l, c1h);
  NP_STORE_ROW(2, c2l, c2h);
  NP_STORE_ROW(3, c3l, c3h);
  NP_STORE_ROW(4, c4l, c4h);
  NP_STORE_ROW(5, c5l, c5h);
  NP_STORE_ROW(6, c6l, c6h);
  NP_STORE_ROW(7, c7l, c7h);
#undef NP_STORE_ROW
}
#else
void micro_kernel_8x8(int64_t kc, const float* __restrict a, const float* __restrict b, float* __restrict c,
                      int64_t ldc, bool accumulate) {
  float acc[MR][NR] = {};
  for (int64_t k = 0; k < kc; ++k) {
    for (int64_t r = 0; r < MR; ++r) {
      const float av = a[r];
      for (int64_t j = 0; j < NR; ++j) acc[r][j] += av * b[j];
    }
    a += MR;
    b += NR;
  }
  for (int64_t r = 0; r < MR; ++r) {
    float* row = c + r * ldc;
    for (int64_t j = 0; j < NR; ++j) row[j] = accumulate ? row[j] + acc[r][j] : acc[r][j];
  }
}
#endif

void gemm_serial(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K, int64_t lda,
                 int64_t ldb, int64_t ldc) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {
    for (int64_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    return;
  }
  thread_local std::vector<float> a_buf, b_buf;
  const int64_t mc_max = std::min(MC, round_up(M, MR));
  const int64_t nc_max = std::min(NC, round_up(N, NR));
  const int64_t kc_max = std::min(KC, K);
  a_buf.resize(static_cast<size_t>(mc_max * kc_max));
  b_buf.resize(static_cast<size_t>(kc_max * nc_max));
  float* Ap = a_buf.data();
  float* Bp = b_buf.data();
  float tile[MR * NR];

  for (int64_t jc = 0; jc < N; jc += NC) {
    const int64_t nc = std::min(NC, N - jc);
    for (int64_t pc = 0; pc < K; pc += KC) {
      const int64_t kc = std::min(KC, K - pc);
      const bool accumulate = pc > 0;
      pack_b(B + pc * ldb + jc, ldb, kc, nc, Bp);
      for (int64_t ic = 0; ic < M; ic += MC) {
        const int64_t mc = std::min(MC, M - ic);
        pack_a(A + ic * lda + pc, lda, mc, kc, Ap);
        for (int64_t jr = 0; jr < nc; jr += NR) {
          const int64_t nr = std::min(NR, nc - jr);
          const float* bp = Bp + (jr / NR) * kc * NR;
          for (int64_t ir = 0; ir < mc; ir += MR) {
            const int64_t mr = std::min(MR, mc - ir);
            const float* ap = Ap + (ir / MR) * kc * MR;
            float* c = C + (ic + ir) * ldc + jc + jr;
            if (mr == MR && nr == NR) {
              micro_kernel_8x8(kc, ap, bp, c, ldc, accumulate);
            } else {
              micro_kernel_8x8(kc, ap, bp, tile, NR, false);
              for (int64_t r = 0; r < mr; ++r)
                for (int64_t j = 0; j < nr; ++j)
                  c[r * ldc + j] = accumulate ? c[r * ldc + j] + tile[r * NR + j] : tile[r * NR + j];
            }
          }
        }
      }
    }
  }
}

// Skinny-M path (GEMV-like, e.g. LLM decode): packing A into MR=8 panels would waste most of the
// micro-kernel, so instead stream B rows directly and keep a 32-column strip of C in registers.
constexpr int64_t kSkinnyM = 4;
constexpr int64_t kStrip = 32;

void gemm_skinny(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K, int64_t n0,
                 int64_t n1) {
  for (int64_t i = 0; i < M; ++i) {
    const float* a = A + i * K;
    float* c = C + i * N;
    int64_t j = n0;
#if NP_NEON
    for (; j + kStrip <= n1; j += kStrip) {
      float32x4_t acc[kStrip / 4];
      for (auto& v : acc) v = vdupq_n_f32(0.f);
      for (int64_t k = 0; k < K; ++k) {
        const float* b = B + k * N + j;
        const float32x4_t av = vdupq_n_f32(a[k]);
        for (int64_t v = 0; v < kStrip / 4; ++v) acc[v] = vfmaq_f32(acc[v], vld1q_f32(b + 4 * v), av);
      }
      for (int64_t v = 0; v < kStrip / 4; ++v) vst1q_f32(c + j + 4 * v, acc[v]);
    }
#endif
    for (; j < n1; ++j) {
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += a[k] * B[k * N + j];
      c[j] = acc;
    }
  }
}

}  // namespace

extern "C" {

int np_has_neon(void) { return NP_NEON; }

const char* np_isa_name(void) { return NP_NEON ? "arm-neon" : "portable-scalar"; }

void np_matmul_f32_ref(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
}

void np_matmul_f32_opt(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K,
                       int num_threads) {
  if (M <= kSkinnyM) {
    neuralprobe::parallel_for(N, std::max(1, num_threads), kStrip,
                              [&](int64_t n0, int64_t n1) { gemm_skinny(A, B, C, M, N, K, n0, n1); });
    return;
  }
  neuralprobe::parallel_for(M, std::max(1, num_threads), MR, [&](int64_t m0, int64_t m1) {
    gemm_serial(A + m0 * K, B, C + m0 * N, m1 - m0, N, K, K, N, N);
  });
}

}  // extern "C"
