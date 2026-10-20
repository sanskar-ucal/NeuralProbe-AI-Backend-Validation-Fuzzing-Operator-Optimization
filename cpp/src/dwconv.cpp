// Depthwise 2D convolution (NCHW, groups == C, dilation 1).
//
// Optimized path:
//   - Each (n, c) plane is copied once into a zero-padded scratch buffer, which removes all
//     boundary checks from the inner loop (handles arbitrary asymmetric padding).
//   - Output is vectorized along W with several independent float32x4 accumulators (hides FMA latency).
//     Stride 1 computes two output rows per pass so every loaded input vector feeds two FMAs;
//     stride 2 uses vld2q de-interleaving loads to gather every other pixel.
//   - 3x3 and 5x5 filters are dispatched to compile-time specializations so the tap loops unroll.
//   - Planes are distributed across threads.
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

// Extra zeroed floats after every padded row; vld2q at the last vector may read one row element past Wp.
constexpr int64_t kRowSlack = 16;

struct DwShape {
  int64_t H, W, KH, KW, sh, sw, pt, pl, pb, pr, OH, OW;
};

template <int FKH, int FKW>
void dw_plane(const float* x, const float* w, float bias, float* y, const DwShape& s, std::vector<float>& buf) {
  const int64_t KH = FKH > 0 ? FKH : s.KH;
  const int64_t KW = FKW > 0 ? FKW : s.KW;
  const int64_t Hp = s.H + s.pt + s.pb;
  const int64_t Wp = s.W + s.pl + s.pr;
  const int64_t ld = Wp + kRowSlack;

  buf.assign(static_cast<size_t>(Hp * ld + kRowSlack), 0.0f);
  for (int64_t h = 0; h < s.H; ++h)
    std::memcpy(buf.data() + (h + s.pt) * ld + s.pl, x + h * s.W, static_cast<size_t>(s.W) * sizeof(float));
  const float* p = buf.data();

  int64_t oh = 0;
#if NP_NEON
  // Stride 1: compute two output rows per pass. Input row oh+r feeds output row oh (tap r) and oh+1 (tap r-1),
  // so each loaded vector is used twice: (KH+1)*KW loads per 2*KH*KW FMAs instead of one load per FMA.
  if (s.sh == 1 && s.sw == 1) {
    for (; oh + 2 <= s.OH; oh += 2) {
      float* y0 = y + oh * s.OW;
      float* y1 = y0 + s.OW;
      const float* base = p + oh * ld;
      int64_t ow = 0;
      for (; ow + 8 <= s.OW; ow += 8) {
        float32x4_t a00 = vdupq_n_f32(bias), a01 = a00, a10 = a00, a11 = a00;
        for (int64_t r = 0; r <= KH; ++r) {
          const float* in = base + r * ld + ow;
          for (int64_t kw = 0; kw < KW; ++kw) {
            const float32x4_t v0 = vld1q_f32(in + kw);
            const float32x4_t v1 = vld1q_f32(in + kw + 4);
            if (r < KH) {
              const float wv = w[r * KW + kw];
              a00 = vfmaq_n_f32(a00, v0, wv);
              a01 = vfmaq_n_f32(a01, v1, wv);
            }
            if (r > 0) {
              const float wv = w[(r - 1) * KW + kw];
              a10 = vfmaq_n_f32(a10, v0, wv);
              a11 = vfmaq_n_f32(a11, v1, wv);
            }
          }
        }
        vst1q_f32(y0 + ow, a00);
        vst1q_f32(y0 + ow + 4, a01);
        vst1q_f32(y1 + ow, a10);
        vst1q_f32(y1 + ow + 4, a11);
      }
      for (; ow + 4 <= s.OW; ow += 4) {
        float32x4_t a0 = vdupq_n_f32(bias), a1 = a0;
        for (int64_t r = 0; r <= KH; ++r) {
          const float* in = base + r * ld + ow;
          for (int64_t kw = 0; kw < KW; ++kw) {
            const float32x4_t v = vld1q_f32(in + kw);
            if (r < KH) a0 = vfmaq_n_f32(a0, v, w[r * KW + kw]);
            if (r > 0) a1 = vfmaq_n_f32(a1, v, w[(r - 1) * KW + kw]);
          }
        }
        vst1q_f32(y0 + ow, a0);
        vst1q_f32(y1 + ow, a1);
      }
      for (; ow < s.OW; ++ow) {
        float acc0 = bias, acc1 = bias;
        for (int64_t kh = 0; kh < KH; ++kh) {
          const float* r0 = base + kh * ld + ow;
          const float* wk = w + kh * KW;
          for (int64_t kw = 0; kw < KW; ++kw) {
            acc0 += r0[kw] * wk[kw];
            acc1 += r0[ld + kw] * wk[kw];
          }
        }
        y0[ow] = acc0;
        y1[ow] = acc1;
      }
    }
  }
#endif

  for (; oh < s.OH; ++oh) {
    float* yrow = y + oh * s.OW;
    const float* in_row0 = p + oh * s.sh * ld;
    int64_t ow = 0;
#if NP_NEON
    // 16 outputs per iteration in 4 independent accumulators: a single accumulator makes each output vector a
    // serial chain of KH*KW dependent FMAs, which is FMA-latency-bound rather than throughput-bound.
    if (s.sw == 1) {
      for (; ow + 16 <= s.OW; ow += 16) {
        float32x4_t a0 = vdupq_n_f32(bias), a1 = a0, a2 = a0, a3 = a0;
        for (int64_t kh = 0; kh < KH; ++kh) {
          const float* r = in_row0 + kh * ld + ow;
          const float* wk = w + kh * KW;
          for (int64_t kw = 0; kw < KW; ++kw) {
            const float wv = wk[kw];
            const float* q = r + kw;
            a0 = vfmaq_n_f32(a0, vld1q_f32(q), wv);
            a1 = vfmaq_n_f32(a1, vld1q_f32(q + 4), wv);
            a2 = vfmaq_n_f32(a2, vld1q_f32(q + 8), wv);
            a3 = vfmaq_n_f32(a3, vld1q_f32(q + 12), wv);
          }
        }
        vst1q_f32(yrow + ow, a0);
        vst1q_f32(yrow + ow + 4, a1);
        vst1q_f32(yrow + ow + 8, a2);
        vst1q_f32(yrow + ow + 12, a3);
      }
      for (; ow + 4 <= s.OW; ow += 4) {
        float32x4_t acc = vdupq_n_f32(bias);
        for (int64_t kh = 0; kh < KH; ++kh) {
          const float* r = in_row0 + kh * ld + ow;
          const float* wk = w + kh * KW;
          for (int64_t kw = 0; kw < KW; ++kw) acc = vfmaq_n_f32(acc, vld1q_f32(r + kw), wk[kw]);
        }
        vst1q_f32(yrow + ow, acc);
      }
    } else if (s.sw == 2) {
      for (; ow + 16 <= s.OW; ow += 16) {
        float32x4_t a0 = vdupq_n_f32(bias), a1 = a0, a2 = a0, a3 = a0;
        for (int64_t kh = 0; kh < KH; ++kh) {
          const float* r = in_row0 + kh * ld + ow * 2;
          const float* wk = w + kh * KW;
          for (int64_t kw = 0; kw < KW; ++kw) {
            const float wv = wk[kw];
            const float* q = r + kw;
            a0 = vfmaq_n_f32(a0, vld2q_f32(q).val[0], wv);
            a1 = vfmaq_n_f32(a1, vld2q_f32(q + 8).val[0], wv);
            a2 = vfmaq_n_f32(a2, vld2q_f32(q + 16).val[0], wv);
            a3 = vfmaq_n_f32(a3, vld2q_f32(q + 24).val[0], wv);
          }
        }
        vst1q_f32(yrow + ow, a0);
        vst1q_f32(yrow + ow + 4, a1);
        vst1q_f32(yrow + ow + 8, a2);
        vst1q_f32(yrow + ow + 12, a3);
      }
      for (; ow + 4 <= s.OW; ow += 4) {
        float32x4_t acc = vdupq_n_f32(bias);
        for (int64_t kh = 0; kh < KH; ++kh) {
          const float* r = in_row0 + kh * ld + ow * 2;
          const float* wk = w + kh * KW;
          for (int64_t kw = 0; kw < KW; ++kw) acc = vfmaq_n_f32(acc, vld2q_f32(r + kw).val[0], wk[kw]);
        }
        vst1q_f32(yrow + ow, acc);
      }
    }
#endif
    for (; ow < s.OW; ++ow) {
      float acc = bias;
      for (int64_t kh = 0; kh < KH; ++kh) {
        const float* r = in_row0 + kh * ld + ow * s.sw;
        const float* wk = w + kh * KW;
        for (int64_t kw = 0; kw < KW; ++kw) acc += r[kw] * wk[kw];
      }
      yrow[ow] = acc;
    }
  }
}

}  // namespace

extern "C" {

int64_t np_conv_out_dim(int64_t in, int64_t k, int64_t stride, int64_t pad_begin, int64_t pad_end) {
  if (stride <= 0 || k <= 0) return 0;
  const int64_t span = in + pad_begin + pad_end - k;
  if (span < 0) return 0;
  return span / stride + 1;
}

void np_dwconv2d_f32_ref(const float* x, const float* w, const float* bias, float* y, int64_t N, int64_t C,
                         int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride_h, int64_t stride_w,
                         int64_t pad_t, int64_t pad_l, int64_t pad_b, int64_t pad_r) {
  const int64_t OH = np_conv_out_dim(H, KH, stride_h, pad_t, pad_b);
  const int64_t OW = np_conv_out_dim(W, KW, stride_w, pad_l, pad_r);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float* xp = x + (n * C + c) * H * W;
      const float* wp = w + c * KH * KW;
      float* yp = y + (n * C + c) * OH * OW;
      for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
          float acc = bias ? bias[c] : 0.0f;
          for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * stride_h + kh - pad_t;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < KW; ++kw) {
              const int64_t iw = ow * stride_w + kw - pad_l;
              if (iw < 0 || iw >= W) continue;
              acc += xp[ih * W + iw] * wp[kh * KW + kw];
            }
          }
          yp[oh * OW + ow] = acc;
        }
      }
    }
  }
}

void np_dwconv2d_f32_opt(const float* x, const float* w, const float* bias, float* y, int64_t N, int64_t C,
                         int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride_h, int64_t stride_w,
                         int64_t pad_t, int64_t pad_l, int64_t pad_b, int64_t pad_r, int num_threads) {
  DwShape s{H, W, KH, KW, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, 0, 0};
  s.OH = np_conv_out_dim(H, KH, stride_h, pad_t, pad_b);
  s.OW = np_conv_out_dim(W, KW, stride_w, pad_l, pad_r);
  if (s.OH <= 0 || s.OW <= 0) return;

  auto plane_fn = [&]() {
    if (KH == 3 && KW == 3) return &dw_plane<3, 3>;
    if (KH == 5 && KW == 5) return &dw_plane<5, 5>;
    return &dw_plane<0, 0>;
  }();

  neuralprobe::parallel_for(N * C, std::max(1, num_threads), 1, [&](int64_t p0, int64_t p1) {
    std::vector<float> buf;
    for (int64_t plane = p0; plane < p1; ++plane) {
      const int64_t c = plane % C;
      plane_fn(x + plane * H * W, w + c * KH * KW, bias ? bias[c] : 0.0f, y + plane * s.OH * s.OW, s, buf);
    }
  });
}

}  // extern "C"
