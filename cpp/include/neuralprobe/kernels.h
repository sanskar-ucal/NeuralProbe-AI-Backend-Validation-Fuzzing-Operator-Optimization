// NeuralProbe native kernels: plain-C ABI so they can be loaded from Python via ctypes
// and fuzzed alongside ONNX Runtime / PyTorch.
//
// All tensors are dense, row-major (matmul) or NCHW (depthwise conv), float32.
#ifndef NEURALPROBE_KERNELS_H_
#define NEURALPROBE_KERNELS_H_

#include <stdint.h>

#if defined(_WIN32)
#define NP_API __declspec(dllexport)
#else
#define NP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 1 if the optimized kernels were compiled with ARM NEON intrinsics, 0 for the portable fallback.
NP_API int np_has_neon(void);
NP_API const char* np_isa_name(void);

// C[M,N] = A[M,K] * B[K,N]
NP_API void np_matmul_f32_ref(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
NP_API void np_matmul_f32_opt(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K,
                              int num_threads);

// Output spatial size for one dimension of a (dilation=1) convolution. Returns <= 0 if invalid.
NP_API int64_t np_conv_out_dim(int64_t in, int64_t k, int64_t stride, int64_t pad_begin, int64_t pad_end);

// Depthwise conv (groups == C, one filter per channel).
//   x: [N, C, H, W], w: [C, 1, KH, KW], bias: [C] or NULL, y: [N, C, OH, OW]
NP_API void np_dwconv2d_f32_ref(const float* x, const float* w, const float* bias, float* y, int64_t N,
                                int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride_h,
                                int64_t stride_w, int64_t pad_t, int64_t pad_l, int64_t pad_b, int64_t pad_r);
NP_API void np_dwconv2d_f32_opt(const float* x, const float* w, const float* bias, float* y, int64_t N,
                                int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride_h,
                                int64_t stride_w, int64_t pad_t, int64_t pad_l, int64_t pad_b, int64_t pad_r,
                                int num_threads);

#ifdef __cplusplus
}
#endif

#endif  // NEURALPROBE_KERNELS_H_
