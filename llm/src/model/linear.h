#pragma once

#include "core/gemm.h"
#include "core/scratch.h"
#include "core/tensor.h"

namespace modernllm {

// Linear layer: Y = X @ W + b
//
// Shapes (all FP32, all CUDA):
//   X  [N, In]    — input (caller may flatten [B, T, In] -> [B*T, In])
//   W  [In, Out]
//   b  [Out]      — optional (pass nullptr to disable)
//   Y  [N, Out]
//
// Backward (analytic):
//   dX = dY @ W^T   shape [N, In]
//   dW = X^T @ dY   shape [In, Out]
//   db = sum_n dY[n, :]  shape [Out]
void linear_forward(cublasHandle_t h,
                     const Tensor& X, const Tensor& W,
                     const Tensor* bias, Tensor& Y);

// d_W and d_b are accumulated into (caller zeros at top of step).
// d_X is overwritten.
void linear_backward(cublasHandle_t h,
                      const Tensor& X, const Tensor& W,
                      const Tensor& d_Y,
                      Tensor& d_X, Tensor& d_W,
                      Tensor* d_b);

// BF16-mixed-precision variants (Stage 3.A — slow path, kept for reference).
// Cast everything inline; allocates fresh scratch tensors via cudaMalloc per
// call. Use the `_arena` variants below for the hot path.
void linear_forward_bf16(cublasHandle_t h,
                          const Tensor& X, const Tensor& W,
                          const Tensor* bias, Tensor& Y);

void linear_backward_bf16(cublasHandle_t h,
                           const Tensor& X, const Tensor& W,
                           const Tensor& d_Y,
                           Tensor& d_X, Tensor& d_W,
                           Tensor* d_b);

// Stage T1 BF16 fast path. Caller pre-casts the weight once per opt-step
// (`W_bf16` is the persistent BF16 mirror of the FP32 master) and supplies a
// scratch arena from which X / dY casts are sub-allocated. No `cudaMalloc`
// happens in this code path.
//
// W_bf16 must already be a BF16 tensor with the same shape as W_fp32.
void linear_forward_bf16_arena(cublasHandle_t h,
                                const Tensor& X_fp32,
                                const Tensor& W_bf16,
                                const Tensor* bias_fp32,
                                Tensor& Y_fp32,
                                ScratchArena& arena);

void linear_backward_bf16_arena(cublasHandle_t h,
                                 const Tensor& X_fp32,
                                 const Tensor& W_bf16,
                                 const Tensor& d_Y_fp32,
                                 Tensor& d_X_fp32,
                                 Tensor& d_W_fp32,
                                 Tensor* d_b_fp32,
                                 ScratchArena& arena);

// T9.B — full-BF16 path: activations stay BF16 throughout, weights are the
// persistent BF16 mirror (no per-call X cast). Bias and FP32 master gradients
// are still FP32 (numerically sensitive accumulation).
void linear_forward_bf16inout(cublasHandle_t h,
                               const Tensor& X_bf16,
                               const Tensor& W_bf16,
                               const Tensor* bias_fp32,
                               Tensor& Y_bf16);

void linear_backward_bf16inout(cublasHandle_t h,
                                const Tensor& X_bf16,
                                const Tensor& W_bf16,
                                const Tensor& d_Y_bf16,
                                Tensor& d_X_bf16,
                                Tensor& d_W_fp32,
                                Tensor* d_b_fp32);

}  // namespace modernllm
