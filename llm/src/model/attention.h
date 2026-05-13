#pragma once

#include "core/gemm.h"
#include "core/tensor.h"

namespace modernllm {

// Single-head scaled-dot causal self-attention (no Q/K/V projection — see
// Linear layer for that). Operates on already-projected Q, K, V.
//
// Forward:
//   scores [B, T, T] = q [B, T, D] @ k^T [B, D, T] * (1 / sqrt(D))
//   mask:  scores[b, t1, t2] = -inf  for t2 > t1
//   probs[b, t1, t2] = softmax_last(scores[b, t1, :])
//   ctx[b, t, :] = probs[b, t, :] @ v[b, :, :]
//
// Caller pre-allocates probs (used as bwd cache) and ctx.
//
// Shapes (FP32 / CUDA):
//   q, k, v, ctx  [B, T, D]   (last two dims contiguous, leading dim T*D)
//   probs         [B, T, T]
void scaled_dot_attention_forward(cublasHandle_t handle,
                                   const Tensor& q, const Tensor& k,
                                   const Tensor& v,
                                   Tensor& ctx, Tensor& probs,
                                   int B, int T, int D);

// Backward. d_q, d_k, d_v are overwritten.
// `probs` must be the cache returned by forward.
void scaled_dot_attention_backward(cublasHandle_t handle,
                                    const Tensor& d_ctx,
                                    const Tensor& q, const Tensor& k,
                                    const Tensor& v, const Tensor& probs,
                                    Tensor& d_q, Tensor& d_k, Tensor& d_v,
                                    int B, int T, int D);

// ---------------------------------------------------------------------------
// Multi-head support (memory layout helpers)
// ---------------------------------------------------------------------------
// "Split heads": permute [B, T, H*d_h] -> [B*H, T, d_h].
//   in[b, t, h*d_h + k]  =>  out[b*H + h, t, k]
// Used to apply single-head scaled-dot attention to each head independently.
void split_heads(const Tensor& in, Tensor& out, int B, int T, int H, int d_h);

// "Merge heads": inverse of split_heads. [B*H, T, d_h] -> [B, T, H*d_h].
void merge_heads(const Tensor& in, Tensor& out, int B, int T, int H, int d_h);

// GQA support: replicate each of `n_kv` KV heads `group_size = n_q / n_kv`
// times along the head dim, so the existing scaled-dot-attention runs with
// n_q effective heads.
//   in  [B*n_kv, T, d_h]
//   out [B*n_q,  T, d_h]   (n_q must be a multiple of n_kv)
void repeat_kv_heads(const Tensor& in, Tensor& out,
                      int B, int n_kv, int n_q, int T, int d_h);

// Backward of repeat_kv_heads: sum the n_q-wide gradient back into the
// n_kv-wide buffer via atomicAdd. Caller must zero `out` first.
//   in  [B*n_q,  T, d_h]
//   out [B*n_kv, T, d_h]
void accumulate_kv_grads(const Tensor& in, Tensor& out,
                          int B, int n_kv, int n_q, int T, int d_h);

}  // namespace modernllm
