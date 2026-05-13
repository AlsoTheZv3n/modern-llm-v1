#pragma once

#include "core/tensor.h"

namespace modernllm {

// Tile-based ("Flash") causal scaled-dot attention forward — O(N) memory,
// no T x T `probs` materialized.
//
// Same input/output shapes as scaled_dot_attention_forward, plus a per-row
// log-sum-exp tensor `L` saved for the eventual T6 backward.
//
// Inputs:
//   q, k, v      [B, T, D]   FP32 / CUDA
// Outputs:
//   ctx          [B, T, D]   FP32 / CUDA
//   L            [B, T]      FP32 / CUDA   (logsumexp per query row,
//                                            i.e. row_max + log(row_sum))
//
// Numerically equivalent to the naive scaled_dot_attention_forward (within
// the FP32 reordering tolerance from the online softmax).
//
// `tile_kv` is the K/V tile size; default 32 works for d_h up to a few
// hundred. Must be > 0 and ≤ T.
void flash_attn_forward(const Tensor& q, const Tensor& k, const Tensor& v,
                         Tensor& ctx, Tensor& L,
                         int B, int T, int D,
                         int tile_kv = 32);

// Tile-based ("Flash") causal scaled-dot attention backward — same memory
// profile as forward. Recomputes P[i,j] = exp(s[i,j] - L[i]) on-the-fly
// from L (logsumexp) cached by the forward pass. Two kernels:
//   1. dQ pass — block per (b, q_idx), iterates K/V tiles, accumulates dQ.
//   2. dKdV pass — block per (b, k_idx), iterates Q tiles, accumulates dK, dV.
//
// Inputs:
//   q, k, v   [B, T, D]   FP32 — same as forward inputs
//   ctx       [B, T, D]   FP32 — forward output (used to compute D[i] = O[i].dO[i])
//   L         [B, T]      FP32 — forward logsumexp cache
//   d_ctx     [B, T, D]   FP32 — incoming gradient on attention output
//
// Outputs (overwritten, not accumulated):
//   d_q, d_k, d_v  [B, T, D]  FP32
//
// `tile` is both the K/V tile in dQ and the Q tile in dKdV; must be > 0,
// is auto-clamped to T.
void flash_attn_backward(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& ctx, const Tensor& L,
                          const Tensor& d_ctx,
                          Tensor& d_q, Tensor& d_k, Tensor& d_v,
                          int B, int T, int D,
                          int tile = 16);

}  // namespace modernllm
