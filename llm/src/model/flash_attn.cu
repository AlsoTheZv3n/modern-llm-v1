#include "model/flash_attn.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// One block per (b, q_idx). Each block streams over K/V tiles of `Bc` rows,
// keeping a per-row online softmax (running max + sum + un-normalized output).
// Causal mask: K rows beyond q_idx are simply not loaded.
//
// Block layout: blockDim.x threads cooperate on the D-dim work. We require
// blockDim.x to be a power of two so the per-tile dot-product reduction
// (Q . K[k_local]) can use the simple shared-memory tree reduction.
//
// Shared memory layout (sized by host launch):
//     s_q          [D]               Q row
//     s_k          [Bc * D]          K tile rows (column-major within row)
//     s_v          [Bc * D]          V tile rows
//     s_o          [D]               un-normalized output accumulator
//     s_scores     [Bc]              per-tile scaled dot products
//     s_reduce     [blockDim.x]      reduction scratch
__global__ void flash_attn_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      float* __restrict__ L,
                                      int T, int D, float scale, int Bc) {
    int b = blockIdx.y;
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    if (q_idx >= T) return;

    extern __shared__ float smem[];
    float* s_q       = smem;
    float* s_k       = s_q + D;
    float* s_v       = s_k + Bc * D;
    float* s_o       = s_v + Bc * D;
    float* s_scores  = s_o + D;
    float* s_reduce  = s_scores + Bc;

    long long batch_base = static_cast<long long>(b) * T * D;
    long long q_base = batch_base + static_cast<long long>(q_idx) * D;

    // Load Q row + zero O accumulator
    for (int d = tid; d < D; d += bd) {
        s_q[d] = Q[q_base + d];
        s_o[d] = 0.f;
    }
    __syncthreads();

    // Per-block running stats — every thread keeps its own copy, but they
    // are deterministic (computed identically) so they stay in sync.
    float row_max = -INFINITY;
    float row_sum = 0.f;

    int q_end = q_idx;  // causal: keys 0..q_idx (inclusive)

    for (int k_start = 0; k_start <= q_end; k_start += Bc) {
        int k_end = k_start + Bc;
        if (k_end > q_end + 1) k_end = q_end + 1;
        int tile_size = k_end - k_start;

        // Load K, V tile
        long long k_base = batch_base + static_cast<long long>(k_start) * D;
        for (int idx = tid; idx < tile_size * D; idx += bd) {
            int kl = idx / D;
            int dd = idx % D;
            s_k[kl * D + dd] = K[k_base + static_cast<long long>(kl) * D + dd];
            s_v[kl * D + dd] = V[k_base + static_cast<long long>(kl) * D + dd];
        }
        __syncthreads();

        // Compute scores[k_local] = scale * (s_q . s_k[k_local, :])
        for (int k_local = 0; k_local < tile_size; ++k_local) {
            float partial = 0.f;
            for (int d = tid; d < D; d += bd) {
                partial += s_q[d] * s_k[k_local * D + d];
            }
            s_reduce[tid] = partial;
            __syncthreads();
            for (int off = bd >> 1; off > 0; off >>= 1) {
                if (tid < off) s_reduce[tid] += s_reduce[tid + off];
                __syncthreads();
            }
            if (tid == 0) s_scores[k_local] = s_reduce[0] * scale;
            __syncthreads();
        }

        // Tile-local max
        float tile_max = -INFINITY;
        for (int k_local = 0; k_local < tile_size; ++k_local) {
            tile_max = fmaxf(tile_max, s_scores[k_local]);
        }
        float new_max = fmaxf(row_max, tile_max);
        float corr = expf(row_max - new_max);
        // Tile-local sum of exp(s - new_max)
        float tile_sum = 0.f;
        for (int k_local = 0; k_local < tile_size; ++k_local) {
            tile_sum += expf(s_scores[k_local] - new_max);
        }
        row_sum = corr * row_sum + tile_sum;

        // O_new = corr * O_old + sum_k exp(s[k] - new_max) * V[k]
        for (int d = tid; d < D; d += bd) {
            float acc = corr * s_o[d];
            for (int k_local = 0; k_local < tile_size; ++k_local) {
                float p = expf(s_scores[k_local] - new_max);
                acc += p * s_v[k_local * D + d];
            }
            s_o[d] = acc;
        }
        row_max = new_max;
        __syncthreads();
    }

    // Final normalize and write output
    float inv_sum = 1.f / row_sum;
    for (int d = tid; d < D; d += bd) {
        O[q_base + d] = s_o[d] * inv_sum;
    }
    if (tid == 0) {
        L[static_cast<long long>(b) * T + q_idx] = row_max + logf(row_sum);
    }
}

// ----------------------------------------------------------------------------
// Backward kernels
// ----------------------------------------------------------------------------
//
// Math (causal):
//   s[i,j]   = scale * Q[i] . K[j]
//   P[i,j]   = exp(s[i,j] - L[i])             (only j ≤ i; L is logsumexp)
//   O[i,d]   = sum_j P[i,j] * V[j,d]
//   dV[j,d]  = sum_i P[i,j] * dO[i,d]          (only i ≥ j)
//   dP[i,j]  = sum_d dO[i,d] * V[j,d]
//   D[i]     = sum_d O[i,d] * dO[i,d]          (helper)
//   dS[i,j]  = P[i,j] * (dP[i,j] - D[i])
//   dQ[i,d]  = scale * sum_j dS[i,j] * K[j,d]   (only j ≤ i)
//   dK[j,d]  = scale * sum_i dS[i,j] * Q[i,d]   (only i ≥ j)

// Compute D[b, q] = sum_d O[b, q, d] * dO[b, q, d]. One block per (b, q),
// reduction over D. blockDim.x must be a power of two.
__global__ void flash_attn_compute_D_kernel(const float* __restrict__ O,
                                             const float* __restrict__ dO,
                                             float* __restrict__ Dh,
                                             int T, int D) {
    int b = blockIdx.y;
    int q = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    extern __shared__ float smem[];

    long long row_off = (long long)(b * T + q) * D;
    float local = 0.f;
    for (int d = tid; d < D; d += bd) {
        local += O[row_off + d] * dO[row_off + d];
    }
    smem[tid] = local;
    __syncthreads();
    for (int off = bd >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    if (tid == 0) Dh[b * T + q] = smem[0];
}

// dQ kernel — one block per (b, q_idx), iterate K/V tiles k=0..q_idx.
__global__ void flash_attn_dq_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     const float* __restrict__ dO,
                                     const float* __restrict__ Lh,
                                     const float* __restrict__ Dh,
                                     float* __restrict__ dQ,
                                     int T, int D, float scale, int Bc) {
    int b = blockIdx.y;
    int q = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    if (q >= T) return;

    extern __shared__ float smem[];
    float* s_q       = smem;
    float* s_do      = s_q + D;
    float* s_dq      = s_do + D;
    float* s_k       = s_dq + D;
    float* s_v       = s_k + Bc * D;
    float* s_s       = s_v + Bc * D;
    float* s_dp      = s_s + Bc;
    float* s_reduce  = s_dp + Bc;

    long long batch_base = (long long)b * T * D;
    long long q_base = batch_base + (long long)q * D;

    // Load Q[q], dO[q]; zero dQ accumulator
    for (int d = tid; d < D; d += bd) {
        s_q[d]  = Q[q_base + d];
        s_do[d] = dO[q_base + d];
        s_dq[d] = 0.f;
    }
    __syncthreads();

    float L_q = Lh[(long long)b * T + q];
    float D_q = Dh[(long long)b * T + q];
    int q_end = q;  // causal

    for (int k_start = 0; k_start <= q_end; k_start += Bc) {
        int k_lim = k_start + Bc;
        if (k_lim > q_end + 1) k_lim = q_end + 1;
        int tile_size = k_lim - k_start;

        // Load K, V tile
        long long k_base = batch_base + (long long)k_start * D;
        for (int idx = tid; idx < tile_size * D; idx += bd) {
            int kl = idx / D;
            int dd = idx % D;
            s_k[kl * D + dd] = K[k_base + (long long)kl * D + dd];
            s_v[kl * D + dd] = V[k_base + (long long)kl * D + dd];
        }
        __syncthreads();

        // Compute s[k_local] = scale * Q.K[k_local]  and  dP[k_local] = dO.V[k_local]
        for (int kl = 0; kl < tile_size; ++kl) {
            float partial = 0.f;
            for (int d = tid; d < D; d += bd)
                partial += s_q[d] * s_k[kl * D + d];
            s_reduce[tid] = partial;
            __syncthreads();
            for (int off = bd >> 1; off > 0; off >>= 1) {
                if (tid < off) s_reduce[tid] += s_reduce[tid + off];
                __syncthreads();
            }
            if (tid == 0) s_s[kl] = s_reduce[0] * scale;
            __syncthreads();

            partial = 0.f;
            for (int d = tid; d < D; d += bd)
                partial += s_do[d] * s_v[kl * D + d];
            s_reduce[tid] = partial;
            __syncthreads();
            for (int off = bd >> 1; off > 0; off >>= 1) {
                if (tid < off) s_reduce[tid] += s_reduce[tid + off];
                __syncthreads();
            }
            if (tid == 0) s_dp[kl] = s_reduce[0];
            __syncthreads();
        }

        // Accumulate dQ[d] += scale * sum_kl dS[kl] * K[kl, d], where
        // dS[kl] = P[kl] * (dP[kl] - D_q),  P[kl] = exp(s_s[kl] - L_q).
        for (int d = tid; d < D; d += bd) {
            float acc = 0.f;
            for (int kl = 0; kl < tile_size; ++kl) {
                float P  = expf(s_s[kl] - L_q);
                float dS = P * (s_dp[kl] - D_q);
                acc += dS * s_k[kl * D + d];
            }
            s_dq[d] += scale * acc;
        }
        __syncthreads();
    }

    // Write dQ
    for (int d = tid; d < D; d += bd) {
        dQ[q_base + d] = s_dq[d];
    }
}

// dKdV kernel — one block per (b, k_idx), iterate Q tiles q=k_idx..T-1.
__global__ void flash_attn_dkdv_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        const float* __restrict__ dO,
                                        const float* __restrict__ Lh,
                                        const float* __restrict__ Dh,
                                        float* __restrict__ dK,
                                        float* __restrict__ dV,
                                        int T, int D, float scale, int Bq) {
    int b = blockIdx.y;
    int k = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    if (k >= T) return;

    extern __shared__ float smem[];
    float* s_k       = smem;
    float* s_v       = s_k + D;
    float* s_dk      = s_v + D;
    float* s_dv      = s_dk + D;
    float* s_qt      = s_dv + D;
    float* s_dot     = s_qt + Bq * D;
    float* s_l       = s_dot + Bq * D;
    float* s_d       = s_l + Bq;
    float* s_reduce  = s_d + Bq;

    long long batch_base = (long long)b * T * D;
    long long k_base = batch_base + (long long)k * D;

    // Load K[k], V[k]; zero dK, dV
    for (int d = tid; d < D; d += bd) {
        s_k[d]  = K[k_base + d];
        s_v[d]  = V[k_base + d];
        s_dk[d] = 0.f;
        s_dv[d] = 0.f;
    }
    __syncthreads();

    int q_start_global = k;  // causal: q ≥ k
    for (int q_start = q_start_global; q_start < T; q_start += Bq) {
        int q_lim = q_start + Bq;
        if (q_lim > T) q_lim = T;
        int tile_size = q_lim - q_start;

        // Load Q tile, dO tile, L, D for this Q range
        long long qt_base = batch_base + (long long)q_start * D;
        for (int idx = tid; idx < tile_size * D; idx += bd) {
            int ql = idx / D;
            int dd = idx % D;
            s_qt[ql * D + dd]  = Q[qt_base + (long long)ql * D + dd];
            s_dot[ql * D + dd] = dO[qt_base + (long long)ql * D + dd];
        }
        if (tid < tile_size) {
            s_l[tid] = Lh[(long long)b * T + q_start + tid];
            s_d[tid] = Dh[(long long)b * T + q_start + tid];
        }
        __syncthreads();

        for (int ql = 0; ql < tile_size; ++ql) {
            // s = scale * Q[q].K[k]
            float partial = 0.f;
            for (int d = tid; d < D; d += bd)
                partial += s_qt[ql * D + d] * s_k[d];
            s_reduce[tid] = partial;
            __syncthreads();
            for (int off = bd >> 1; off > 0; off >>= 1) {
                if (tid < off) s_reduce[tid] += s_reduce[tid + off];
                __syncthreads();
            }
            float s_score = s_reduce[0] * scale;
            __syncthreads();

            // dP = dO[q].V[k]
            partial = 0.f;
            for (int d = tid; d < D; d += bd)
                partial += s_dot[ql * D + d] * s_v[d];
            s_reduce[tid] = partial;
            __syncthreads();
            for (int off = bd >> 1; off > 0; off >>= 1) {
                if (tid < off) s_reduce[tid] += s_reduce[tid + off];
                __syncthreads();
            }
            float dP = s_reduce[0];
            __syncthreads();

            float P  = expf(s_score - s_l[ql]);
            float dS = P * (dP - s_d[ql]);

            // dK[d] += scale * dS * Q[q,d];  dV[d] += P * dO[q,d]
            for (int d = tid; d < D; d += bd) {
                s_dk[d] += scale * dS * s_qt[ql * D + d];
                s_dv[d] += P  * s_dot[ql * D + d];
            }
            __syncthreads();
        }
    }

    // Write dK, dV
    for (int d = tid; d < D; d += bd) {
        dK[k_base + d] = s_dk[d];
        dV[k_base + d] = s_dv[d];
    }
}

}  // namespace

void flash_attn_forward(const Tensor& q, const Tensor& k, const Tensor& v,
                         Tensor& ctx, Tensor& L,
                         int B, int T, int D, int tile_kv) {
    if (q.dtype() != DType::FP32 || k.dtype() != DType::FP32 ||
        v.dtype() != DType::FP32 || ctx.dtype() != DType::FP32 ||
        L.dtype() != DType::FP32) {
        throw std::invalid_argument("flash_attn_forward: dtype must be FP32");
    }
    if (q.device() != Device::Cuda || k.device() != Device::Cuda ||
        v.device() != Device::Cuda || ctx.device() != Device::Cuda ||
        L.device() != Device::Cuda) {
        throw std::invalid_argument("flash_attn_forward: tensors must be CUDA");
    }
    if (tile_kv <= 0) {
        throw std::invalid_argument("flash_attn_forward: tile_kv must be > 0");
    }
    if (tile_kv > T) tile_kv = T;  // clamp; one tile is fine
    if (q.numel() != static_cast<long long>(B) * T * D ||
        k.numel() != q.numel() || v.numel() != q.numel() ||
        ctx.numel() != q.numel()) {
        throw std::invalid_argument("flash_attn_forward: shape mismatch (qkv/ctx)");
    }
    if (L.numel() != static_cast<long long>(B) * T) {
        throw std::invalid_argument("flash_attn_forward: L must be [B, T]");
    }

    const int block = 128;  // power of two; works for any D
    if ((block & (block - 1)) != 0) {
        throw std::runtime_error("flash_attn_forward: block must be power of two");
    }

    // Shared memory bytes: s_q + 2*Bc*D + s_o + Bc + block (reduction)
    std::size_t smem_bytes =
        (std::size_t)(D + 2 * tile_kv * D + D + tile_kv + block) * sizeof(float);

    float scale = 1.f / std::sqrt(static_cast<float>(D));

    dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
    flash_attn_fwd_kernel<<<grid, block, smem_bytes>>>(
        q.data_as<float>(), k.data_as<float>(), v.data_as<float>(),
        ctx.data_as<float>(), L.data_as<float>(),
        T, D, scale, tile_kv);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void flash_attn_backward(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& ctx, const Tensor& L,
                          const Tensor& d_ctx,
                          Tensor& d_q, Tensor& d_k, Tensor& d_v,
                          int B, int T, int D, int tile) {
    auto check_fp32 = [](const Tensor& t, const char* name) {
        if (t.dtype() != DType::FP32)
            throw std::invalid_argument(std::string("flash_attn_backward: ") +
                                         name + " must be FP32");
        if (t.device() != Device::Cuda)
            throw std::invalid_argument(std::string("flash_attn_backward: ") +
                                         name + " must be CUDA");
    };
    check_fp32(q, "q"); check_fp32(k, "k"); check_fp32(v, "v");
    check_fp32(ctx, "ctx"); check_fp32(L, "L"); check_fp32(d_ctx, "d_ctx");
    check_fp32(d_q, "d_q"); check_fp32(d_k, "d_k"); check_fp32(d_v, "d_v");
    if (tile <= 0)
        throw std::invalid_argument("flash_attn_backward: tile must be > 0");
    if (tile > T) tile = T;

    long long n_total = static_cast<long long>(B) * T * D;
    if (q.numel() != n_total || k.numel() != n_total || v.numel() != n_total ||
        ctx.numel() != n_total || d_ctx.numel() != n_total ||
        d_q.numel() != n_total || d_k.numel() != n_total || d_v.numel() != n_total) {
        throw std::invalid_argument("flash_attn_backward: shape mismatch");
    }
    if (L.numel() != static_cast<long long>(B) * T)
        throw std::invalid_argument("flash_attn_backward: L must be [B, T]");

    const int block = 128;
    if ((block & (block - 1)) != 0) {
        throw std::runtime_error("flash_attn_backward: block must be power of two");
    }

    // 1. D[b, q] = sum_d ctx[b, q, d] * d_ctx[b, q, d]
    Tensor Dh({B, T}, DType::FP32, Device::Cuda);
    {
        dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
        std::size_t smem = static_cast<std::size_t>(block) * sizeof(float);
        flash_attn_compute_D_kernel<<<grid, block, smem>>>(
            ctx.data_as<float>(), d_ctx.data_as<float>(),
            Dh.data_as<float>(), T, D);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }

    float scale = 1.f / std::sqrt(static_cast<float>(D));

    // 2. dQ pass — block per (b, q), iterate K/V tiles.
    {
        // smem: s_q + s_do + s_dq (3*D) + 2*Bc*D + 2*Bc + block
        std::size_t smem_bytes =
            (std::size_t)(3 * D + 2 * tile * D + 2 * tile + block) *
            sizeof(float);
        dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
        flash_attn_dq_kernel<<<grid, block, smem_bytes>>>(
            q.data_as<float>(), k.data_as<float>(), v.data_as<float>(),
            d_ctx.data_as<float>(), L.data_as<float>(), Dh.data_as<float>(),
            d_q.data_as<float>(),
            T, D, scale, tile);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }

    // 3. dKdV pass — block per (b, k), iterate Q tiles.
    {
        // smem: s_k + s_v + s_dk + s_dv (4*D) + 2*Bq*D + 2*Bq + block
        std::size_t smem_bytes =
            (std::size_t)(4 * D + 2 * tile * D + 2 * tile + block) *
            sizeof(float);
        dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
        flash_attn_dkdv_kernel<<<grid, block, smem_bytes>>>(
            q.data_as<float>(), k.data_as<float>(), v.data_as<float>(),
            d_ctx.data_as<float>(), L.data_as<float>(), Dh.data_as<float>(),
            d_k.data_as<float>(), d_v.data_as<float>(),
            T, D, scale, tile);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace modernllm
