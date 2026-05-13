#include "model/attention.h"

#include <cmath>
#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// Causal softmax along the last (T) dim with implicit -inf above diagonal.
// One block per (b, t1). blockDim.x must be a power of two.
//
// scores: in-place [B, T, T]
__global__ void causal_softmax_inplace_kernel(float* __restrict__ scores,
                                              int T) {
    int b = blockIdx.y;
    int t1 = blockIdx.x;
    extern __shared__ float smem[];

    float* row = scores + (static_cast<long long>(b) * T + t1) * T;
    int valid_n = t1 + 1;  // positions 0..t1 are unmasked

    // Pass 1: row max over the unmasked prefix
    float local_max = -INFINITY;
    for (int t2 = threadIdx.x; t2 < valid_n; t2 += blockDim.x) {
        local_max = fmaxf(local_max, row[t2]);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + off]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: sum of exp
    float local_sum = 0.f;
    for (int t2 = threadIdx.x; t2 < valid_n; t2 += blockDim.x) {
        local_sum += expf(row[t2] - row_max);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv_sum = 1.f / row_sum;
    __syncthreads();

    // Pass 3: write probs (zero in masked region)
    for (int t2 = threadIdx.x; t2 < T; t2 += blockDim.x) {
        if (t2 < valid_n) {
            row[t2] = expf(row[t2] - row_max) * inv_sum;
        } else {
            row[t2] = 0.f;
        }
    }
}

// Softmax backward, with the row already containing zeros in masked positions
// (from forward). Computes:
//   d_scores[t2] = probs[t2] * (d_probs[t2] - sum_t2'(d_probs[t2'] * probs[t2']))
// for t2 in unmasked region; 0 elsewhere.
//
// One block per (b, t1).
__global__ void causal_softmax_backward_kernel(const float* __restrict__ d_probs,
                                                const float* __restrict__ probs,
                                                float* __restrict__ d_scores,
                                                int T) {
    int b = blockIdx.y;
    int t1 = blockIdx.x;
    extern __shared__ float smem[];

    long long row_off = (static_cast<long long>(b) * T + t1) * T;
    const float* p = probs + row_off;
    const float* dp = d_probs + row_off;
    float* ds = d_scores + row_off;
    int valid_n = t1 + 1;

    // sum_d = sum_t2 d_probs[t2] * probs[t2]   (only unmasked contributes
    // since probs is 0 elsewhere)
    float local = 0.f;
    for (int t2 = threadIdx.x; t2 < valid_n; t2 += blockDim.x) {
        local += dp[t2] * p[t2];
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float dot = smem[0];
    __syncthreads();

    for (int t2 = threadIdx.x; t2 < T; t2 += blockDim.x) {
        if (t2 < valid_n) {
            ds[t2] = p[t2] * (dp[t2] - dot);
        } else {
            ds[t2] = 0.f;
        }
    }
}

void check_3d_fp32_cuda(const Tensor& t, const char* name) {
    if (t.ndim() != 3 || t.dtype() != DType::FP32 ||
        t.device() != Device::Cuda) {
        throw std::invalid_argument(std::string("attn: bad ") + name);
    }
}

}  // namespace

void scaled_dot_attention_forward(cublasHandle_t handle,
                                   const Tensor& q, const Tensor& k,
                                   const Tensor& v,
                                   Tensor& ctx, Tensor& probs,
                                   int B, int T, int D) {
    check_3d_fp32_cuda(q, "q");
    check_3d_fp32_cuda(k, "k");
    check_3d_fp32_cuda(v, "v");
    check_3d_fp32_cuda(ctx, "ctx");
    check_3d_fp32_cuda(probs, "probs");
    if (q.shape()[0] != B || q.shape()[1] != T || q.shape()[2] != D ||
        k.shape() != q.shape() || v.shape() != q.shape() ||
        ctx.shape() != q.shape()) {
        throw std::invalid_argument("attn fwd: shape mismatch");
    }
    if (probs.shape()[0] != B || probs.shape()[1] != T ||
        probs.shape()[2] != T) {
        throw std::invalid_argument("attn fwd: probs shape");
    }

    float scale = 1.f / std::sqrt(static_cast<float>(D));

    // Per-batch GEMMs. Each batch's slice is contiguous [T, D] / [T, T].
    for (int b = 0; b < B; ++b) {
        const float* qb = q.data_as<float>() + static_cast<long long>(b) * T * D;
        const float* kb = k.data_as<float>() + static_cast<long long>(b) * T * D;
        const float* vb = v.data_as<float>() + static_cast<long long>(b) * T * D;
        float* sb = probs.data_as<float>() +
                    static_cast<long long>(b) * T * T;
        float* cb = ctx.data_as<float>() + static_cast<long long>(b) * T * D;

        // scores [T, T] = qb [T, D] @ kb^T [D, T] * scale
        gemm_fp32_rowmajor(handle, sb, qb, kb, T, T, D,
                            /*trans_a=*/false, /*trans_b=*/true,
                            /*alpha=*/scale, /*beta=*/0.0f);
    }

    // Softmax with causal mask, in-place over `probs`.
    {
        const int block = 128;
        dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
        causal_softmax_inplace_kernel<<<grid, block, block * sizeof(float)>>>(
            probs.data_as<float>(), T);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }

    // ctx [T, D] = probs [T, T] @ v [T, D]
    for (int b = 0; b < B; ++b) {
        const float* sb = probs.data_as<float>() +
                            static_cast<long long>(b) * T * T;
        const float* vb = v.data_as<float>() +
                            static_cast<long long>(b) * T * D;
        float* cb = ctx.data_as<float>() +
                     static_cast<long long>(b) * T * D;
        gemm_fp32_rowmajor(handle, cb, sb, vb, T, D, T,
                            /*trans_a=*/false, /*trans_b=*/false);
    }
}

void scaled_dot_attention_backward(cublasHandle_t handle,
                                    const Tensor& d_ctx,
                                    const Tensor& q, const Tensor& k,
                                    const Tensor& v, const Tensor& probs,
                                    Tensor& d_q, Tensor& d_k, Tensor& d_v,
                                    int B, int T, int D) {
    if (d_ctx.shape() != q.shape() || d_q.shape() != q.shape() ||
        d_k.shape() != q.shape() || d_v.shape() != q.shape()) {
        throw std::invalid_argument("attn bwd: shape mismatch");
    }
    if (probs.shape()[0] != B || probs.shape()[1] != T ||
        probs.shape()[2] != T) {
        throw std::invalid_argument("attn bwd: probs shape");
    }

    float scale = 1.f / std::sqrt(static_cast<float>(D));

    // Allocate temps on the same device as q (used per-batch).
    // We can reuse a single [B, T, T] buffer for d_probs and d_scores.
    Tensor d_probs({B, T, T}, DType::FP32, Device::Cuda);

    // Step 1 (per batch):
    //   d_v[b] = probs[b]^T @ d_ctx[b]    (T x T)^T @ (T x D) = (T x D)
    //   d_probs[b] = d_ctx[b] @ v[b]^T    (T x D) @ (D x T) = (T x T)
    for (int b = 0; b < B; ++b) {
        const float* pb = probs.data_as<float>() +
                            static_cast<long long>(b) * T * T;
        const float* dcb = d_ctx.data_as<float>() +
                            static_cast<long long>(b) * T * D;
        const float* vb = v.data_as<float>() +
                            static_cast<long long>(b) * T * D;
        float* dvb = d_v.data_as<float>() +
                      static_cast<long long>(b) * T * D;
        float* dpb = d_probs.data_as<float>() +
                      static_cast<long long>(b) * T * T;

        // d_v = probs^T @ d_ctx
        gemm_fp32_rowmajor(handle, dvb, pb, dcb, T, D, T,
                            /*trans_a=*/true, /*trans_b=*/false);
        // d_probs = d_ctx @ v^T
        gemm_fp32_rowmajor(handle, dpb, dcb, vb, T, T, D,
                            /*trans_a=*/false, /*trans_b=*/true);
    }

    // Step 2: d_scores = softmax_bwd(d_probs, probs), masked positions zeroed.
    // We reuse d_probs storage as the destination by writing to a separate
    // tensor for clarity (avoid in-place aliasing pitfalls).
    Tensor d_scores({B, T, T}, DType::FP32, Device::Cuda);
    {
        const int block = 128;
        dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(B));
        causal_softmax_backward_kernel<<<grid, block, block * sizeof(float)>>>(
            d_probs.data_as<float>(), probs.data_as<float>(),
            d_scores.data_as<float>(), T);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }

    // Step 3 (per batch):
    //   d_scores *= scale  — fold into the GEMMs below as alpha
    //   d_q[b] = (scale * d_scores[b]) @ k[b]
    //   d_k[b] = (scale * d_scores[b])^T @ q[b]
    for (int b = 0; b < B; ++b) {
        const float* dsb = d_scores.data_as<float>() +
                            static_cast<long long>(b) * T * T;
        const float* kb = k.data_as<float>() +
                            static_cast<long long>(b) * T * D;
        const float* qb = q.data_as<float>() +
                            static_cast<long long>(b) * T * D;
        float* dqb = d_q.data_as<float>() +
                      static_cast<long long>(b) * T * D;
        float* dkb = d_k.data_as<float>() +
                      static_cast<long long>(b) * T * D;

        gemm_fp32_rowmajor(handle, dqb, dsb, kb, T, D, T,
                            /*trans_a=*/false, /*trans_b=*/false,
                            /*alpha=*/scale, /*beta=*/0.0f);
        gemm_fp32_rowmajor(handle, dkb, dsb, qb, T, D, T,
                            /*trans_a=*/true, /*trans_b=*/false,
                            /*alpha=*/scale, /*beta=*/0.0f);
    }
}

// ---------------------------------------------------------------------------
// Multi-head split / merge
// ---------------------------------------------------------------------------
namespace {

__global__ void split_heads_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int B, int T, int H, int d_h) {
    long long total = static_cast<long long>(B) * H * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k = static_cast<int>(idx % d_h);
    int t = static_cast<int>((idx / d_h) % T);
    int bh = static_cast<int>(idx / (static_cast<long long>(T) * d_h));
    int b = bh / H;
    int h = bh % H;
    long long src = (((static_cast<long long>(b) * T) + t) * H + h) * d_h + k;
    out[idx] = in[src];
}

__global__ void merge_heads_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int B, int T, int H, int d_h) {
    long long total = static_cast<long long>(B) * H * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k = static_cast<int>(idx % d_h);
    int t = static_cast<int>((idx / d_h) % T);
    int bh = static_cast<int>(idx / (static_cast<long long>(T) * d_h));
    int b = bh / H;
    int h = bh % H;
    long long dst = (((static_cast<long long>(b) * T) + t) * H + h) * d_h + k;
    out[dst] = in[idx];
}

// BF16 sister kernels (pure byte movement — no math).
__global__ void split_heads_kernel_bf16(const unsigned short* __restrict__ in,
                                         unsigned short* __restrict__ out,
                                         int B, int T, int H, int d_h) {
    long long total = static_cast<long long>(B) * H * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k = static_cast<int>(idx % d_h);
    int t = static_cast<int>((idx / d_h) % T);
    int bh = static_cast<int>(idx / (static_cast<long long>(T) * d_h));
    int b = bh / H;
    int h = bh % H;
    long long src = (((static_cast<long long>(b) * T) + t) * H + h) * d_h + k;
    out[idx] = in[src];
}

__global__ void merge_heads_kernel_bf16(const unsigned short* __restrict__ in,
                                         unsigned short* __restrict__ out,
                                         int B, int T, int H, int d_h) {
    long long total = static_cast<long long>(B) * H * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k = static_cast<int>(idx % d_h);
    int t = static_cast<int>((idx / d_h) % T);
    int bh = static_cast<int>(idx / (static_cast<long long>(T) * d_h));
    int b = bh / H;
    int h = bh % H;
    long long dst = (((static_cast<long long>(b) * T) + t) * H + h) * d_h + k;
    out[dst] = in[idx];
}

void check_layout_for_perm(const Tensor& t, int total, const char* name) {
    if ((t.dtype() != DType::FP32 && t.dtype() != DType::BF16) ||
        t.device() != Device::Cuda) {
        throw std::invalid_argument(std::string("perm: bad dtype/device on ") +
                                     name);
    }
    if (t.numel() != total) {
        throw std::invalid_argument(std::string("perm: numel mismatch on ") +
                                     name);
    }
}

}  // namespace

void split_heads(const Tensor& in, Tensor& out, int B, int T, int H, int d_h) {
    int total = B * H * T * d_h;
    check_layout_for_perm(in, total, "in");
    check_layout_for_perm(out, total, "out");
    if (in.dtype() != out.dtype())
        throw std::invalid_argument("split_heads: in/out dtype mismatch");
    if (total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    if (in.dtype() == DType::FP32) {
        split_heads_kernel<<<grid, block>>>(in.data_as<float>(),
                                              out.data_as<float>(),
                                              B, T, H, d_h);
    } else {
        split_heads_kernel_bf16<<<grid, block>>>(
            static_cast<const unsigned short*>(in.data()),
            static_cast<unsigned short*>(out.data()),
            B, T, H, d_h);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void merge_heads(const Tensor& in, Tensor& out, int B, int T, int H, int d_h) {
    int total = B * H * T * d_h;
    check_layout_for_perm(in, total, "in");
    check_layout_for_perm(out, total, "out");
    if (in.dtype() != out.dtype())
        throw std::invalid_argument("merge_heads: in/out dtype mismatch");
    if (total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    if (in.dtype() == DType::FP32) {
        merge_heads_kernel<<<grid, block>>>(in.data_as<float>(),
                                              out.data_as<float>(),
                                              B, T, H, d_h);
    } else {
        merge_heads_kernel_bf16<<<grid, block>>>(
            static_cast<const unsigned short*>(in.data()),
            static_cast<unsigned short*>(out.data()),
            B, T, H, d_h);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// GQA helpers: KV head replication + gradient accumulation
// ---------------------------------------------------------------------------
namespace {

__global__ void repeat_kv_heads_kernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int B, int n_kv, int n_q,
                                        int T, int d_h, int group_size) {
    long long total = static_cast<long long>(B) * n_q * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k  = static_cast<int>(idx % d_h);
    int t  = static_cast<int>((idx / d_h) % T);
    int hq = static_cast<int>((idx / (static_cast<long long>(T) * d_h)) % n_q);
    int b  = static_cast<int>(idx /
                (static_cast<long long>(n_q) * T * d_h));
    int h_kv = hq / group_size;
    long long src = (((static_cast<long long>(b) * n_kv) + h_kv) * T + t) * d_h
                    + k;
    out[idx] = in[src];
}

__global__ void accumulate_kv_grads_kernel(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int B, int n_kv, int n_q,
                                            int T, int d_h, int group_size) {
    long long total = static_cast<long long>(B) * n_q * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k  = static_cast<int>(idx % d_h);
    int t  = static_cast<int>((idx / d_h) % T);
    int hq = static_cast<int>((idx / (static_cast<long long>(T) * d_h)) % n_q);
    int b  = static_cast<int>(idx /
                (static_cast<long long>(n_q) * T * d_h));
    int h_kv = hq / group_size;
    long long dst = (((static_cast<long long>(b) * n_kv) + h_kv) * T + t) * d_h
                    + k;
    atomicAdd(&out[dst], in[idx]);
}

// BF16 sister: pure byte-copy (replication is just reads, no math).
__global__ void repeat_kv_heads_kernel_bf16(const unsigned short* __restrict__ in,
                                             unsigned short* __restrict__ out,
                                             int B, int n_kv, int n_q,
                                             int T, int d_h, int group_size) {
    long long total = static_cast<long long>(B) * n_q * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k  = static_cast<int>(idx % d_h);
    int t  = static_cast<int>((idx / d_h) % T);
    int hq = static_cast<int>((idx / (static_cast<long long>(T) * d_h)) % n_q);
    int b  = static_cast<int>(idx /
                (static_cast<long long>(n_q) * T * d_h));
    int h_kv = hq / group_size;
    long long src = (((static_cast<long long>(b) * n_kv) + h_kv) * T + t) * d_h
                    + k;
    out[idx] = in[src];
}

// BF16 source → FP32 destination accumulator. CUDA's atomicAdd works on
// fp32 only; we cast each BF16 value to FP32 inside the atomic. The caller
// is then responsible for casting the FP32 dest back to BF16 if needed.
__device__ __forceinline__ float bf16_to_f32_dev_attn(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    return __uint_as_float(bits);
}

__global__ void accumulate_kv_grads_bf16_to_fp32_kernel(
        const unsigned short* __restrict__ in,
        float* __restrict__ out,
        int B, int n_kv, int n_q,
        int T, int d_h, int group_size) {
    long long total = static_cast<long long>(B) * n_q * T * d_h;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int k  = static_cast<int>(idx % d_h);
    int t  = static_cast<int>((idx / d_h) % T);
    int hq = static_cast<int>((idx / (static_cast<long long>(T) * d_h)) % n_q);
    int b  = static_cast<int>(idx /
                (static_cast<long long>(n_q) * T * d_h));
    int h_kv = hq / group_size;
    long long dst = (((static_cast<long long>(b) * n_kv) + h_kv) * T + t) * d_h
                    + k;
    atomicAdd(&out[dst], bf16_to_f32_dev_attn(in[idx]));
}

}  // namespace

void repeat_kv_heads(const Tensor& in, Tensor& out,
                      int B, int n_kv, int n_q, int T, int d_h) {
    if (n_q % n_kv != 0) {
        throw std::invalid_argument(
            "repeat_kv_heads: n_q must be a multiple of n_kv");
    }
    int group_size = n_q / n_kv;
    int in_total = B * n_kv * T * d_h;
    int out_total = B * n_q * T * d_h;
    check_layout_for_perm(in, in_total, "in");
    check_layout_for_perm(out, out_total, "out");
    if (in.dtype() != out.dtype())
        throw std::invalid_argument("repeat_kv_heads: dtype mismatch");
    if (out_total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((out_total + block - 1) / block);
    if (in.dtype() == DType::FP32) {
        repeat_kv_heads_kernel<<<grid, block>>>(
            in.data_as<float>(), out.data_as<float>(),
            B, n_kv, n_q, T, d_h, group_size);
    } else {
        repeat_kv_heads_kernel_bf16<<<grid, block>>>(
            static_cast<const unsigned short*>(in.data()),
            static_cast<unsigned short*>(out.data()),
            B, n_kv, n_q, T, d_h, group_size);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void accumulate_kv_grads(const Tensor& in, Tensor& out,
                          int B, int n_kv, int n_q, int T, int d_h) {
    if (n_q % n_kv != 0) {
        throw std::invalid_argument(
            "accumulate_kv_grads: n_q must be a multiple of n_kv");
    }
    int group_size = n_q / n_kv;
    int in_total = B * n_q * T * d_h;
    int out_total = B * n_kv * T * d_h;
    check_layout_for_perm(in, in_total, "in");
    check_layout_for_perm(out, out_total, "out");
    if (in_total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((in_total + block - 1) / block);
    // The output buffer is always FP32 (atomicAdd doesn't work natively on
    // BF16 — caller casts back to BF16 if the rest of the bwd path is BF16).
    if (out.dtype() != DType::FP32) {
        throw std::invalid_argument(
            "accumulate_kv_grads: out must be FP32 (atomic accumulator)");
    }
    if (in.dtype() == DType::FP32) {
        accumulate_kv_grads_kernel<<<grid, block>>>(
            in.data_as<float>(), out.data_as<float>(),
            B, n_kv, n_q, T, d_h, group_size);
    } else if (in.dtype() == DType::BF16) {
        accumulate_kv_grads_bf16_to_fp32_kernel<<<grid, block>>>(
            static_cast<const unsigned short*>(in.data()),
            out.data_as<float>(),
            B, n_kv, n_q, T, d_h, group_size);
    } else {
        throw std::invalid_argument("accumulate_kv_grads: in dtype");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
