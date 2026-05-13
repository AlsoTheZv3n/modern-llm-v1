// Quick FA2 forward vs naive scaled_dot_attention_forward benchmark.
//
// Not a correctness test — that's test_flash_attn. Just timing.

#include <chrono>
#include <cstdio>
#include <functional>
#include <random>

#include "core/gemm.h"
#include "core/tensor.h"
#include "model/attention.h"
#include "model/flash_attn.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

double bench(const char* label, int iters,
              std::function<void()> fn) {
    // Warm-up
    for (int i = 0; i < 3; ++i) fn();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("    %-30s %8.3f ms / call (%d iters)\n",
                 label, ms / iters, iters);
    return ms / iters;
}

}  // namespace

int main() {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    CublasHandle handle;

    auto run_size = [&](int B, int T, int D, int tile) {
        int N = B * T * D;
        Tensor q_h({B, T, D}, DType::FP32, Device::Host);
        Tensor k_h({B, T, D}, DType::FP32, Device::Host);
        Tensor v_h({B, T, D}, DType::FP32, Device::Host);
        Tensor dctx_h({B, T, D}, DType::FP32, Device::Host);
        auto* qp = q_h.data_as<float>();
        auto* kp = k_h.data_as<float>();
        auto* vp = v_h.data_as<float>();
        auto* dp = dctx_h.data_as<float>();
        for (int i = 0; i < N; ++i) qp[i] = dist(rng);
        for (int i = 0; i < N; ++i) kp[i] = dist(rng);
        for (int i = 0; i < N; ++i) vp[i] = dist(rng);
        for (int i = 0; i < N; ++i) dp[i] = dist(rng);
        Tensor q = q_h.to(Device::Cuda);
        Tensor k = k_h.to(Device::Cuda);
        Tensor v = v_h.to(Device::Cuda);
        Tensor dctx = dctx_h.to(Device::Cuda);

        Tensor ctx({B, T, D}, DType::FP32, Device::Cuda);
        Tensor probs({B, T, T}, DType::FP32, Device::Cuda);
        Tensor L({B, T}, DType::FP32, Device::Cuda);
        Tensor dq({B, T, D}, DType::FP32, Device::Cuda);
        Tensor dk({B, T, D}, DType::FP32, Device::Cuda);
        Tensor dv({B, T, D}, DType::FP32, Device::Cuda);

        // Pre-populate caches once for backward bench
        modernllm::scaled_dot_attention_forward(handle, q, k, v, ctx,
                                                  probs, B, T, D);
        modernllm::flash_attn_forward(q, k, v, ctx, L, B, T, D, tile);

        std::printf("  B=%d T=%d D=%d  (probs would be %lld MB)\n",
                     B, T, D, (long long)B * T * T * 4 / (1024 * 1024));
        double tf_naive = bench("naive scaled_dot fwd", 10, [&]() {
            modernllm::scaled_dot_attention_forward(handle, q, k, v, ctx,
                                                      probs, B, T, D);
        });
        double tf_flash = bench("flash_attn fwd", 10, [&]() {
            modernllm::flash_attn_forward(q, k, v, ctx, L, B, T, D, tile);
        });
        std::printf("    => fwd speedup %.2fx\n", tf_naive / tf_flash);

        double tb_naive = bench("naive scaled_dot bwd", 10, [&]() {
            modernllm::scaled_dot_attention_backward(handle, dctx, q, k, v, probs,
                                                       dq, dk, dv, B, T, D);
        });
        double tb_flash = bench("flash_attn bwd", 10, [&]() {
            modernllm::flash_attn_backward(q, k, v, ctx, L, dctx,
                                            dq, dk, dv, B, T, D, tile);
        });
        std::printf("    => bwd speedup %.2fx\n\n", tb_naive / tb_flash);
    };

    std::printf("Flash-Attn 2 fwd+bwd vs naive (RTX 3080 Ti, FP32)\n\n");
    run_size(/*B=*/8, /*T=*/128, /*D=*/64, 32);
    run_size(/*B=*/8, /*T=*/256, /*D=*/64, 32);
    run_size(/*B=*/8, /*T=*/512, /*D=*/64, 32);
    run_size(/*B=*/4, /*T=*/1024, /*D=*/64, 32);
    return 0;
}
