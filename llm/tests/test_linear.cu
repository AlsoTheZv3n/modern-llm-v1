// Linear layer: forward correctness vs host reference + finite-diff bwd check.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/gemm.h"
#include "core/tensor.h"
#include "model/linear.h"
#include "tests/test_util.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void linear_host(const float* X, const float* W, const float* b, float* Y,
                  int N, int In, int Out) {
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < Out; ++o) {
            float acc = b ? b[o] : 0.f;
            for (int i = 0; i < In; ++i) {
                acc += X[n * In + i] * W[i * Out + o];
            }
            Y[n * Out + o] = acc;
        }
    }
}

}  // namespace

MLLM_TEST(test_linear_forward_matches_host) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int N = 17, In = 13, Out = 11;

    std::vector<float> X(N * In), W(In * Out), bias(Out), Y_ref(N * Out);
    for (auto& x : X) x = d(rng);
    for (auto& x : W) x = d(rng);
    for (auto& x : bias) x = d(rng);
    linear_host(X.data(), W.data(), bias.data(), Y_ref.data(), N, In, Out);

    Tensor X_d({N, In}, DType::FP32, Device::Cuda);
    Tensor W_d({In, Out}, DType::FP32, Device::Cuda);
    Tensor b_d({Out}, DType::FP32, Device::Cuda);
    Tensor Y_d({N, Out}, DType::FP32, Device::Cuda);

    Tensor X_h({N, In}, DType::FP32, Device::Host);
    std::memcpy(X_h.data(), X.data(), X.size() * sizeof(float));
    Tensor W_h({In, Out}, DType::FP32, Device::Host);
    std::memcpy(W_h.data(), W.data(), W.size() * sizeof(float));
    Tensor b_h({Out}, DType::FP32, Device::Host);
    std::memcpy(b_h.data(), bias.data(), bias.size() * sizeof(float));
    X_d.copy_from(X_h);
    W_d.copy_from(W_h);
    b_d.copy_from(b_h);

    CublasHandle h;
    modernllm::linear_forward(h, X_d, W_d, &b_d, Y_d);

    Tensor Y_back = Y_d.to(Device::Host);
    auto* y = Y_back.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < N * Out; ++i) {
        max_d = std::fmax(max_d, std::fabs(y[i] - Y_ref[i]));
    }
    std::printf("    fwd max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-4f);
}

MLLM_TEST(test_linear_backward_finite_diff) {
    // Define scalar loss = sum(Y * upstream). Then:
    //   dY = upstream
    //   dX = upstream @ W^T
    //   dW = X^T @ upstream
    //   db = sum_n upstream[n, :]
    // Compare with finite-difference numerical gradients.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    const int N = 4, In = 5, Out = 3;

    std::vector<float> X(N * In), W(In * Out), bias(Out), upstream(N * Out);
    for (auto& x : X) x = d(rng);
    for (auto& x : W) x = d(rng);
    for (auto& x : bias) x = d(rng);
    for (auto& x : upstream) x = d(rng);

    auto loss = [&](const std::vector<float>& X_, const std::vector<float>& W_,
                     const std::vector<float>& b_) {
        std::vector<float> Y(N * Out);
        linear_host(X_.data(), W_.data(), b_.data(), Y.data(), N, In, Out);
        double s = 0.0;
        for (int i = 0; i < N * Out; ++i) s += Y[i] * upstream[i];
        return static_cast<float>(s);
    };

    // Analytical: run linear_backward with d_Y = upstream
    Tensor X_d({N, In}, DType::FP32, Device::Cuda);
    Tensor W_d({In, Out}, DType::FP32, Device::Cuda);
    Tensor b_d({Out}, DType::FP32, Device::Cuda);
    Tensor dY_d({N, Out}, DType::FP32, Device::Cuda);
    Tensor dX_d({N, In}, DType::FP32, Device::Cuda);
    Tensor dW_d = Tensor::zeros({In, Out}, DType::FP32, Device::Cuda);
    Tensor db_d = Tensor::zeros({Out}, DType::FP32, Device::Cuda);

    auto upload = [&](const std::vector<float>& src, Tensor& dst) {
        Tensor h(dst.shape(), DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        dst.copy_from(h);
    };
    upload(X, X_d);
    upload(W, W_d);
    upload(bias, b_d);
    upload(upstream, dY_d);

    CublasHandle h;
    modernllm::linear_backward(h, X_d, W_d, dY_d, dX_d, dW_d, &db_d);

    Tensor dX_h = dX_d.to(Device::Host);
    Tensor dW_h = dW_d.to(Device::Host);
    Tensor db_h = db_d.to(Device::Host);
    auto* dXa = dX_h.data_as<float>();
    auto* dWa = dW_h.data_as<float>();
    auto* dba = db_h.data_as<float>();

    const float eps = 1e-3f;
    auto perturb = [&](std::vector<float>& v, int i, auto computer) {
        float saved = v[i];
        v[i] = saved + eps;
        float lp = computer();
        v[i] = saved - eps;
        float lm = computer();
        v[i] = saved;
        return (lp - lm) / (2.f * eps);
    };

    float max_dX = 0.f, max_dW = 0.f, max_db = 0.f;
    auto compute_loss = [&]() { return loss(X, W, bias); };
    for (int i = 0; i < N * In; ++i) {
        float num = perturb(X, i, compute_loss);
        max_dX = std::fmax(max_dX, std::fabs(num - dXa[i]));
    }
    for (int i = 0; i < In * Out; ++i) {
        float num = perturb(W, i, compute_loss);
        max_dW = std::fmax(max_dW, std::fabs(num - dWa[i]));
    }
    for (int i = 0; i < Out; ++i) {
        float num = perturb(bias, i, compute_loss);
        max_db = std::fmax(max_db, std::fabs(num - dba[i]));
    }
    std::printf("    dX max_diff=%.3e  dW max_diff=%.3e  db max_diff=%.3e\n",
                 max_dX, max_dW, max_db);
    MLLM_EXPECT(max_dX < 1e-3f);
    MLLM_EXPECT(max_dW < 1e-3f);
    MLLM_EXPECT(max_db < 1e-3f);
}

int main() {
    MLLM_RUN_TEST(test_linear_forward_matches_host);
    MLLM_RUN_TEST(test_linear_backward_finite_diff);
    std::printf("\nAll linear tests passed.\n");
    return 0;
}
