#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "model/embedding.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

MLLM_TEST(test_embedding_forward_gather) {
    const int V = 7, D = 4, N = 5;

    std::vector<float> w_data(V * D);
    for (int i = 0; i < V * D; ++i) w_data[i] = static_cast<float>(i) * 0.5f;
    std::vector<int> ids_data = {3, 0, 6, 2, 5};

    Tensor w_h({V, D}, DType::FP32, Device::Host);
    std::memcpy(w_h.data(), w_data.data(), V * D * sizeof(float));
    Tensor ids_h({N}, DType::INT32, Device::Host);
    std::memcpy(ids_h.data(), ids_data.data(), N * sizeof(int));

    Tensor w = w_h.to(Device::Cuda);
    Tensor ids = ids_h.to(Device::Cuda);
    Tensor out({N, D}, DType::FP32, Device::Cuda);

    modernllm::embedding_forward(w, ids, out);

    Tensor out_h = out.to(Device::Host);
    auto* o = out_h.data_as<float>();
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            float expected = w_data[ids_data[n] * D + d];
            MLLM_EXPECT_NEAR(o[n * D + d], expected, 1e-9);
        }
    }
}

MLLM_TEST(test_embedding_backward_scatter_add) {
    // Two ids point to the same row → backward should sum gradients.
    const int V = 4, D = 3, N = 5;
    std::vector<int> ids_data = {2, 0, 2, 1, 2};  // row 2 referenced thrice

    std::vector<float> d_out_data(N * D);
    for (int i = 0; i < N * D; ++i) d_out_data[i] = static_cast<float>(i + 1);

    Tensor ids_h({N}, DType::INT32, Device::Host);
    std::memcpy(ids_h.data(), ids_data.data(), N * sizeof(int));
    Tensor d_out_h({N, D}, DType::FP32, Device::Host);
    std::memcpy(d_out_h.data(), d_out_data.data(), N * D * sizeof(float));

    Tensor ids = ids_h.to(Device::Cuda);
    Tensor d_out = d_out_h.to(Device::Cuda);
    Tensor d_weight = Tensor::zeros({V, D}, DType::FP32, Device::Cuda);

    modernllm::embedding_backward(d_out, ids, d_weight);

    Tensor d_w_h = d_weight.to(Device::Host);
    auto* dw = d_w_h.data_as<float>();

    // Compute reference scatter-sum on host
    std::vector<float> ref(V * D, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            ref[ids_data[n] * D + d] += d_out_data[n * D + d];
        }
    }

    float max_d = 0.f;
    for (int i = 0; i < V * D; ++i) {
        max_d = std::fmax(max_d, std::fabs(dw[i] - ref[i]));
    }
    std::printf("    max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-5f);
}

MLLM_TEST(test_embedding_grad_check_finite_diff) {
    // Numerical check: if weight perturbed by eps at position [v0, d0],
    // out[n, d] changes only when ids[n] == v0 and d == d0.
    const int V = 6, D = 5, N = 8;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> w_data(V * D);
    for (auto& x : w_data) x = dist(rng);
    std::vector<int> ids_data(N);
    for (auto& t : ids_data) t = std::uniform_int_distribution<int>(0, V - 1)(rng);

    // Use a scalar loss: sum(out * upstream).
    std::vector<float> upstream(N * D);
    for (auto& x : upstream) x = dist(rng);

    auto compute_loss = [&](const std::vector<float>& w) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                float v = w[ids_data[n] * D + d];
                s += static_cast<double>(v) * static_cast<double>(upstream[n * D + d]);
            }
        }
        return static_cast<float>(s);
    };

    // Analytical via embedding_backward
    Tensor w_h({V, D}, DType::FP32, Device::Host);
    std::memcpy(w_h.data(), w_data.data(), V * D * sizeof(float));
    Tensor ids_h({N}, DType::INT32, Device::Host);
    std::memcpy(ids_h.data(), ids_data.data(), N * sizeof(int));
    Tensor up_h({N, D}, DType::FP32, Device::Host);
    std::memcpy(up_h.data(), upstream.data(), N * D * sizeof(float));

    Tensor ids = ids_h.to(Device::Cuda);
    Tensor up = up_h.to(Device::Cuda);
    Tensor d_w = Tensor::zeros({V, D}, DType::FP32, Device::Cuda);
    modernllm::embedding_backward(up, ids, d_w);
    Tensor d_w_h = d_w.to(Device::Host);
    auto* dw_a = d_w_h.data_as<float>();

    // Numerical
    const float eps = 1e-3f;
    float max_d = 0.f;
    for (int v = 0; v < V; ++v) {
        for (int d = 0; d < D; ++d) {
            int idx = v * D + d;
            float saved = w_data[idx];
            w_data[idx] = saved + eps;
            float lp = compute_loss(w_data);
            w_data[idx] = saved - eps;
            float lm = compute_loss(w_data);
            w_data[idx] = saved;
            float num = (lp - lm) / (2.f * eps);
            max_d = std::fmax(max_d, std::fabs(num - dw_a[idx]));
        }
    }
    std::printf("    finite-diff max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-3f);
}

int main() {
    MLLM_RUN_TEST(test_embedding_forward_gather);
    MLLM_RUN_TEST(test_embedding_backward_scatter_add);
    MLLM_RUN_TEST(test_embedding_grad_check_finite_diff);
    std::printf("\nAll embedding tests passed.\n");
    return 0;
}
