#include <cstdio>
#include <vector>

#include "core/tensor.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

MLLM_TEST(test_host_alloc_fill_zero) {
    Tensor t({4, 6}, DType::FP32, Device::Host);
    MLLM_EXPECT_EQ(t.numel(), 24);
    MLLM_EXPECT_EQ(t.bytes(), static_cast<std::size_t>(24 * 4));
    MLLM_EXPECT_EQ(t.ndim(), 2);
    MLLM_EXPECT_EQ(t.dim(0), 4);
    MLLM_EXPECT_EQ(t.dim(1), 6);

    t.fill(3.5f);
    auto* p = t.data_as<float>();
    for (int i = 0; i < t.numel(); ++i) {
        MLLM_EXPECT_NEAR(p[i], 3.5f, 1e-9);
    }

    t.zero();
    for (int i = 0; i < t.numel(); ++i) {
        MLLM_EXPECT_NEAR(p[i], 0.0f, 1e-9);
    }
}

MLLM_TEST(test_host_to_device_to_host_roundtrip) {
    Tensor h({3, 5}, DType::FP32, Device::Host);
    auto* hp = h.data_as<float>();
    for (int i = 0; i < h.numel(); ++i) hp[i] = static_cast<float>(i) * 0.25f;

    Tensor d = h.to(Device::Cuda);
    MLLM_EXPECT_EQ(static_cast<int>(d.device()), static_cast<int>(Device::Cuda));
    MLLM_EXPECT_EQ(d.numel(), h.numel());

    Tensor back = d.to(Device::Host);
    MLLM_EXPECT_EQ(static_cast<int>(back.device()),
                   static_cast<int>(Device::Host));
    auto* bp = back.data_as<float>();
    for (int i = 0; i < back.numel(); ++i) {
        MLLM_EXPECT_NEAR(bp[i], static_cast<float>(i) * 0.25f, 1e-9);
    }
}

MLLM_TEST(test_device_fill) {
    Tensor d({64}, DType::FP32, Device::Cuda);
    d.fill(7.25f);
    Tensor h = d.to(Device::Host);
    auto* hp = h.data_as<float>();
    for (int i = 0; i < h.numel(); ++i) {
        MLLM_EXPECT_NEAR(hp[i], 7.25f, 1e-9);
    }
}

MLLM_TEST(test_view_shares_data) {
    Tensor t({2, 6}, DType::FP32, Device::Host);
    t.fill(1.0f);
    Tensor v = t.view({3, 4});
    MLLM_EXPECT_EQ(v.numel(), t.numel());
    // Views share the same buffer.
    MLLM_EXPECT(v.data() == t.data());
}

MLLM_TEST(test_bf16_roundtrip) {
    Tensor h({8}, DType::BF16, Device::Host);
    h.fill(1.5f);  // exact in bf16
    Tensor d = h.to(Device::Cuda);
    Tensor back = d.to(Device::Host);
    auto* bp = back.data_as<std::uint16_t>();
    for (int i = 0; i < back.numel(); ++i) {
        float f = modernllm::bf16_to_f32(bp[i]);
        MLLM_EXPECT_NEAR(f, 1.5f, 1e-9);
    }
}

int main() {
    MLLM_RUN_TEST(test_host_alloc_fill_zero);
    MLLM_RUN_TEST(test_host_to_device_to_host_roundtrip);
    MLLM_RUN_TEST(test_device_fill);
    MLLM_RUN_TEST(test_view_shares_data);
    MLLM_RUN_TEST(test_bf16_roundtrip);
    std::printf("\nAll tensor tests passed.\n");
    return 0;
}
