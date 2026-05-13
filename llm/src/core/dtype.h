#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace modernllm {

enum class DType {
    FP32,
    BF16,
    INT32,
};

inline std::size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::FP32: return 4;
        case DType::BF16: return 2;
        case DType::INT32: return 4;
    }
    return 0;
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::FP32: return "fp32";
        case DType::BF16: return "bf16";
        case DType::INT32: return "int32";
    }
    return "?";
}

// BF16 = top 16 bits of FP32. Round-to-nearest-even via bit manipulation.
inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round-to-nearest-even
    std::uint32_t lsb = (bits >> 16) & 1;
    std::uint32_t rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float bf16_to_f32(std::uint16_t b) {
    std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace modernllm
