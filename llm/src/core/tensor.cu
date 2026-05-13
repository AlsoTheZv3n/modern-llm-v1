#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "core/cuda_check.h"
#include "core/dtype.h"

namespace modernllm {

namespace {

// Aligned host alloc (64-byte). Free with aligned_free_host.
void* aligned_alloc_host(std::size_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(_MSC_VER)
    void* p = _aligned_malloc(bytes, 64);
    if (!p) throw std::bad_alloc();
    return p;
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) throw std::bad_alloc();
    return p;
#endif
}

void aligned_free_host(void* p) {
    if (!p) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

__global__ void fill_fp32_kernel(float* dst, float value, std::int64_t n) {
    std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (idx < n) dst[idx] = value;
}

__global__ void fill_bf16_kernel(std::uint16_t* dst, std::uint16_t value,
                                 std::int64_t n) {
    std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (idx < n) dst[idx] = value;
}

}  // namespace

std::int64_t Tensor::compute_numel(const std::vector<std::int64_t>& s) {
    std::int64_t n = 1;
    for (auto d : s) {
        if (d < 0) throw std::invalid_argument("negative dimension");
        n *= d;
    }
    return n;
}

Tensor::Tensor(std::vector<std::int64_t> shape, DType dtype, Device device)
    : device_(device), dtype_(dtype), shape_(std::move(shape)),
      numel_(compute_numel(shape_)), owns_(true) {
    allocate_();
}

Tensor::~Tensor() { free_(); }

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), device_(other.device_), dtype_(other.dtype_),
      shape_(std::move(other.shape_)), numel_(other.numel_),
      owns_(other.owns_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.owns_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_();
        data_ = other.data_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        owns_ = other.owns_;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.owns_ = false;
    }
    return *this;
}

void Tensor::allocate_() {
    if (numel_ == 0) {
        data_ = nullptr;
        return;
    }
    const std::size_t b = bytes();
    if (device_ == Device::Host) {
        data_ = aligned_alloc_host(b);
    } else {
        MLLM_CUDA_CHECK(cudaMalloc(&data_, b));
    }
}

void Tensor::free_() {
    if (!owns_ || !data_) {
        data_ = nullptr;
        return;
    }
    if (device_ == Device::Host) {
        aligned_free_host(data_);
    } else {
        // Don't throw from destructor — log via stderr if it ever fires.
        cudaError_t e = cudaFree(data_);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[modernllm] cudaFree failed: %s\n",
                         cudaGetErrorString(e));
        }
    }
    data_ = nullptr;
}

Tensor Tensor::from_blob(void* data, std::vector<std::int64_t> shape,
                         DType dtype, Device device) {
    Tensor t;
    t.data_ = data;
    t.device_ = device;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = compute_numel(t.shape_);
    t.owns_ = false;
    return t;
}

Tensor Tensor::zeros(std::vector<std::int64_t> shape, DType dtype,
                     Device device) {
    Tensor t(std::move(shape), dtype, device);
    t.zero();
    return t;
}

Tensor Tensor::view(std::vector<std::int64_t> new_shape) const {
    std::int64_t n = compute_numel(new_shape);
    if (n != numel_) {
        throw std::invalid_argument("Tensor::view: numel mismatch");
    }
    Tensor t = Tensor::from_blob(const_cast<void*>(data_), std::move(new_shape),
                                  dtype_, device_);
    return t;
}

Tensor Tensor::to(Device target) const {
    if (target == device_) {
        // Non-owning alias — same memory, same lifetime as caller.
        return Tensor::from_blob(const_cast<void*>(data_), shape_, dtype_,
                                  device_);
    }
    Tensor out(shape_, dtype_, target);
    if (numel_ == 0) return out;

    cudaMemcpyKind kind;
    if (device_ == Device::Host && target == Device::Cuda) {
        kind = cudaMemcpyHostToDevice;
    } else if (device_ == Device::Cuda && target == Device::Host) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDefault;
    }
    MLLM_CUDA_CHECK(cudaMemcpy(out.data_, data_, bytes(), kind));
    return out;
}

void Tensor::copy_from(const Tensor& src) {
    if (src.numel_ != numel_) {
        throw std::invalid_argument("Tensor::copy_from: numel mismatch");
    }
    if (src.dtype_ != dtype_) {
        throw std::invalid_argument("Tensor::copy_from: dtype mismatch");
    }
    if (numel_ == 0) return;

    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (src.device_ == Device::Host && device_ == Device::Host) {
        std::memcpy(data_, src.data_, bytes());
        return;
    } else if (src.device_ == Device::Host && device_ == Device::Cuda) {
        kind = cudaMemcpyHostToDevice;
    } else if (src.device_ == Device::Cuda && device_ == Device::Host) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    MLLM_CUDA_CHECK(cudaMemcpy(data_, src.data_, bytes(), kind));
}

void Tensor::fill(float value) {
    if (numel_ == 0) return;
    if (dtype_ == DType::INT32) {
        throw std::invalid_argument(
            "Tensor::fill(float) not supported for INT32 — use copy_from()");
    }

    if (device_ == Device::Host) {
        if (dtype_ == DType::FP32) {
            float* p = static_cast<float*>(data_);
            for (std::int64_t i = 0; i < numel_; ++i) p[i] = value;
        } else {
            std::uint16_t b = f32_to_bf16(value);
            std::uint16_t* p = static_cast<std::uint16_t*>(data_);
            for (std::int64_t i = 0; i < numel_; ++i) p[i] = b;
        }
        return;
    }

    // Device fill via small kernel.
    const int block = 256;
    const std::int64_t grid =
        (numel_ + block - 1) / block;
    if (dtype_ == DType::FP32) {
        fill_fp32_kernel<<<static_cast<unsigned>(grid), block>>>(
            static_cast<float*>(data_), value, numel_);
    } else {
        fill_bf16_kernel<<<static_cast<unsigned>(grid), block>>>(
            static_cast<std::uint16_t*>(data_), f32_to_bf16(value), numel_);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void Tensor::zero() {
    if (numel_ == 0) return;
    if (device_ == Device::Host) {
        std::memset(data_, 0, bytes());
    } else {
        MLLM_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }
}

}  // namespace modernllm
