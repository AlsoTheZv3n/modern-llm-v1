#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "core/device.h"
#include "core/dtype.h"

namespace modernllm {

// A simple tensor: contiguous, row-major, optionally on device.
//
// Phase A scope:
//   - alloc / free on Host or Cuda
//   - fill(float), zero()
//   - copy_from(other)  (any device direction)
//   - to(device)         returns a new tensor on the target device
//   - view(new_shape)    reinterprets shape, no copy, must match numel
//
// Non-owning views are allowed via from_blob().
//
// No autograd, no broadcasting, no slicing yet — those come later.
class Tensor {
   public:
    Tensor() = default;
    Tensor(std::vector<std::int64_t> shape, DType dtype, Device device);
    Tensor(std::initializer_list<std::int64_t> shape, DType dtype, Device device)
        : Tensor(std::vector<std::int64_t>(shape), dtype, device) {}

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Borrow externally-owned memory. Caller is responsible for lifetime.
    static Tensor from_blob(void* data, std::vector<std::int64_t> shape,
                            DType dtype, Device device);

    // Allocate uninitialized.
    static Tensor empty(std::vector<std::int64_t> shape, DType dtype,
                        Device device) {
        return Tensor(std::move(shape), dtype, device);
    }
    static Tensor zeros(std::vector<std::int64_t> shape, DType dtype,
                        Device device);

    // Reinterpret as a different shape with same numel. No copy.
    Tensor view(std::vector<std::int64_t> new_shape) const;

    // Move data to target device. If already on target, returns a non-owning
    // alias so existing memory is reused.
    Tensor to(Device target) const;

    // Copy element-wise from src to *this. Shapes and dtypes must match.
    void copy_from(const Tensor& src);

    // Fill with a scalar value (currently FP32-only path; BF16 fills via cast).
    void fill(float value);
    void zero();

    // Accessors
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    template <typename T>
    T* data_as() noexcept {
        return static_cast<T*>(data_);
    }
    template <typename T>
    const T* data_as() const noexcept {
        return static_cast<const T*>(data_);
    }

    Device device() const noexcept { return device_; }
    DType dtype() const noexcept { return dtype_; }
    const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
    std::int64_t numel() const noexcept { return numel_; }
    std::size_t bytes() const noexcept { return numel_ * dtype_bytes(dtype_); }
    int ndim() const noexcept { return static_cast<int>(shape_.size()); }
    std::int64_t dim(int i) const noexcept { return shape_[i]; }
    bool defined() const noexcept { return data_ != nullptr; }

   private:
    void* data_{nullptr};
    Device device_{Device::Host};
    DType dtype_{DType::FP32};
    std::vector<std::int64_t> shape_;
    std::int64_t numel_{0};
    bool owns_{false};

    static std::int64_t compute_numel(const std::vector<std::int64_t>& s);
    void allocate_();
    void free_();
};

}  // namespace modernllm
