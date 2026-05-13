#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace modernllm {

[[noreturn]] inline void throw_cuda_error(const char* expr, const char* file,
                                          int line, cudaError_t err) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "CUDA error at %s:%d in `%s`: %s (%s)",
                  file, line, expr, cudaGetErrorName(err),
                  cudaGetErrorString(err));
    throw std::runtime_error(buf);
}

[[noreturn]] inline void throw_cublas_error(const char* expr, const char* file,
                                            int line, cublasStatus_t status) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "cuBLAS error at %s:%d in `%s`: status=%d",
                  file, line, expr, static_cast<int>(status));
    throw std::runtime_error(buf);
}

}  // namespace modernllm

#define MLLM_CUDA_CHECK(expr)                                       \
    do {                                                            \
        cudaError_t _e = (expr);                                    \
        if (_e != cudaSuccess) {                                    \
            ::modernllm::throw_cuda_error(#expr, __FILE__, __LINE__, _e); \
        }                                                           \
    } while (0)

#define MLLM_CUBLAS_CHECK(expr)                                          \
    do {                                                                 \
        cublasStatus_t _s = (expr);                                      \
        if (_s != CUBLAS_STATUS_SUCCESS) {                               \
            ::modernllm::throw_cublas_error(#expr, __FILE__, __LINE__, _s); \
        }                                                                \
    } while (0)
