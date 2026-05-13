cmake_minimum_required(VERSION 3.24)

# Must be set BEFORE project() — CMake's CUDA compiler detection otherwise
# fills in a default (typically 52) before our if-check runs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86" CACHE STRING
        "CUDA architectures to build for (Ampere by default)")
endif()

project(modernllm
    LANGUAGES CXX CUDA
    VERSION 0.0.1
    DESCRIPTION "ModernLLM - C++/CUDA transformer LLM trained from scratch")

# C++23 for host code, CUDA 17/20 (CUDA C++23 not yet supported by all nvcc)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Compile commands for clangd / IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA toolkit (provides cuBLAS, cuRAND, etc.)
find_package(CUDAToolkit REQUIRED)

# Warnings
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# CUDA: line info for profiling, fast math is opt-in per-kernel later
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ---------------------------------------------------------------------------
# Core library: tensor, dtype, device, gemm
# ---------------------------------------------------------------------------
add_library(modernllm_core STATIC
    src/core/tensor.cu
    src/core/gemm.cu
    src/core/random.cu
    src/core/cast.cu
    src/core/scratch.cu
    src/train/adamw.cu
    src/train/loss.cu
    src/train/grad_utils.cu
    src/train/checkpoint.cu
    src/model/modern_gpt.cu
    src/model/embedding.cu
    src/model/linear.cu
    src/model/layernorm.cu
    src/model/attention.cu
    src/model/activations.cu
    src/model/rope.cu
    src/model/rmsnorm.cu
    src/model/flash_attn.cu
    src/tokenizer/char_tokenizer.cpp
    src/data/dataloader.cpp
    src/data/token_meta.cpp
)

target_include_directories(modernllm_core PUBLIC src)

target_link_libraries(modernllm_core
    PUBLIC
        CUDA::cudart
        CUDA::cublas
)

set_target_properties(modernllm_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_tensor tests/test_tensor.cu)
target_include_directories(test_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_tensor PRIVATE modernllm_core)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_gemm tests/test_gemm.cu)
target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_gemm PRIVATE modernllm_core)
add_test(NAME gemm COMMAND test_gemm)

add_executable(test_adamw tests/test_adamw.cu)
target_include_directories(test_adamw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_adamw PRIVATE modernllm_core)
add_test(NAME adamw COMMAND test_adamw)

add_executable(test_loss tests/test_loss.cu)
target_include_directories(test_loss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_loss PRIVATE modernllm_core)
add_test(NAME loss COMMAND test_loss)

add_executable(test_embedding tests/test_embedding.cu)
target_include_directories(test_embedding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_embedding PRIVATE modernllm_core)
add_test(NAME embedding COMMAND test_embedding)

add_executable(test_linear tests/test_linear.cu)
target_include_directories(test_linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linear PRIVATE modernllm_core)
add_test(NAME linear COMMAND test_linear)

add_executable(test_layernorm tests/test_layernorm.cu)
target_include_directories(test_layernorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_layernorm PRIVATE modernllm_core)
add_test(NAME layernorm COMMAND test_layernorm)

add_executable(test_attention tests/test_attention.cu)
target_include_directories(test_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_attention PRIVATE modernllm_core)
add_test(NAME attention COMMAND test_attention)

add_executable(test_activations tests/test_activations.cu)
target_include_directories(test_activations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_activations PRIVATE modernllm_core)
add_test(NAME activations COMMAND test_activations)

add_executable(test_mha tests/test_mha.cu)
target_include_directories(test_mha PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_mha PRIVATE modernllm_core)
add_test(NAME mha COMMAND test_mha)

add_executable(test_rope tests/test_rope.cu)
target_include_directories(test_rope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rope PRIVATE modernllm_core)
add_test(NAME rope COMMAND test_rope)

add_executable(test_rmsnorm tests/test_rmsnorm.cu)
target_include_directories(test_rmsnorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rmsnorm PRIVATE modernllm_core)
add_test(NAME rmsnorm COMMAND test_rmsnorm)

add_executable(test_bf16 tests/test_bf16.cu)
target_include_directories(test_bf16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_bf16 PRIVATE modernllm_core)
add_test(NAME bf16 COMMAND test_bf16)

add_executable(test_flash_attn tests/test_flash_attn.cu)
target_include_directories(test_flash_attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_flash_attn PRIVATE modernllm_core)
add_test(NAME flash_attn COMMAND test_flash_attn)

add_executable(test_bf16_kernels tests/test_bf16_kernels.cu)
target_include_directories(test_bf16_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_bf16_kernels PRIVATE modernllm_core)
add_test(NAME bf16_kernels COMMAND test_bf16_kernels)

# Benchmark, not registered with ctest (manual run only)
add_executable(bench_flash_attn tests/bench_flash_attn.cu)
target_include_directories(bench_flash_attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_flash_attn PRIVATE modernllm_core)

# ---------------------------------------------------------------------------
# Apps
# ---------------------------------------------------------------------------
add_executable(train_bigram apps/train_bigram.cu)
target_link_libraries(train_bigram PRIVATE modernllm_core)

add_executable(train_minigpt apps/train_minigpt.cu)
target_link_libraries(train_minigpt PRIVATE modernllm_core)

add_executable(train_modern_gpt apps/train_modern_gpt.cu)
target_link_libraries(train_modern_gpt PRIVATE modernllm_core)

add_executable(infer_modern_gpt apps/infer_modern_gpt.cu)
target_link_libraries(infer_modern_gpt PRIVATE modernllm_core)

message(STATUS "ModernLLM ${PROJECT_VERSION}")
message(STATUS "  CUDA archs : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA       : ${CUDAToolkit_VERSION} @ ${CUDAToolkit_BIN_DIR}")
