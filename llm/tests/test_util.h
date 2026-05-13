#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Tiny test helpers — no external framework dependency.
// On failure: print message and exit(1).

#define MLLM_EXPECT(cond)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d  expected: %s\n",         \
                         __FILE__, __LINE__, #cond);                   \
            std::exit(1);                                              \
        }                                                              \
    } while (0)

#define MLLM_EXPECT_EQ(a, b)                                                \
    do {                                                                    \
        auto _a = (a);                                                      \
        auto _b = (b);                                                      \
        if (!(_a == _b)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d  %s == %s  (lhs=%lld rhs=%lld)\n", \
                         __FILE__, __LINE__, #a, #b,                        \
                         static_cast<long long>(_a),                        \
                         static_cast<long long>(_b));                       \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define MLLM_EXPECT_NEAR(a, b, tol)                                          \
    do {                                                                     \
        double _a = static_cast<double>(a);                                  \
        double _b = static_cast<double>(b);                                  \
        double _t = static_cast<double>(tol);                                \
        if (std::fabs(_a - _b) > _t) {                                       \
            std::fprintf(stderr,                                             \
                         "FAIL %s:%d  |%s - %s| > %s  (a=%g b=%g diff=%g)\n",\
                         __FILE__, __LINE__, #a, #b, #tol, _a, _b,           \
                         _a - _b);                                           \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define MLLM_TEST(name) static void name()

#define MLLM_RUN_TEST(name)                                  \
    do {                                                     \
        std::printf("[ RUN  ] %s\n", #name);                 \
        name();                                              \
        std::printf("[  OK  ] %s\n", #name);                 \
    } while (0)
