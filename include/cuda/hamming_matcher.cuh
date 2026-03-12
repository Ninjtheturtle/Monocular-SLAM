#pragma once

#include <cstdint>
#include <cstdio>

// CUDA error-checking macro — aborts with file/line on failure

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ORB descriptors are 256-bit = 32 bytes = 8 × uint32_t

static constexpr int kDescBytes  = 32;
static constexpr int kDescUint32 = 8;   // 32 bytes / 4 bytes per uint32_t
static constexpr int kMaxHamming = 256; // maximum possible Hamming distance

// GPU nearest-neighbour matcher for ORB descriptors (Hamming distance).
// all pointers are host pointers — device allocation is internal.

/// GPU Hamming NN matcher. writes best match index + distance per query.
/// h_query/h_train: N×32 byte row-major descriptor arrays (host pointers)
void cuda_match_hamming(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    int*           h_best_idx,
    int*           h_best_dist
);

/// same as cuda_match_hamming but applies Lowe ratio test.
/// retains match i only when best_dist[i] / second_dist[i] < ratio.
void cuda_match_hamming_ratio(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);

/// stereo epipolar matcher: restricts candidates to those satisfying
///   |y_q - y_t| <= epi_tol   and   d_min <= x_q - x_t <= d_max
/// then applies Lowe ratio test. writes best right index per left descriptor.
void cuda_match_stereo_epipolar(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    const float*   h_y_query,
    const float*   h_y_train,
    const float*   h_x_query,
    const float*   h_x_train,
    float          epi_tol,
    float          d_min,
    float          d_max,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);
