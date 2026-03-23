#pragma once
// FP16 L2-distance NN matching w/ Lowe ratio test
//
// grid: (N_q, 1, 1) — one block per query
// block: 256 threads, stride over train set w/ thread-local best/second L2^2
// warp reduction via __shfl_down_sync, then cross-warp via shared mem
// half2 vectorization halves bandwidth for 64-dim FP16 descs
//
// pseudo-confidence output:
//   w = clamp(1 - best/ratio*second, 0.1, 1.0)
//   not currently used by BA but available for future per-obs weighting

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// d_query/d_train: device ptrs, [N x D] FP16 row-major
// D must be even (half2); typically 64
// d_best_idx: -1 if ratio test failed
// d_pseudo_conf: [0.1, 1.0] confidence
// all outputs written async on `stream`
void cuda_match_l2_fp16(
    const __half* d_query,
    const __half* d_train,
    int N_q, int N_t, int D,
    float ratio,
    int*   d_best_idx,
    float* d_best_dist,
    float* d_pseudo_conf,
    cudaStream_t stream = 0
);

// stereo variant: gates on |y_q-y_t| <= epi_tol & d_min <= (x_q-x_t) <= d_max
// before computing L2 — only epipolar-compliant right descs compete
void cuda_match_l2_stereo_epipolar(
    const __half* d_query,
    const __half* d_train,
    const float*  d_y_q,
    const float*  d_y_t,
    const float*  d_x_q,
    const float*  d_x_t,
    int N_q, int N_t, int D,
    float epi_tol, float d_min, float d_max, float ratio,
    int*   d_best_idx,
    float* d_best_dist,
    float* d_pseudo_conf,
    cudaStream_t stream = 0
);
