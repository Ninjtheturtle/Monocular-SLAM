// l2_matcher.cu — FP16 L2-distance nearest-neighbour matching.
//
// Kernel design recap:
//   Grid  : (N_q, 1, 1)
//   Block : BLOCK_SIZE = 256 threads
//   Each thread strides over ALL train descriptors with step BLOCK_SIZE,
//   maintaining a thread-local best and second-best L2² accumulator.
//   half2 vectorization: loads two FP16 values per instruction (64-dim / 2 = 32 half2 per desc).
//   Warp-level reduction (32 lanes) via __shfl_down_sync; then one cross-warp
//   reduction at the end (8 warps per 256-thread block → 8 entries in shared memory).

#include "../include/cuda/l2_matcher.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <assert.h>

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE  = 32;

// Pack (dist_float, idx_int) into a uint64 so we can use a single min() reduction.
// dist occupies the high 32 bits so that min(uint64) == argmin by distance.
__device__ __forceinline__ uint64_t pack(float dist, int idx) {
    uint32_t d_bits;
    memcpy(&d_bits, &dist, 4);
    return ((uint64_t)d_bits << 32) | (uint32_t)idx;
}
__device__ __forceinline__ float unpack_dist(uint64_t v) {
    uint32_t d_bits = (uint32_t)(v >> 32);
    float f; memcpy(&f, &d_bits, 4); return f;
}
__device__ __forceinline__ int unpack_idx(uint64_t v) {
    return (int)(uint32_t)(v & 0xFFFFFFFFull);
}

// ---------------------------------------------------------------------------
// l2_ratio_kernel
// Computes best and second-best L2² distance per query via half2 dot products,
// then applies Lowe ratio test and writes pseudo-confidence.
// ---------------------------------------------------------------------------
__global__ void l2_ratio_kernel(
    const __half* __restrict__ d_query,   // [N_q × D]
    const __half* __restrict__ d_train,   // [N_t × D]
    int N_q, int N_t, int D,
    float ratio,
    int*   __restrict__ d_best_idx,
    float* __restrict__ d_best_dist,
    float* __restrict__ d_pseudo_conf)
{
    const int qid = blockIdx.x;
    if (qid >= N_q) return;

    const int half2_per_desc = D / 2;  // D must be even (64 / 2 = 32)
    const __half2* q_h2 = reinterpret_cast<const __half2*>(d_query + qid * D);

    // Thread-local best and second-best (packed dist+idx)
    uint64_t my_best1 = pack(FLT_MAX, -1);
    uint64_t my_best2 = pack(FLT_MAX, -1);

    for (int tid = threadIdx.x; tid < N_t; tid += BLOCK_SIZE) {
        const __half2* t_h2 = reinterpret_cast<const __half2*>(d_train + tid * D);

        // Accumulate L2² via half2 FMAs  (a-b)² = a²-2ab+b², but we just compute
        // sum |q-t|² directly to avoid overflow in half precision for long descriptors.
        // We accumulate in float for accuracy.
        float acc = 0.0f;
        for (int k = 0; k < half2_per_desc; ++k) {
            __half2 diff = __hsub2(q_h2[k], t_h2[k]);
            // Convert diff to float2 and accumulate
            float2 df = __half22float2(diff);
            acc += df.x * df.x + df.y * df.y;
        }

        uint64_t entry = pack(acc, tid);
        if (entry < my_best1) {
            my_best2 = my_best1;
            my_best1 = entry;
        } else if (entry < my_best2) {
            my_best2 = entry;
        }
    }

    // Warp reduction: each warp reduces 32 threads → 1 (best1 and best2 separately)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        uint64_t nb1 = __shfl_down_sync(0xFFFFFFFF, my_best1, offset);
        uint64_t nb2 = __shfl_down_sync(0xFFFFFFFF, my_best2, offset);

        // Merge: track two smallest values
        if (nb1 < my_best1) {
            if (my_best1 < nb2) nb2 = my_best1;
            my_best1 = nb1;
        } else if (nb1 < my_best2) {
            my_best2 = nb1;
        }
        if (nb2 < my_best2) {
            my_best2 = nb2;
        }
    }

    // Each warp lane 0 writes its result to shared memory
    const int num_warps = BLOCK_SIZE / WARP_SIZE;  // 8
    __shared__ uint64_t s_best1[8];
    __shared__ uint64_t s_best2[8];

    int lane  = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        s_best1[warpid] = my_best1;
        s_best2[warpid] = my_best2;
    }
    __syncthreads();

    // Thread 0 reduces across warps
    if (threadIdx.x == 0) {
        uint64_t best1 = s_best1[0];
        uint64_t best2 = s_best2[0];
        for (int w = 1; w < num_warps; ++w) {
            uint64_t b1 = s_best1[w];
            uint64_t b2 = s_best2[w];
            if (b1 < best1) {
                if (best1 < best2) best2 = best1;
                else if (b2 < best2) best2 = b2;
                best1 = b1;
            } else {
                if (b1 < best2) best2 = b1;
                if (b2 < best2) best2 = b2;
            }
        }

        float d1 = unpack_dist(best1);
        float d2 = unpack_dist(best2);
        int   i1 = unpack_idx(best1);

        // Lowe ratio test (on L2 distances, not squared)
        float sd1 = sqrtf(d1);
        float sd2 = sqrtf(d2);
        bool  pass = (d2 > 0.0f) && (sd1 < ratio * sd2);

        d_best_idx[qid]    = pass ? i1 : -1;
        d_best_dist[qid]   = sd1;

        // Pseudo-confidence: how much better is best vs second
        float conf = (sd2 > 1e-6f) ? (1.0f - sd1 / (ratio * sd2)) : 0.0f;
        conf = fmaxf(0.1f, fminf(1.0f, conf));
        d_pseudo_conf[qid] = pass ? conf : 0.1f;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
void cuda_match_l2_fp16(
    const __half* d_query,
    const __half* d_train,
    int N_q, int N_t, int D,
    float ratio,
    int*   d_best_idx,
    float* d_best_dist,
    float* d_pseudo_conf,
    cudaStream_t stream)
{
    assert(D % 2 == 0 && "Descriptor dimension must be even for half2 vectorization");
    if (N_q <= 0 || N_t <= 0) return;

    dim3 grid(N_q, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    l2_ratio_kernel<<<grid, block, 0, stream>>>(
        d_query, d_train, N_q, N_t, D, ratio,
        d_best_idx, d_best_dist, d_pseudo_conf);
}

// ---------------------------------------------------------------------------
// l2_stereo_epipolar_kernel
// Like l2_ratio_kernel but gates each right-frame candidate on:
//   |y_q - y_t| <= epi_tol                    (rectified epipolar row check)
//   d_min <= (x_q - x_t) <= d_max             (valid disparity range)
// Only epipolar-compliant right descriptors enter the L2 accumulator.
// ---------------------------------------------------------------------------
__global__ void l2_stereo_epipolar_kernel(
    const __half* __restrict__ d_query,
    const __half* __restrict__ d_train,
    const float*  __restrict__ d_y_q,
    const float*  __restrict__ d_y_t,
    const float*  __restrict__ d_x_q,
    const float*  __restrict__ d_x_t,
    int N_q, int N_t, int D,
    float epi_tol, float d_min, float d_max, float ratio,
    int*   __restrict__ d_best_idx,
    float* __restrict__ d_best_dist,
    float* __restrict__ d_pseudo_conf)
{
    const int qid = blockIdx.x;
    if (qid >= N_q) return;

    const int half2_per_desc = D / 2;
    const __half2* q_h2 = reinterpret_cast<const __half2*>(d_query + qid * D);
    const float y_q = d_y_q[qid];
    const float x_q = d_x_q[qid];

    uint64_t my_best1 = pack(FLT_MAX, -1);
    uint64_t my_best2 = pack(FLT_MAX, -1);

    for (int tid = threadIdx.x; tid < N_t; tid += BLOCK_SIZE) {
        // Epipolar + disparity gate — reject before any FP16 arithmetic
        float dy = y_q - d_y_t[tid];
        if (dy < 0.f) dy = -dy;
        if (dy > epi_tol) continue;
        float disp = x_q - d_x_t[tid];
        if (disp < d_min || disp > d_max) continue;

        const __half2* t_h2 = reinterpret_cast<const __half2*>(d_train + tid * D);
        float acc = 0.f;
        for (int k = 0; k < half2_per_desc; ++k) {
            float2 df = __half22float2(__hsub2(q_h2[k], t_h2[k]));
            acc += df.x * df.x + df.y * df.y;
        }
        uint64_t entry = pack(acc, tid);
        if (entry < my_best1) { my_best2 = my_best1; my_best1 = entry; }
        else if (entry < my_best2) { my_best2 = entry; }
    }

    // Warp reduction (identical to l2_ratio_kernel)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        uint64_t nb1 = __shfl_down_sync(0xFFFFFFFF, my_best1, offset);
        uint64_t nb2 = __shfl_down_sync(0xFFFFFFFF, my_best2, offset);
        if (nb1 < my_best1) {
            if (my_best1 < nb2) nb2 = my_best1;
            my_best1 = nb1;
        } else if (nb1 < my_best2) { my_best2 = nb1; }
        if (nb2 < my_best2) my_best2 = nb2;
    }

    __shared__ uint64_t s_best1[8], s_best2[8];
    int lane  = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;
    if (lane == 0) { s_best1[warpid] = my_best1; s_best2[warpid] = my_best2; }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t best1 = s_best1[0], best2 = s_best2[0];
        for (int w = 1; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            uint64_t b1 = s_best1[w], b2 = s_best2[w];
            if (b1 < best1) {
                if (best1 < best2) best2 = best1; else if (b2 < best2) best2 = b2;
                best1 = b1;
            } else { if (b1 < best2) best2 = b1; if (b2 < best2) best2 = b2; }
        }
        float d1 = unpack_dist(best1), d2 = unpack_dist(best2);
        int   i1 = unpack_idx(best1);
        float sd1 = sqrtf(d1), sd2 = sqrtf(d2);
        bool  pass = (d2 > 0.f) && (sd1 < ratio * sd2);
        d_best_idx[qid]  = pass ? i1 : -1;
        d_best_dist[qid] = sd1;
        float conf = (sd2 > 1e-6f) ? (1.f - sd1 / (ratio * sd2)) : 0.f;
        d_pseudo_conf[qid] = pass ? fmaxf(0.1f, fminf(1.f, conf)) : 0.1f;
    }
}

void cuda_match_l2_stereo_epipolar(
    const __half* d_query, const __half* d_train,
    const float* d_y_q, const float* d_y_t,
    const float* d_x_q, const float* d_x_t,
    int N_q, int N_t, int D,
    float epi_tol, float d_min, float d_max, float ratio,
    int* d_best_idx, float* d_best_dist, float* d_pseudo_conf,
    cudaStream_t stream)
{
    assert(D % 2 == 0 && "Descriptor dimension must be even for half2 vectorization");
    if (N_q <= 0 || N_t <= 0) return;
    l2_stereo_epipolar_kernel<<<dim3(N_q, 1, 1), dim3(BLOCK_SIZE, 1, 1), 0, stream>>>(
        d_query, d_train,
        d_y_q, d_y_t, d_x_q, d_x_t,
        N_q, N_t, D, epi_tol, d_min, d_max, ratio,
        d_best_idx, d_best_dist, d_pseudo_conf);
}
