#pragma once
// GPU Adaptive Non-Maximal Suppression on VRAM heatmap
// operates entirely in device memory — no host round-trip between XFeat & kp selection
//
// pass 1: NMS — mark local max in (2*R+1)^2 neighbourhood above threshold
//          shared-mem tile max filter
// pass 2: stream-compact candidates, thrust::sort by response, keep top-K

#include <cuda_runtime.h>

// d_heatmap: [HxW] float32 device ptr (row-major)
// d_out_x/y/scores: pre-allocated device arrays [max_kps]
// returns actual kp count (<= max_kps) via sync device->host copy internally
int cuda_anms(
    const float* d_heatmap,
    int H, int W,
    float min_response_thresh,
    int max_kps,
    int nms_radius,
    float* d_out_x,
    float* d_out_y,
    float* d_out_scores,
    cudaStream_t stream = 0
);
