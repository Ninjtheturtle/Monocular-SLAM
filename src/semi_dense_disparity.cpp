// 1D cross-correlation on XFeat 1/8-res feature maps for semi-dense depth
//
// ISOLATION: output is viz-only (rerun world/map/semi_dense) — never enters Map or Ceres
//
// per pixel (r,c) in left feat map:
//   1. L2-normalize left feat vec at (r,c)
//   2. search right cols in [d_min_8, d_max_8], compute NCC w/ right feat vec at (r, c-d)
//   3. find peak & second-peak, apply sharpness filter
//   4. sub-pixel refine via parabolic fit
//   5. depth Z = fx * baseline / (d_subpx * 8)
//   6. reject if Z outside [min_depth, max_depth], unproject -> world frame

#include "../include/deep/semi_dense_disparity.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "../include/deep/xfeat_extractor.hpp"

namespace deep {

SemiDenseDisparity::SemiDenseDisparity(const Config& cfg) : cfg_(cfg) {}

std::vector<SemiDensePoint3D> SemiDenseDisparity::compute(const cv::Mat& feat_left,
                                                          const cv::Mat& feat_right,
                                                          const Eigen::Isometry3d& T_wc) const {
    // feat layout: [C*fh rows x fw cols] — channel c, row r, col x -> feat.at<float>(c*fh+r, x)
    const int C = kXFeatFeatMapC;
    const int fh = feat_left.rows / C;
    const int fw = feat_left.cols;
    assert(feat_left.type() == CV_32F);
    assert(feat_right.type() == CV_32F);
    assert(feat_left.rows == feat_right.rows);
    assert(feat_left.cols == feat_right.cols);
    assert(fh > 0 && fw > 0);

    const float d_min_8 = cfg_.d_min_full / 8.0f; // convert disp limits to 1/8-res
    const float d_max_8 = cfg_.d_max_full / 8.0f;

    std::vector<SemiDensePoint3D> pts;
    pts.reserve(fh * fw / 4);

    const float* L_ptr = feat_left.ptr<float>(0);
    const float* R_ptr = feat_right.ptr<float>(0);

    for (int r = 0; r < fh; ++r) {
        for (int c_left = 0; c_left < fw; ++c_left) {
            // --- L2-normalize left descriptor ---
            float norm2 = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                float v = L_ptr[ch * fh * fw + r * fw + c_left];
                norm2 += v * v;
            }
            if (norm2 < 1e-8f) continue;
            float inv_norm_L = 1.0f / std::sqrt(norm2);

            // --- 1D epipolar search (same row assumption — valid for rectified stereo) ---
            int d_lo = (int)std::ceil(d_min_8);
            int d_hi = (int)std::floor(d_max_8);
            d_lo = std::max(d_lo, 0);
            d_hi = std::min(d_hi, c_left); // c_right must stay >= 0

            if (d_lo > d_hi) continue;

            float best_score = -1.0f, second_score = -1.0f;
            int best_d = -1;

            for (int d = d_lo; d <= d_hi; ++d) {
                int c_right = c_left - d;

                float norm2r = 0.0f;
                for (int ch = 0; ch < C; ++ch) {
                    float v = R_ptr[ch * fh * fw + r * fw + c_right];
                    norm2r += v * v;
                }
                if (norm2r < 1e-8f) continue;
                float inv_norm_R = 1.0f / std::sqrt(norm2r);

                // cosine similarity (= dot after L2-norm)
                float dot = 0.0f;
                for (int ch = 0; ch < C; ++ch) {
                    dot += L_ptr[ch * fh * fw + r * fw + c_left] * inv_norm_L *
                           R_ptr[ch * fh * fw + r * fw + c_right] * inv_norm_R;
                }

                if (dot > best_score) {
                    second_score = best_score;
                    best_score = dot;
                    best_d = d;
                } else if (dot > second_score) {
                    second_score = dot;
                }
            }

            if (best_d < 0) continue;
            if (best_score < 0.0f) continue;

            // sharpness filter — rejects ambiguous matches
            float sharpness =
                (second_score > 0.0f) ? best_score / second_score : best_score * 10.0f;
            if (sharpness < cfg_.min_peak_ratio) continue;

            // sub-pixel refinement via 3-point parabolic fit
            float d_subpx = (float)best_d;
            if (best_d > d_lo && best_d < d_hi) {
                auto score_at = [&](int d) -> float {
                    int c_right = c_left - d;
                    if (c_right < 0 || c_right >= fw) return -1.0f;
                    float norm2r = 0.0f;
                    for (int ch = 0; ch < C; ++ch) {
                        float v = R_ptr[ch * fh * fw + r * fw + c_right];
                        norm2r += v * v;
                    }
                    if (norm2r < 1e-8f) return -1.0f;
                    float invr = 1.0f / std::sqrt(norm2r);
                    float dt = 0.0f;
                    for (int ch = 0; ch < C; ++ch)
                        dt += L_ptr[ch * fh * fw + r * fw + c_left] * inv_norm_L *
                              R_ptr[ch * fh * fw + r * fw + c_right] * invr;
                    return dt;
                };
                float s_m1 = score_at(best_d - 1);
                float s_p1 = score_at(best_d + 1);
                float denom = 2.0f * (2.0f * best_score - s_m1 - s_p1);
                if (std::abs(denom) > 1e-6f) d_subpx = best_d + (s_m1 - s_p1) / denom;
            }

            float d_full = d_subpx * 8.0f; // back to full-res disparity
            if (d_full < 1e-4f) continue;
            float Z = cfg_.fx * cfg_.baseline / d_full;
            if (Z < cfg_.min_depth || Z > cfg_.max_depth) continue;

            // unproject to cam frame — pixel center at 1/8-res position * 8
            float u_full = c_left * 8.0f;
            float v_full = r * 8.0f;
            float X_c = (u_full - cfg_.cx) * Z / cfg_.fx;
            float Y_c = (v_full - cfg_.cy) * Z / cfg_.fx;  // assumes fx == fy (square pixels)

            Eigen::Vector3d p_c(X_c, Y_c, Z);
            Eigen::Vector3d p_w = T_wc * p_c;

            float conf = std::min(1.0f, (sharpness - 1.0f) / (cfg_.min_peak_ratio * 3.0f)); // normalize to [0,1]

            pts.push_back({(float)p_w.x(), (float)p_w.y(), (float)p_w.z(), conf});
        }
    }

    return pts;
}

}  // namespace deep
