#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <opencv2/features2d.hpp>
#include <memory>

// fwd-declare deep components to avoid pulling torch/TRT into every TU
namespace deep {
class XFeatExtractor;
class TTTLoopDetector;
struct XFeatResult;
}

namespace slam {

enum class TrackingState {
    NOT_INITIALIZED,  // waiting for stereo/mono init
    OK,               // normal tracking
    COASTING,         // tracking failed; dead-reckoning w/ velocity model
    LOST              // coasting limit exceeded; attempting relocalization
};

// front-end tracker — hybrid deep-geometric pipeline:
//   1. XFeat kps + FP16 descs
//   2. stereo single-frame init (metric scale from baseline)
//   3. constant-velocity prediction -> FP16 L2 match -> PnP RANSAC
//   4. L2 ratio confidence -> match_confidence[]
//   5. coast up to 8 frames before LOST
//   6. LOST -> synchronous relocalize or map reset
//   7. KF insertion -> push descs to TTT loop detector
class Tracker {
public:
    struct Config {
        int   max_keypoints      = 2000;
        float anms_min_response  = 0.005f;
        float l2_ratio           = 0.8f;  // Lowe ratio for FP16 L2 matching

        // legacy ORB (non-hybrid mode)
        int   orb_features       = 2000;
        float orb_scale_factor   = 1.2f;
        int   orb_levels         = 8;
        int   orb_edge_threshold = 31;
        int   hamming_threshold  = 60;
        float lowe_ratio         = 0.75f;

        int   min_tracked_points = 80;
        int   pnp_iterations     = 500;
        float pnp_reprojection   = 5.5f;
        int   pnp_min_inliers    = 15;
        float stereo_epi_tol     = 2.0f;
        float stereo_d_min       = 3.0f;
        float stereo_d_max       = 300.0f;

        int   coast_limit        = 8;   // frames before LOST
        int   reloc_timeout      = 20;  // frames before giving up on LG
    };

    using Ptr = std::shared_ptr<Tracker>;

    // basic factory (ORB fallback, no deep components)
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    // hybrid factory — takes ownership of deep component ptrs
    static Ptr create_hybrid(
        const Camera& cam, Map::Ptr map,
        std::shared_ptr<deep::XFeatExtractor>  xfeat,
        std::shared_ptr<deep::TTTLoopDetector>  ttt,
        const Config& cfg = Config{});

    bool track(Frame::Ptr frame);  // returns true if tracking succeeded
    TrackingState state() const { return state_; }

    // call after BA to re-derive velocity from corrected KF poses
    void notify_ba_update();

private:
    bool initialize(Frame::Ptr frame);
    bool track_with_motion_model(Frame::Ptr frame);
    bool track_local_map(Frame::Ptr frame);
    bool need_new_keyframe(Frame::Ptr frame) const;
    void insert_keyframe(Frame::Ptr frame);

    // hybrid feature extraction: XFeat inference -> kps, descs, feat_maps
    void extract_features_hybrid(Frame::Ptr frame);

    void ensure_l2_buffers(int N_q, int N_t);  // lazy alloc for L2 match buffers

    // spatial distribution diagnostic — logged at every KF
    void log_anms_grid_stats(const std::vector<cv::KeyPoint>& kps,
                             int img_w, int img_h,
                             int grid_cols, int grid_rows) const;

    // FP16 L2 matching for XFeat descs
    std::vector<cv::DMatch> match_l2_fp16(
        const cv::Mat& query_descs_fp32,
        const cv::Mat& train_descs_fp32,
        std::vector<float>& out_confidence
    );

    // GPU Hamming matcher w/ ratio test
    std::vector<cv::DMatch> match_descriptors(
        const cv::Mat& query_desc,
        const cv::Mat& train_desc,
        bool use_ratio = true
    );

    int triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                            const std::vector<cv::DMatch>& matches);

    void match_stereo(Frame::Ptr frame);  // GPU stereo epipolar -> fills uR
    int triangulate_stereo(Frame::Ptr frame);  // metric depth from stereo disparity
    bool try_relocalize(Frame::Ptr frame);  // match against full global map

    double compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const;

    Camera         cam_;
    Map::Ptr       map_;
    Config         cfg_;
    TrackingState  state_ = TrackingState::NOT_INITIALIZED;

    cv::Ptr<cv::ORB> orb_;  // legacy ORB (also fallback in hybrid mode)

    // deep frontend (null in non-hybrid mode)
    std::shared_ptr<deep::XFeatExtractor>   xfeat_;
    std::shared_ptr<deep::TTTLoopDetector>  ttt_;
    bool hybrid_mode_ = false;

    // L2 match device buffers (lazy alloc, hybrid mode only)
    __half* d_query_descs_ = nullptr;
    __half* d_train_descs_ = nullptr;
    int*    d_best_idx_    = nullptr;
    float*  d_best_dist_   = nullptr;
    float*  d_pseudo_conf_ = nullptr;
    float*  d_y_q_         = nullptr;  // stereo epipolar coord buffers
    float*  d_y_t_         = nullptr;
    float*  d_x_q_         = nullptr;
    float*  d_x_t_         = nullptr;
    int     d_buf_capacity_ = 0;

    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    int last_kf_pnp_tracked_ = 0;  // PnP inlier count at last KF (pre-triangulation)

    // SE(3) constant velocity model w/ exponential decay
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool velocity_valid_ = false;
    void recompute_velocity_from_ba();

    int lost_streak_     = 0;
};

}  // namespace slam
