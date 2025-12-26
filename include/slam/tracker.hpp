#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <opencv2/features2d.hpp>
#include <memory>

namespace slam {

enum class TrackingState {
    NOT_INITIALIZED,
    OK,
    LOST
};

/// Front-end tracker.
///
/// Lifecycle per frame:
///   1. Extract ORB keypoints + descriptors
///   2. If not initialized: attempt stereo-like initialization from two frames
///   3. If tracking: predict pose with constant velocity model, then refine
///      with GPU-matched map-point re-projections and solvePnPRansac
///   4. Decide keyframe insertion criterion
class Tracker {
public:
    struct Config {
        int   orb_features       = 2000;
        float orb_scale_factor   = 1.2f;
        int   orb_levels         = 8;
        int   orb_edge_threshold = 31;
        int   hamming_threshold  = 60;   // max Hamming for a valid match
        float lowe_ratio         = 0.75f;
        int   min_tracked_points = 80;   // below → keyframe insertion
        int   pnp_iterations     = 200;
        float pnp_reprojection   = 5.5f; // pixels (RANSAC threshold)
        int   pnp_min_inliers    = 15;   // minimum PnP inliers to accept pose
        float init_min_disparity = 5.0f;  // min mean 2D disparity (px) before init attempt
        float init_median_depth  = 20.0f; // target median map-point depth after init (m)
    };

    using Ptr = std::shared_ptr<Tracker>;
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    /// Process a new frame. Returns the estimated T_cw pose.
    /// @param frame  New frame (descriptors already extracted, or will be here)
    /// @returns true if tracking succeeded
    bool track(Frame::Ptr frame);

    TrackingState state() const { return state_; }

    /// Call after local_ba->optimize() to refresh the constant-velocity
    /// estimate from the BA-refined keyframe poses.  Without this, velocity_
    /// would be stale (computed from pre-BA PnP poses), causing the next
    /// frame's prediction to drift.
    void notify_ba_update();

    // ── Initialization (called internally) ───────────────────────────────────
private:
    bool initialize(Frame::Ptr frame);
    bool track_with_motion_model(Frame::Ptr frame);
    bool track_local_map(Frame::Ptr frame);
    bool need_new_keyframe(Frame::Ptr frame) const;
    void insert_keyframe(Frame::Ptr frame);

    /// Run GPU Hamming matcher between two descriptor matrices.
    /// Returns vector of cv::DMatch (filtered by ratio test).
    std::vector<cv::DMatch> match_descriptors(
        const cv::Mat& query_desc,
        const cv::Mat& train_desc,
        bool use_ratio = true
    );

    /// Triangulate points between two frames and add them to the map.
    int triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                            const std::vector<cv::DMatch>& matches);

    /// Attempt to recover pose against the full global map when LOST.
    /// Returns true if >= pnp_min_inliers*3 PnP inliers found.
    bool try_relocalize(Frame::Ptr frame);

    Camera         cam_;
    Map::Ptr       map_;
    Config         cfg_;
    TrackingState  state_ = TrackingState::NOT_INITIALIZED;

    cv::Ptr<cv::ORB> orb_;

    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    // PnP-inlier count at the time of the last keyframe insertion.
    // Stored BEFORE triangulation so the KF ratio test compares like-for-like
    // (both sides = PnP inliers only, not inflated by triangulated new points).
    int last_kf_pnp_tracked_ = 0;

    // Constant velocity motion model
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool velocity_valid_ = false;
};

}  // namespace slam
