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

/// front-end tracker. per-frame pipeline:
///   1. extract ORB keypoints + descriptors
///   2. if not initialized: stereo single-frame init (or monocular two-frame)
///   3. if tracking: predict pose with constant velocity, refine with GPU-matched
///      map-point reprojections and PnP RANSAC
///   4. decide whether to insert a new keyframe
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
        float stereo_epi_tol  = 2.0f;   // stereo epipolar row tolerance (pixels)
        float stereo_d_min    = 3.0f;   // minimum stereo disparity (pixels; ~128 m max depth, 17% depth uncertainty)
        float stereo_d_max    = 300.0f; // maximum stereo disparity (pixels; ~0.35 m depth)
    };

    using Ptr = std::shared_ptr<Tracker>;
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    /// process a new frame; returns true if tracking succeeded
    bool track(Frame::Ptr frame);

    TrackingState state() const { return state_; }

    /// call after BA to invalidate the stale velocity estimate.
    /// see notify_ba_update() in tracker.cpp for why we don't re-derive velocity_ from BA poses.
    void notify_ba_update();

private:
    bool initialize(Frame::Ptr frame);
    bool track_with_motion_model(Frame::Ptr frame);
    bool track_local_map(Frame::Ptr frame);
    bool need_new_keyframe(Frame::Ptr frame) const;
    void insert_keyframe(Frame::Ptr frame);

    /// GPU Hamming matcher; returns cv::DMatch vector filtered by ratio test
    std::vector<cv::DMatch> match_descriptors(
        const cv::Mat& query_desc,
        const cv::Mat& train_desc,
        bool use_ratio = true
    );

    /// triangulate points between two frames and add them to the map
    int triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                            const std::vector<cv::DMatch>& matches);

    /// GPU stereo epipolar matching: fills frame->uR with right x-coords
    void match_stereo(Frame::Ptr frame);

    /// triangulate metric map points from a single stereo frame (frame->uR must be set)
    int triangulate_stereo(Frame::Ptr frame);

    /// attempt to recover pose against the full global map when LOST
    bool try_relocalize(Frame::Ptr frame);

    /// median angular parallax (radians) of tracked map points between frame and ref_kf
    double compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const;

    Camera         cam_;
    Map::Ptr       map_;
    Config         cfg_;
    TrackingState  state_ = TrackingState::NOT_INITIALIZED;

    cv::Ptr<cv::ORB> orb_;

    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    // PnP inlier count at the last KF insertion — saved BEFORE triangulation
    // so the KF ratio test compares like-for-like (not inflated by new points)
    int last_kf_pnp_tracked_ = 0;

    // constant velocity motion model
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool velocity_valid_ = false;

    // consecutive tracking failures; only go LOST after ≥8 (coasting)
    int lost_streak_ = 0;
};

}  // namespace slam
