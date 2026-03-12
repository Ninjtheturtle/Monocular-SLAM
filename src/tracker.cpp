#include "slam/tracker.hpp"
#include "slam/map_point.hpp"
#include "cuda/hamming_matcher.cuh"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/SVD>
#include <iostream>
#include <atomic>
#include <unordered_set>

namespace slam {

// global frame and point ID counters
static std::atomic<long> g_frame_id{0};
static std::atomic<long> g_point_id{0};

// factory

Tracker::Ptr Tracker::create(const Camera& cam, Map::Ptr map, const Config& cfg)
{
    auto t = std::shared_ptr<Tracker>(new Tracker());
    t->cam_ = cam;
    t->map_ = map;
    t->cfg_ = cfg;
    t->orb_ = cv::ORB::create(
        cfg.orb_features,
        cfg.orb_scale_factor,
        cfg.orb_levels,
        cfg.orb_edge_threshold
    );
    return t;
}

// main entry point

bool Tracker::track(Frame::Ptr frame)
{
    // extract ORB features on left image
    orb_->detectAndCompute(frame->image_gray, cv::noArray(),
                           frame->keypoints, frame->descriptors);
    frame->map_points.resize(frame->keypoints.size(), nullptr);

    // extract ORB on right image and run stereo epipolar matching
    if (!frame->image_right.empty()) {
        orb_->detectAndCompute(frame->image_right, cv::noArray(),
                               frame->keypoints_right, frame->descriptors_right);
        frame->uR.assign(frame->keypoints.size(), -1.0f);
        match_stereo(frame);
    }

    // LOST: try to relocalize against the full map before reinitializing
    if (state_ == TrackingState::LOST) {
        if (try_relocalize(frame)) {
            std::cout << "[Tracker] Relocalized successfully — resuming on existing map\n";
            velocity_valid_ = false;   // old velocity invalid after gap
            state_          = TrackingState::OK;
            last_frame_     = frame;
            return true;
        }
        // relocalization failed — reset and reinit from current position
        std::cerr << "[Tracker] LOST — relocalization failed, resetting map + re-initializing\n";
        frame->T_cw = velocity_valid_ ? (velocity_ * last_frame_->T_cw) : last_frame_->T_cw;
        map_->reset();
        last_keyframe_       = nullptr;
        last_kf_pnp_tracked_ = 0;
        velocity_valid_      = false;
        last_frame_          = frame;
        state_               = TrackingState::NOT_INITIALIZED;
        return false;
    }

    if (state_ == TrackingState::NOT_INITIALIZED)
        return initialize(frame);

    // OK state
    bool ok = track_with_motion_model(frame);
    if (ok) ok = track_local_map(frame);
    if (ok) {
        lost_streak_ = 0;
    } else {
        ++lost_streak_;
        if (lost_streak_ < 8) {
            // coast: dead-reckon for a few frames before declaring LOST
            frame->T_cw = velocity_valid_
                ? velocity_ * last_frame_->T_cw
                : last_frame_->T_cw;
            last_frame_ = frame;
            return false;
        }
        lost_streak_ = 0;
        state_ = TrackingState::LOST;
    }
    last_frame_ = frame;
    return ok;
}

// initialization

bool Tracker::initialize(Frame::Ptr frame)
{
    if (cam_.is_stereo() && !frame->uR.empty()) {
        // stereo path: propagate last pose so reinit doesn't jump to origin
        if (last_frame_) {
            frame->T_cw = last_frame_->T_cw;
        }
        int n_pts = triangulate_stereo(frame);
        if (n_pts < 50) {
            std::cerr << "[Tracker] Stereo init: too few points (" << n_pts << "), retrying\n";
            last_frame_ = frame;
            return false;
        }
        insert_keyframe(frame);
        velocity_valid_ = false;   // no velocity until second tracked frame
        state_          = TrackingState::OK;
        last_frame_     = frame;
        std::cout << "[Tracker] Stereo initialized: " << n_pts << " metric map points\n";
        return true;
    }

    // monocular fallback
    if (!last_frame_) {
        last_frame_ = frame;
        return false;
    }

    auto matches = match_descriptors(last_frame_->descriptors,
                                     frame->descriptors, /*ratio=*/true);
    if ((int)matches.size() < 50) {
        std::cerr << "[Tracker] Init: too few matches (" << matches.size()
                  << "), q=" << last_frame_->descriptors.rows
                  << " t=" << frame->descriptors.rows << "\n";
        last_frame_ = frame;
        return false;
    }

    std::vector<cv::Point2f> pts0, pts1;
    for (auto& m : matches) {
        pts0.push_back(last_frame_->keypoints[m.queryIdx].pt);
        pts1.push_back(frame->keypoints[m.trainIdx].pt);
    }

    // require sufficient 2D feature displacement to avoid degenerate init
    {
        double sum_disp = 0.0;
        for (size_t i = 0; i < pts0.size(); ++i) {
            double dx = pts1[i].x - pts0[i].x;
            double dy = pts1[i].y - pts0[i].y;
            sum_disp += std::sqrt(dx*dx + dy*dy);
        }
        double mean_disp = sum_disp / pts0.size();
        if (mean_disp < 5.0) {
            std::cerr << "[Tracker] Init: low disparity (" << mean_disp
                      << " px, need 5.0), holding anchor\n";
            return false;
        }
    }

    cv::Mat E, mask;
    E = cv::findEssentialMat(pts0, pts1,
                             cam_.K_cv(), cv::RANSAC,
                             0.999, 1.0, 1000, mask);
    if (E.empty()) {
        std::cerr << "[Tracker] Init: essential matrix empty\n";
        last_frame_ = frame; return false;
    }

    cv::Mat R_cv, t_cv;
    int inliers = cv::recoverPose(E, pts0, pts1,
                                  cam_.K_cv(), R_cv, t_cv, mask);
    if (inliers < 20) {
        std::cerr << "[Tracker] Init: recoverPose inliers too low (" << inliers << ")\n";
        last_frame_ = frame; return false;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R(r, c) = R_cv.at<double>(r, c);
    t << t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2);

    Eigen::Isometry3d T_rel = Eigen::Isometry3d::Identity();
    T_rel.linear()      = R;
    T_rel.translation() = t;
    frame->T_cw = T_rel * last_frame_->T_cw;

    int n_pts = triangulate_and_add(last_frame_, frame, matches);
    if (n_pts < 20) {
        std::cerr << "[Tracker] Init: triangulation too sparse (" << n_pts << " pts)\n";
        last_frame_ = frame; return false;
    }

    // scale to ~20 m median depth (monocular scale ambiguity workaround)
    {
        std::vector<double> depths;
        depths.reserve(n_pts);
        for (auto& mp : map_->all_map_points()) {
            Eigen::Vector3d Xc = frame->T_cw * mp->position;
            if (Xc.z() > 0.0) depths.push_back(Xc.z());
        }
        if (!depths.empty()) {
            std::sort(depths.begin(), depths.end());
            double median = depths[depths.size() / 2];
            double scale  = 20.0 / median;
            Eigen::Vector3d C0 = last_frame_->camera_center();
            Eigen::Vector3d C1 = frame->camera_center();
            Eigen::Vector3d C1_s = C0 + scale * (C1 - C0);
            frame->T_cw.translation() = -frame->T_cw.linear() * C1_s;
            for (auto& mp : map_->all_map_points())
                mp->position = C0 + scale * (mp->position - C0);
        }
    }

    insert_keyframe(last_frame_);
    insert_keyframe(frame);

    velocity_       = frame->T_cw * last_frame_->T_cw.inverse();
    velocity_valid_ = true;
    state_          = TrackingState::OK;
    last_frame_     = frame;

    std::cout << "[Tracker] Monocular initialized with " << n_pts << " map points\n";
    return true;
}

// constant-velocity tracking — matches against all local KF map points

bool Tracker::track_with_motion_model(Frame::Ptr frame)
{
    // predict pose
    if (velocity_valid_) {
        frame->T_cw = velocity_ * last_frame_->T_cw;
    } else {
        frame->T_cw = last_frame_->T_cw;
    }

    // build descriptor pool from the last 30 KFs
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        for (auto& kf : map_->local_window(30)) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (mp->observed_times < 2) continue; // skip unverified single-view points
                if (!seen_ids.insert(mp->id).second) continue; // skip duplicates
                if (i >= kf->descriptors.rows) continue;
                pool_desc.push_back(kf->descriptors.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if ((int)pool_desc.rows < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: pool too small (" << pool_desc.rows << " pts)\n";
        return false;
    }

    // phase 1: spatial search around projected pool points. phase 2: GPU Hamming fallback.
    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    // phase 1: projection-based spatial matching
    {
        const int   cell     = 16;
        const int   frame_w  = frame->image_gray.cols;
        const int   frame_h  = frame->image_gray.rows;
        const int   n_cols_g = (frame_w + cell - 1) / cell;
        const int   n_rows_g = (frame_h + cell - 1) / cell;
        // widen search at turns; base_r=70 when velocity is invalid (one frame after BA)
        const double pred_ang = velocity_valid_
            ? Eigen::AngleAxisd(velocity_.rotation()).angle() : 0.0;
        const float base_r = velocity_valid_ ? 40.0f : 70.0f;
        const float search_r = (pred_ang > 0.03)
            ? std::min(120.0f, base_r + float(pred_ang / 0.03) * 20.0f)
            : base_r;
        const int   max_ham  = cfg_.hamming_threshold;

        // build spatial grid: cell → list of keypoint indices
        std::vector<std::vector<int>> kp_grid(n_cols_g * n_rows_g);
        for (int j = 0; j < (int)frame->keypoints.size(); ++j) {
            int cx = std::min(n_cols_g - 1, (int)(frame->keypoints[j].pt.x / cell));
            int cy = std::min(n_rows_g - 1, (int)(frame->keypoints[j].pt.y / cell));
            if (cx >= 0 && cy >= 0) kp_grid[cy * n_cols_g + cx].push_back(j);
        }

        for (int mi = 0; mi < (int)pool_mps.size(); ++mi) {
            auto& mp = pool_mps[mi];
            Eigen::Vector3d Xc = frame->T_cw * mp->position;   // predicted T_cw
            if (Xc.z() <= 0.0) continue;
            float u = (float)(cam_.fx * Xc.x() / Xc.z() + cam_.cx);
            float v = (float)(cam_.fy * Xc.y() / Xc.z() + cam_.cy);
            if (u < 0 || u >= frame_w || v < 0 || v >= frame_h) continue;

            int cx0 = std::max(0,            (int)((u - search_r) / cell));
            int cx1 = std::min(n_cols_g - 1, (int)((u + search_r) / cell));
            int cy0 = std::max(0,            (int)((v - search_r) / cell));
            int cy1 = std::min(n_rows_g - 1, (int)((v + search_r) / cell));

            int best_j = -1, best_d = max_ham + 1;
            for (int gy = cy0; gy <= cy1; ++gy)
                for (int gx = cx0; gx <= cx1; ++gx)
                    for (int kp_j : kp_grid[gy * n_cols_g + gx]) {
                        if (used_kp.count(kp_j)) continue;
                        int d = cv::norm(pool_desc.row(mi),
                                         frame->descriptors.row(kp_j),
                                         cv::NORM_HAMMING);
                        if (d < best_d) { best_d = d; best_j = kp_j; }
                    }

            if (best_j < 0) continue;
            if (!used_kp.insert(best_j).second) continue;
            auto& p = mp->position;
            pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
            pts2d.push_back(frame->keypoints[best_j].pt);
            match_idxs.push_back(best_j);
            match_mps.push_back(mp);
        }
        std::cout << "[Tracker] Proj-match: " << pts3d.size()
                  << "/" << pool_mps.size() << " pts\n";
    }

    // phase 2: GPU Hamming fallback
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        pts3d.clear(); pts2d.clear(); match_idxs.clear(); match_mps.clear(); used_kp.clear();
        auto raw_matches = match_descriptors(pool_desc, frame->descriptors, /*use_ratio=*/true);
        for (auto& m : raw_matches) {
            if (!used_kp.insert(m.trainIdx).second) continue;
            auto& mp = pool_mps[m.queryIdx];
            auto& p  = mp->position;
            pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
            pts2d.push_back(frame->keypoints[m.trainIdx].pt);
            match_idxs.push_back(m.trainIdx);
            match_mps.push_back(mp);
        }
    }

    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: too few correspondences (" << pts3d.size() << ")\n";
        return false;
    }

    // PnP RANSAC with velocity-predicted pose as initial guess
    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F), inlier_mask;
    {
        Eigen::AngleAxisd   aa(frame->T_cw.rotation());
        Eigen::Vector3d     ax = aa.angle() * aa.axis();
        rvec.at<double>(0) = ax.x();
        rvec.at<double>(1) = ax.y();
        rvec.at<double>(2) = ax.z();
        tvec.at<double>(0) = frame->T_cw.translation().x();
        tvec.at<double>(1) = frame->T_cw.translation().y();
        tvec.at<double>(2) = frame->T_cw.translation().z();
    }
    // skip initial guess at sharp turns — overshoots bias RANSAC (threshold: 0.3 rad ≈ 17°)
    const double pred_angle = velocity_valid_
        ? Eigen::AngleAxisd(velocity_.rotation()).angle() : 0.0;
    const bool use_guess = velocity_valid_ && (pred_angle < 0.3);

    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/use_guess,
        cfg_.pnp_iterations, cfg_.pnp_reprojection, 0.99,
        inlier_mask, cv::SOLVEPNP_SQPNP);
    if (!ok) {
        std::cerr << "[Tracker] Track: solvePnPRansac failed (" << pts3d.size() << " corr)\n";
        return false;
    }

    // inlier mask can be CV_8U (mask) or CV_32S (index vector) depending on OpenCV version
    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_inliers; }
    } else {
        // index mode (CV_32S): each element is an inlier index into pts3d
        for (int k = 0; k < inlier_mask.rows; ++k) {
            int idx = inlier_mask.at<int>(k);
            if (idx >= 0 && idx < (int)pts3d.size()) { is_inlier[idx] = true; ++n_inliers; }
        }
    }
    if (n_inliers < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: PnP inliers too few (" << n_inliers << ")\n";
        return false;
    }

    // LM refinement on inliers
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, /*useExtrinsicGuess=*/true, cv::SOLVEPNP_ITERATIVE);

    // build candidate pose from rvec/tvec
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Isometry3d T_cw_candidate = Eigen::Isometry3d::Identity();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            T_cw_candidate.linear()(r, c) = R_cv.at<double>(r, c);
    T_cw_candidate.translation() << tvec.at<double>(0),
                                     tvec.at<double>(1),
                                     tvec.at<double>(2);

    // delta sanity checks — a car can't rotate >~29° or move >5 m between frames at 10 Hz
    {
        Eigen::Isometry3d delta = T_cw_candidate * last_frame_->T_cw.inverse();
        double delta_angle = Eigen::AngleAxisd(delta.rotation()).angle();
        if (delta_angle > 0.5) {    // 0.5 rad ≈ 29°
            std::cerr << "[Tracker] PnP rejected: delta rot "
                      << (delta_angle * (180.0 / 3.14159265358979323846)) << " deg\n";
            return false;
        }
        double delta_trans = delta.translation().norm();
        if (delta_trans > 5.0) {   // >5 m/frame ≈ 180 km/h at 10 Hz
            std::cerr << "[Tracker] PnP rejected: delta trans "
                      << delta_trans << " m\n";
            return false;
        }
    }

    // commit pose
    frame->T_cw = T_cw_candidate;

    // assign inlier map points
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    // project-and-search: pull in more map points after PnP without a second RANSAC pass
    {
        const int   cell      = 16;
        const int   frame_w   = frame->image_gray.cols;
        const int   frame_h   = frame->image_gray.rows;
        const int   n_cols_g  = (frame_w + cell - 1) / cell;
        const int   n_rows_g  = (frame_h + cell - 1) / cell;
        const float search_r  = 15.0f;
        const int   max_ham   = 50;
        const float max_repr2 = 25.0f;   // 5 px validation threshold²

        // spatial grid: only keypoints that still need a map point
        std::vector<std::vector<int>> kp_grid(n_cols_g * n_rows_g);
        for (int j = 0; j < (int)frame->keypoints.size(); ++j) {
            if (frame->map_points[j]) continue;
            if (used_kp.count(j)) continue;
            int cx = std::min(n_cols_g - 1, (int)(frame->keypoints[j].pt.x / cell));
            int cy = std::min(n_rows_g - 1, (int)(frame->keypoints[j].pt.y / cell));
            if (cx >= 0 && cy >= 0) kp_grid[cy * n_cols_g + cx].push_back(j);
        }

        // track which map-point IDs were already tried in the pool phase
        std::unordered_set<long> proj_seen;
        for (auto& mp : pool_mps) proj_seen.insert(mp->id);

        for (auto& kf : map_->local_window(30)) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!proj_seen.insert(mp->id).second) continue;  // already tried
                if (i >= kf->descriptors.rows) continue;

                // project into frame using committed T_cw
                Eigen::Vector3d Xc = frame->T_cw * mp->position;
                if (Xc.z() <= 0.0) continue;
                float u = (float)(cam_.fx * Xc.x() / Xc.z() + cam_.cx);
                float v = (float)(cam_.fy * Xc.y() / Xc.z() + cam_.cy);
                if (u < 0 || u >= frame_w || v < 0 || v >= frame_h) continue;

                // search grid cells in (u±search_r, v±search_r)
                int cx0 = std::max(0,            (int)((u - search_r) / cell));
                int cx1 = std::min(n_cols_g - 1, (int)((u + search_r) / cell));
                int cy0 = std::max(0,            (int)((v - search_r) / cell));
                int cy1 = std::min(n_rows_g - 1, (int)((v + search_r) / cell));

                int best_j = -1, best_d = max_ham;
                for (int gy = cy0; gy <= cy1; ++gy)
                    for (int gx = cx0; gx <= cx1; ++gx)
                        for (int kp_j : kp_grid[gy * n_cols_g + gx]) {
                            int d = cv::norm(kf->descriptors.row(i),
                                            frame->descriptors.row(kp_j),
                                            cv::NORM_HAMMING);
                            if (d < best_d) { best_d = d; best_j = kp_j; }
                        }

                if (best_j < 0 || used_kp.count(best_j)) continue;

                // validate with reprojection error at committed pose
                float du = u - frame->keypoints[best_j].pt.x;
                float dv = v - frame->keypoints[best_j].pt.y;
                if (du*du + dv*dv > max_repr2) continue;

                used_kp.insert(best_j);
                frame->map_points[best_j] = mp;
            }
        }
    }

    velocity_       = frame->T_cw * last_frame_->T_cw.inverse();
    velocity_valid_ = true;
    return true;
}

// local map tracking

bool Tracker::track_local_map(Frame::Ptr frame)
{
    if (need_new_keyframe(frame)) {
        insert_keyframe(frame);
    }
    return frame->num_tracked() >= cfg_.pnp_min_inliers;
}

// keyframe decision

bool Tracker::need_new_keyframe(Frame::Ptr frame) const
{
    if (!last_keyframe_) return true;

    int tracked = frame->num_tracked();  // PnP inliers only (not yet triangulated)

    // compare against last_kf_pnp_tracked_ (pre-triangulation count) to avoid inflated ratio
    if (tracked < cfg_.min_tracked_points) return true;
    if (last_kf_pnp_tracked_ > 0 && (float)tracked / last_kf_pnp_tracked_ < 0.8f) return true;

    return false;
}

double Tracker::compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const
{
    if (!ref_kf) return 0.0;
    const Eigen::Vector3d C_ref = ref_kf->camera_center();
    const Eigen::Vector3d C_cur = frame->camera_center();
    std::vector<double> angles;
    for (int i = 0; i < (int)frame->map_points.size(); ++i) {
        auto& mp = frame->map_points[i];
        if (!mp || mp->is_bad) continue;
        Eigen::Vector3d v1 = (mp->position - C_ref).normalized();
        Eigen::Vector3d v2 = (mp->position - C_cur).normalized();
        double cos_a = std::max(-1.0, std::min(1.0, v1.dot(v2)));
        angles.push_back(std::acos(std::abs(cos_a)));
    }
    if (angles.empty()) return 0.0;
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

void Tracker::insert_keyframe(Frame::Ptr frame)
{
    // save PnP inlier count BEFORE triangulation inflates frame->map_points
    last_kf_pnp_tracked_ = frame->num_tracked();

    frame->id = g_frame_id++;
    frame->is_keyframe = true;

    // triangulate against last 3 KFs (oldest first for largest baseline)
    for (auto& tri_kf : map_->local_window(3)) {
        if (tri_kf->id == frame->id) continue;
        auto kf_matches = match_descriptors(tri_kf->descriptors,
                                            frame->descriptors, /*ratio=*/true);
        std::vector<cv::DMatch> new_matches;
        for (auto& m : kf_matches) {
            if (m.trainIdx < (int)frame->map_points.size() &&
                !frame->map_points[m.trainIdx]) {
                new_matches.push_back(m);
            }
        }
        if (!new_matches.empty()) {
            int n_new = triangulate_and_add(tri_kf, frame, new_matches);
            if (n_new > 0)
                std::cout << "[Tracker] KF " << frame->id
                          << ": triangulated " << n_new
                          << " pts vs KF " << tri_kf->id << "\n";
        }
    }

    // stereo enrichment: fill unmapped keypoints with metric-depth points
    if (cam_.is_stereo() && !frame->uR.empty()) {
        int n_stereo = triangulate_stereo(frame);
        if (n_stereo > 0)
            std::cout << "[Tracker] KF " << frame->id
                      << ": stereo added " << n_stereo << " metric pts\n";
    }

    map_->insert_keyframe(frame);
    last_keyframe_ = frame;
}

// GPU descriptor matching

std::vector<cv::DMatch> Tracker::match_descriptors(
    const cv::Mat& query_desc,
    const cv::Mat& train_desc,
    bool use_ratio)
{
    int N_q = query_desc.rows;
    int N_t = train_desc.rows;

    if (N_q == 0 || N_t == 0) return {};

    // ensure descriptors are continuous and CV_8U
    cv::Mat q = query_desc.isContinuous() ? query_desc : query_desc.clone();
    cv::Mat t = train_desc.isContinuous()  ? train_desc  : train_desc.clone();

    std::vector<int> best_idx(N_q, -1);
    std::vector<int> best_dist(N_q, kMaxHamming);

    if (use_ratio) {
        cuda_match_hamming_ratio(
            q.data, t.data, N_q, N_t, cfg_.lowe_ratio,
            best_idx.data(), best_dist.data());
    } else {
        cuda_match_hamming(
            q.data, t.data, N_q, N_t,
            best_idx.data(), best_dist.data());
    }

    std::vector<cv::DMatch> matches;
    matches.reserve(N_q);
    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            matches.push_back(cv::DMatch(i, best_idx[i], (float)best_dist[i]));
        }
    }
    return matches;
}

// triangulation

int Tracker::triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                                  const std::vector<cv::DMatch>& matches)
{
    // build projection matrices (3×4)
    auto make_proj = [&](const Eigen::Isometry3d& T_cw) -> cv::Mat {
        cv::Mat P(3, 4, CV_64F);
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.block<3,3>(0,0) = T_cw.rotation();
        Rt.block<3,1>(0,3) = T_cw.translation();
        Eigen::Matrix<double, 3, 4> KRt = cam_.K() * Rt;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                P.at<double>(r, c) = KRt(r, c);
        return P;
    };

    cv::Mat P0 = make_proj(ref->T_cw);
    cv::Mat P1 = make_proj(cur->T_cw);

    // collect matched point pairs
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<int>         ref_kp_idxs, cur_kp_idxs;
    for (auto& m : matches) {
        pts0.push_back(ref->keypoints[m.queryIdx].pt);
        pts1.push_back(cur->keypoints[m.trainIdx].pt);
        ref_kp_idxs.push_back(m.queryIdx);
        cur_kp_idxs.push_back(m.trainIdx);
    }

    cv::Mat pts4d;
    cv::triangulatePoints(P0, P1, pts0, pts1, pts4d);  // 4×N homogeneous

    int n_added = 0;
    for (int i = 0; i < pts4d.cols; ++i) {
        float w = pts4d.at<float>(3, i);   // triangulatePoints outputs CV_32F
        if (std::abs(w) < 1e-6f) continue;

        Eigen::Vector3d Xw(pts4d.at<float>(0, i) / w,
                           pts4d.at<float>(1, i) / w,
                           pts4d.at<float>(2, i) / w);

        // depth check in both cameras
        Eigen::Vector3d Xc0 = ref->T_cw * Xw;
        Eigen::Vector3d Xc1 = cur->T_cw * Xw;
        if (Xc0.z() < 0.05 || Xc1.z() < 0.05) continue;
        if (Xc0.z() > 200.0 || Xc1.z() > 200.0) continue;

        // skip near-degenerate triangulations (cos_pa > 0.9998 ≈ <1.1° parallax)
        {
            Eigen::Vector3d O0 = ref->camera_center();
            Eigen::Vector3d O1 = cur->camera_center();
            double cos_pa = std::abs((Xw - O0).normalized().dot((Xw - O1).normalized()));
            if (cos_pa > 0.9998) continue;
        }

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->add_observation(ref->id, ref_kp_idxs[i]);
        mp->add_observation(cur->id, cur_kp_idxs[i]);

        ref->map_points[ref_kp_idxs[i]] = mp;
        cur->map_points[cur_kp_idxs[i]] = mp;

        map_->insert_map_point(mp);
        ++n_added;
    }
    return n_added;
}

// relocalization — match against all KFs with a stricter inlier threshold

bool Tracker::try_relocalize(Frame::Ptr frame)
{
    // build pool from ALL keyframes
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        for (auto& kf : map_->all_keyframes()) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!seen_ids.insert(mp->id).second) continue;
                if (i >= kf->descriptors.rows) continue;
                pool_desc.push_back(kf->descriptors.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if (pool_desc.rows < cfg_.pnp_min_inliers) return false;

    std::cout << "[Reloc] Matching against " << pool_desc.rows << " global map pts\n";

    // GPU descriptor matching
    auto raw_matches = match_descriptors(pool_desc, frame->descriptors, true);

    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    for (auto& m : raw_matches) {
        if (!used_kp.insert(m.trainIdx).second) continue;
        auto& mp = pool_mps[m.queryIdx];
        auto& p  = mp->position;
        pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
        pts2d.push_back(frame->keypoints[m.trainIdx].pt);
        match_idxs.push_back(m.trainIdx);
        match_mps.push_back(mp);
    }
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Reloc] Too few correspondences (" << pts3d.size() << ")\n";
        return false;
    }
    std::cout << "[Reloc] " << pts3d.size() << " 3D-2D correspondences\n";

    // PnP RANSAC — no initial guess (no valid velocity)
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inlier_mask;

    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/false,
        cfg_.pnp_iterations, cfg_.pnp_reprojection, 0.99,
        inlier_mask, cv::SOLVEPNP_SQPNP);

    if (!ok) { std::cerr << "[Reloc] solvePnPRansac failed\n"; return false; }

    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_reloc_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_reloc_inliers; }
    } else {
        for (int k = 0; k < inlier_mask.rows; ++k) {
            int idx = inlier_mask.at<int>(k);
            if (idx >= 0 && idx < (int)pts3d.size()) { is_inlier[idx] = true; ++n_reloc_inliers; }
        }
    }

    const int reloc_min = cfg_.pnp_min_inliers * 2;  // 30
    if (n_reloc_inliers < reloc_min) {
        std::cerr << "[Reloc] Inliers too few (" << n_reloc_inliers
                  << " < " << reloc_min << ")\n";
        return false;
    }

    // LM refinement on inliers
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    // update frame pose
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            frame->T_cw.linear()(r, c) = R_cv.at<double>(r, c);
    frame->T_cw.translation() << tvec.at<double>(0),
                                  tvec.at<double>(1),
                                  tvec.at<double>(2);

    // assign inlier map points
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    std::cout << "[Reloc] SUCCESS — " << inlier_mask.rows << " inliers\n";
    return true;
}

// stereo epipolar matching

void Tracker::match_stereo(Frame::Ptr frame)
{
    if (frame->descriptors.empty() || frame->descriptors_right.empty()) return;
    int N_q = frame->descriptors.rows;
    int N_t = frame->descriptors_right.rows;
    if (N_q == 0 || N_t == 0) return;

    std::vector<float> y_q(N_q), y_t(N_t), x_q(N_q), x_t(N_t);
    for (int i = 0; i < N_q; ++i) {
        y_q[i] = frame->keypoints[i].pt.y;
        x_q[i] = frame->keypoints[i].pt.x;
    }
    for (int i = 0; i < N_t; ++i) {
        y_t[i] = frame->keypoints_right[i].pt.y;
        x_t[i] = frame->keypoints_right[i].pt.x;
    }

    cv::Mat q = frame->descriptors.isContinuous()       ? frame->descriptors       : frame->descriptors.clone();
    cv::Mat t = frame->descriptors_right.isContinuous() ? frame->descriptors_right : frame->descriptors_right.clone();

    std::vector<int> best_idx(N_q, -1), best_dist(N_q, kMaxHamming);
    cuda_match_stereo_epipolar(
        q.data, t.data, N_q, N_t,
        y_q.data(), y_t.data(), x_q.data(), x_t.data(),
        cfg_.stereo_epi_tol, cfg_.stereo_d_min, cfg_.stereo_d_max,
        cfg_.lowe_ratio, best_idx.data(), best_dist.data());

    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            frame->uR[i] = frame->keypoints_right[best_idx[i]].pt.x;
        }
    }
}

// stereo triangulation

int Tracker::triangulate_stereo(Frame::Ptr frame)
{
    if (frame->uR.empty()) return 0;
    int n_added = 0;
    for (int i = 0; i < (int)frame->keypoints.size(); ++i) {
        if (frame->uR[i] < 0.0f) continue;      // no stereo match
        if (frame->map_points[i]) continue;       // already has a map point

        float u_L = frame->keypoints[i].pt.x;
        float v_L = frame->keypoints[i].pt.y;
        float u_R = frame->uR[i];
        float d   = u_L - u_R;
        if (d < cfg_.stereo_d_min || d > cfg_.stereo_d_max) continue;

        double Z = cam_.fx * cam_.baseline / (double)d;
        double X = ((double)u_L - cam_.cx) * Z / cam_.fx;
        double Y = ((double)v_L - cam_.cy) * Z / cam_.fy;
        if (Z < 0.5 || Z > 150.0) continue;

        Eigen::Vector3d Xw = frame->T_cw.inverse() * Eigen::Vector3d(X, Y, Z);

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->observed_times = 2;   // stereo = two-view constraint; treat as verified
        frame->map_points[i] = mp;
        map_->insert_map_point(mp);
        ++n_added;
    }
    return n_added;
}

// invalidate velocity after BA — inter-KF delta is a correction, not physical motion

void Tracker::notify_ba_update()
{
    velocity_valid_ = false;
}

}  // namespace slam
