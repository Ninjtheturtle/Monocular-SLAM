// visualizer.cpp
//
// Rerun.io C++ SDK logging for SLAM state.
//
// Rerun entity hierarchy:
//   world/
//     camera/
//       image        — 2D image panel (via Pinhole set in log_pinhole)
//       keypoints    — purple 2D keypoint dots in the image panel
//     map/
//       points       — Points3D (active map point cloud)
//       trajectory   — LineStrips3D (camera centre history)

#include "slam/visualizer.hpp"

#include <rerun.hpp>
#include <rerun/archetypes/image.hpp>
#include <rerun/archetypes/pinhole.hpp>
#include <rerun/archetypes/points2d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/view_coordinates.hpp>
#include <rerun/components/color.hpp>

#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <iostream>
#include <vector>

namespace slam {

// ─── Factory ─────────────────────────────────────────────────────────────────

Visualizer::Ptr Visualizer::create(const Config& cfg)
{
    auto v = std::shared_ptr<Visualizer>(new Visualizer());
    v->cfg_ = cfg;

    v->rec_ = std::make_unique<rerun::RecordingStream>(cfg.app_id);
    auto result = v->rec_->connect_tcp(cfg.addr);
    if (!result.is_ok()) {
        std::cerr << "[Visualizer] Warning: could not connect to Rerun at "
                  << cfg.addr << " — " << result.description << "\n"
                  << "  Start the viewer with:  rerun\n";
    } else {
        std::cout << "[Visualizer] Connected to Rerun at " << cfg.addr << "\n";
    }

    // Tell Rerun that the camera space uses RDF convention: x=Right, y=Down, z=Forward.
    // Logging to world/camera (the 3D view root auto-created by Rerun for the Pinhole)
    // so the coordinate axes are correct in the 3D panel.
    v->rec_->log_static("world/camera", rerun::archetypes::ViewCoordinates::RDF);

    return v;
}

Visualizer::~Visualizer() = default;

// ─── log_pinhole ─────────────────────────────────────────────────────────────

void Visualizer::log_pinhole(const Camera& cam)
{
    if (!rec_) return;
    cam_ = cam;  // store for depth projection in log_frame()

    // Log Pinhole intrinsics to the image entity so Rerun renders:
    //   • a proper camera frustum in the 3D view
    //   • a 2D image panel showing the camera feed
    rec_->log("world/camera/image",
        rerun::archetypes::Pinhole::from_focal_length_and_resolution(
            {(float)cam.fx, (float)cam.fy},
            {(float)cam.width, (float)cam.height}
        )
    );
}

// ─── log_frame ───────────────────────────────────────────────────────────────

void Visualizer::log_frame(const Frame::Ptr& frame)
{
    if (!rec_) return;

    // Advance Rerun timeline so each frame's data is time-stamped and the
    // viewer can scrub through the sequence.
    rec_->set_time_seconds("time", frame->timestamp);

    // ── Camera image → 2D panel ───────────────────────────────────────────────
    // Log to world/camera/image. Rerun creates a dedicated 2D panel for any
    // entity that has a Pinhole ancestor (set once in log_pinhole()).
    // No Transform3D is logged here, so no camera frustum appears in the 3D view.
    if (cfg_.log_image && !frame->image_gray.empty()) {
        cv::Mat rgb;
        cv::cvtColor(frame->image_gray, rgb, cv::COLOR_GRAY2RGB);
        auto bytes = std::vector<uint8_t>(rgb.data, rgb.data + rgb.total() * 3);
        rec_->log("world/camera/image",
            rerun::archetypes::Image::from_rgb24(
                std::move(bytes), {(uint32_t)rgb.cols, (uint32_t)rgb.rows}));
    }

    // ── Purple keypoints in the 2D panel ─────────────────────────────────────
    if (cfg_.log_keypoints && !frame->keypoints.empty()) {
        std::vector<rerun::datatypes::Vec2D> pts;
        pts.reserve(frame->keypoints.size());
        for (auto& kp : frame->keypoints)
            pts.push_back({kp.pt.x, kp.pt.y});
        rec_->log("world/camera/keypoints",
            rerun::archetypes::Points2D(pts)
                .with_colors(rerun::components::Color(190, 75, 230, 220))
                .with_radii(std::vector<float>(pts.size(), 2.5f)));
    }

    // ── Trajectory (camera centre in world space) ─────────────────────────────
    // Only append when the frame has a valid tracked pose (num_tracked > 0).
    // Untracked frames (NOT_INITIALIZED / LOST) have T_cw = Identity, which
    // would log (0,0,0) and make the trajectory appear to stall at the origin.
    if (frame->num_tracked() > 0) {
        Eigen::Vector3d t = frame->T_wc().translation();

        if (strips_.empty()) strips_.emplace_back();
        strips_.back().push_back({(float)t.x(), (float)t.y(), (float)t.z()});

        // Collect all segments; include single-point segments so the Rerun
        // entity is created on the very first tracked frame (size < 2 hid it).
        std::vector<rerun::components::LineStrip3D> all_strips;
        for (auto& seg : strips_) {
            if (seg.empty()) continue;
            std::vector<rerun::datatypes::Vec3D> pts3;
            pts3.reserve(seg.size());
            for (auto& p : seg) pts3.push_back({p[0], p[1], p[2]});
            all_strips.emplace_back(pts3);
        }
        if (!all_strips.empty()) {
            rec_->log("world/trajectory",
                rerun::archetypes::LineStrips3D(all_strips));
        }
    }
}

// ─── new_trajectory_segment ──────────────────────────────────────────────────

void Visualizer::new_trajectory_segment()
{
    strips_.emplace_back();  // next log_frame() appends to a fresh strip
}

// ─── log_map ─────────────────────────────────────────────────────────────────

void Visualizer::log_map(const Map::Ptr& map, double timestamp)
{
    if (!rec_) return;
    rec_->set_time_seconds("time", timestamp);

    auto map_pts = map->all_map_points();
    if (map_pts.empty()) return;

    std::vector<rerun::datatypes::Vec3D> positions;
    positions.reserve(map_pts.size());

    for (auto& mp : map_pts) {
        auto& p = mp->position;
        positions.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
    }

    rec_->log("world/map/points",
        rerun::archetypes::Points3D(positions)
            .with_radii(std::vector<float>(positions.size(), 0.03f))
    );
}

// ─── log_ground_truth ─────────────────────────────────────────────────────────

void Visualizer::log_ground_truth(const std::vector<std::array<float, 3>>& centers)
{
    if (!rec_ || centers.size() < 2) return;

    std::vector<rerun::datatypes::Vec3D> pts;
    pts.reserve(centers.size());
    for (auto& c : centers)
        pts.push_back({c[0], c[1], c[2]});

    rec_->log_static("world/ground_truth/trajectory",
        rerun::archetypes::LineStrips3D(
            {rerun::components::LineStrip3D(pts)})
            .with_colors({rerun::components::Color(255, 165, 0)}));  // orange
}

}  // namespace slam
