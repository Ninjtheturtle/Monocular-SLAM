#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

// fwd-declare to avoid pulling deep headers into every TU
namespace deep { struct SemiDensePoint3D; }
namespace rerun { class RecordingStream; }

namespace slam {

// logs SLAM state to Rerun.io via TCP (default localhost:9876)
// call log_frame() every frame, log_map() after each BA
class Visualizer {
public:
    struct Config {
        std::string app_id   = "vslam2";
        std::string addr     = "127.0.0.1:9876";
        bool log_image       = true;
        bool log_keypoints   = true;
    };

    using Ptr = std::shared_ptr<Visualizer>;
    static Ptr create(const Config& cfg = Config{});

    ~Visualizer();

    // log camera intrinsics once so rerun renders frustum + image panel; call before log_frame()
    void log_pinhole(const Camera& cam);

    void log_frame(const Frame::Ptr& frame);  // image, pose, 2D kps
    void log_map(const Map::Ptr& map, double timestamp = 0.0);  // 3D point cloud

    // rebuilt every call from BA-refined KF poses — auto-reflects corrections
    void log_trajectory(const Map::Ptr& map,
                        const Frame::Ptr& current_frame,
                        double ts);

    // static orange GT trajectory — call once before main loop w/ KITTI poses
    void log_ground_truth(const std::vector<std::array<float, 3>>& centers);

    // semi-dense point cloud viz (entity: world/map/semi_dense, not in Ceres)
    void log_semi_dense(const std::vector<deep::SemiDensePoint3D>& pts, double ts);

private:
    Visualizer() = default;

    Config cfg_;
    Camera cam_;
    std::unique_ptr<rerun::RecordingStream> rec_;
};

}  // namespace slam
