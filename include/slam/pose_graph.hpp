#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace slam {

/// pose graph optimization (PGO) over all keyframes.
/// detects loop candidates via co-visibility (shared map points) between the newest KF
/// and KFs outside the local BA window. adds relative-pose edges and runs a lightweight
/// Ceres PGO to correct drift that accumulates beyond the BA window.
class PoseGraph {
public:
    using Ptr = std::shared_ptr<PoseGraph>;

    struct Config {
        int  min_shared_points = 15;   // min shared map points for a co-visibility edge
        int  pgo_interval      = 5;    // run PGO every N new KFs
        int  max_iterations    = 30;   // Ceres iterations for PGO solve
        double w_t             = 10.0; // translation residual weight
        double w_r             = 50.0; // rotation residual weight
        int  visual_min_inliers = 20;  // PnP inliers required to accept a visual loop edge
        int  visual_sample_step = 10;  // sample every Nth KF for visual loop search
    };

    static Ptr create(Map::Ptr map, const Camera& cam, const Config& cfg = Config{});

    /// register a newly inserted KF (does not run detection or PGO)
    void add_keyframe(Frame::Ptr kf);

    /// scan KFs outside the BA window for co-visibility edges with the latest KF
    void detect_and_add_loops();

    /// appearance-based loop detection (currently disabled — see pose_graph.cpp)
    void detect_and_add_loops_visual(Frame::Ptr query);

    /// true if the last detect_and_add_loops() call found at least one new edge
    bool has_new_loops() const { return new_loops_; }

    /// run Ceres PGO over all KF poses and write back optimized T_cw
    void optimize();

    int num_edges() const { return static_cast<int>(edges_.size()); }

private:
    struct Edge {
        long id_a, id_b;
        // measured relative transform: T_ab = T_a_cw * T_b_cw.inverse()
        double R_meas[9];  // row-major rotation
        double t_meas[3];  // translation
    };

    Camera     cam_;
    Map::Ptr   map_;
    Config     cfg_;
    std::vector<Frame::Ptr> kf_order_;  // insertion order, mirrors map — used for gauge anchor
    std::vector<Edge>       edges_;
    bool new_loops_ = false;

    PoseGraph() = default;
};

}  // namespace slam
