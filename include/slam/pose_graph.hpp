#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace slam {

// PGO over all KFs
// detects loop candidates via co-vis (shared map pts) between newest KF
// and KFs outside the local BA window; adds relative-pose edges & runs
// lightweight Ceres PGO to correct drift beyond BA window
class PoseGraph {
public:
    using Ptr = std::shared_ptr<PoseGraph>;

    struct Config {
        int  min_shared_points = 15;   // min shared pts for covis edge
        int  pgo_interval      = 5;    // run PGO every N new KFs
        int  max_iterations    = 30;
        double w_t             = 10.0; // translation residual weight
        double w_r             = 50.0; // rotation residual weight
    };

    static Ptr create(Map::Ptr map, const Camera& cam, const Config& cfg = Config{});

    void add_keyframe(Frame::Ptr kf);   // register new KF (no detection/PGO)
    void detect_and_add_loops();         // scan for covis edges outside BA window
    bool has_new_loops() const { return new_loops_; }
    void optimize();                     // Ceres PGO over all KF poses -> write back T_cw
    int num_edges() const { return static_cast<int>(edges_.size()); }

private:
    struct Edge {
        long id_a, id_b;
        double R_meas[9];  // row-major rotation (measured relative transform)
        double t_meas[3];
    };

    Camera     cam_;
    Map::Ptr   map_;
    Config     cfg_;
    std::vector<Frame::Ptr> kf_order_;  // insertion order — used for gauge anchor
    std::vector<Edge>       edges_;
    bool new_loops_ = false;

    PoseGraph() = default;
};

}  // namespace slam
