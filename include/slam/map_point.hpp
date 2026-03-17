#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <atomic>

namespace slam {

class Frame;

// a 3D landmark: world-frame position, descriptor, & observation list
class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;

    static Ptr create(const Eigen::Vector3d& position, long id);

    long id;
    bool is_bad = false;  // flagged for removal

    Eigen::Vector3d position;    // X_w
    cv::Mat         descriptor;  // representative 32-byte ORB desc

    int observed_times  = 0;  // total observation count
    int visible_times   = 0;  // times visible in a frame (for match ratio culling)

    // observations: frame_id -> kp index
    std::map<long, int> observations;
    mutable std::mutex  obs_mutex;

    void add_observation(long frame_id, int kp_idx);
    void remove_observation(long frame_id);
    int  get_keypoint_idx(long frame_id) const;
    int  num_observations() const;

    // recompute representative desc from all current observations
    void update_descriptor(
        const std::vector<std::shared_ptr<Frame>>& frames);

private:
    MapPoint() = default;
    static std::atomic<long> next_id_;
};

}  // namespace slam
