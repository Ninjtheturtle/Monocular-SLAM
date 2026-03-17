#pragma once

#include "slam/frame.hpp"
#include "slam/map_point.hpp"
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>

namespace slam {

// thread-safe container for active map state:
// KFs (ordered + indexed by id), 3D map pts, sliding window for local BA
class Map {
public:
    static constexpr int kWindowSize = 30;  // BA sliding window size

    using Ptr = std::shared_ptr<Map>;
    static Ptr create() { return Ptr(new Map()); }

    // --- kf management ---
    void insert_keyframe(Frame::Ptr kf);
    void remove_keyframe(long id);
    Frame::Ptr get_keyframe(long id) const;
    std::vector<Frame::Ptr> all_keyframes() const;  // oldest first
    std::vector<Frame::Ptr> local_window() const;    // last kWindowSize KFs
    std::vector<Frame::Ptr> local_window(int n) const;

    // --- map point management ---
    void insert_map_point(MapPoint::Ptr mp);
    void remove_map_point(long id);
    MapPoint::Ptr get_map_point(long id) const;
    std::vector<MapPoint::Ptr> all_map_points() const;  // active (non-bad) only
    void cleanup_bad_map_points();

    // clear all KFs & pts; archives current KFs into trajectory_archive_ first
    void reset();

    // KFs archived from prior resets — for trajectory viz only, never in BA/tracking
    std::vector<Frame::Ptr> trajectory_archive() const;

    int count_shared_map_points(long kf_id_a, long kf_id_b) const;

    // --- co-visibility graph ---
    void update_covisibility(long kf_id, MapPoint::Ptr mp);

    // top-N most co-visible KFs to kf_id, sorted by shared pt count (descending)
    std::vector<std::pair<Frame::Ptr, int>> get_covisible_keyframes(
        long kf_id, int top_n = 20) const;

    // KFs co-visible w/ kf_id having >= min_shared shared pts
    std::vector<Frame::Ptr> get_covisible_keyframes_above(
        long kf_id, int min_shared) const;

    size_t num_keyframes()  const;
    size_t num_map_points() const;

private:
    Map() = default;

    mutable std::mutex kf_mutex_;
    mutable std::mutex mp_mutex_;
    mutable std::mutex covis_mutex_;

    std::deque<Frame::Ptr>                         keyframe_order_;
    std::unordered_map<long, Frame::Ptr>            keyframes_;
    std::unordered_map<long, MapPoint::Ptr>         map_points_;

    // covis_[kf_a][kf_b] = shared point count
    std::unordered_map<long, std::unordered_map<long, int>> covis_;

    // never cleared by reset() — accumulates KFs from all prior map segments
    std::vector<Frame::Ptr> trajectory_archive_;
};

}  // namespace slam
