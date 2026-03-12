#pragma once

#include "slam/frame.hpp"
#include "slam/map_point.hpp"
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>

namespace slam {

/// thread-safe container for all active map state:
/// keyframes (ordered + indexed by id), 3D map points, and a sliding window of
/// the N most recent KFs for local BA.
class Map {
public:
    static constexpr int kWindowSize = 30;  // sliding window for local BA

    using Ptr = std::shared_ptr<Map>;
    static Ptr create() { return Ptr(new Map()); }

    // keyframe management
    void insert_keyframe(Frame::Ptr kf);
    void remove_keyframe(long id);
    Frame::Ptr get_keyframe(long id) const;

    /// ordered (oldest first) list of all keyframes
    std::vector<Frame::Ptr> all_keyframes() const;

    /// last kWindowSize keyframes (or all if fewer) — used for BA
    std::vector<Frame::Ptr> local_window() const;

    /// last n keyframes (or all if fewer) — used for the tracking descriptor pool
    std::vector<Frame::Ptr> local_window(int n) const;

    // map point management
    void insert_map_point(MapPoint::Ptr mp);
    void remove_map_point(long id);
    MapPoint::Ptr get_map_point(long id) const;

    /// all active (non-bad) map points
    std::vector<MapPoint::Ptr> all_map_points() const;

    /// remove map points flagged as bad
    void cleanup_bad_map_points();

    /// clear all KFs and map points; archives current KFs into trajectory_archive_ first
    void reset();

    /// KFs archived from all prior map resets — for trajectory visualization only,
    /// never included in BA or the tracking pool.
    std::vector<Frame::Ptr> trajectory_archive() const;

    /// count map points observed by both kf_id_a and kf_id_b
    int count_shared_map_points(long kf_id_a, long kf_id_b) const;

    // statistics
    size_t num_keyframes()  const;
    size_t num_map_points() const;

private:
    Map() = default;

    mutable std::mutex kf_mutex_;
    mutable std::mutex mp_mutex_;

    // ordered by insertion
    std::deque<Frame::Ptr>                         keyframe_order_;
    std::unordered_map<long, Frame::Ptr>            keyframes_;
    std::unordered_map<long, MapPoint::Ptr>         map_points_;

    // not cleared by reset() — accumulates KFs from all prior map segments
    std::vector<Frame::Ptr> trajectory_archive_;
};

}  // namespace slam
