#include "slam/map.hpp"
#include <algorithm>

namespace slam {

// ─── Keyframe management ─────────────────────────────────────────────────────

void Map::insert_keyframe(Frame::Ptr kf)
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    kf->is_keyframe = true;
    keyframes_[kf->id] = kf;
    keyframe_order_.push_back(kf);
}

void Map::remove_keyframe(long id)
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    keyframes_.erase(id);
    auto it = std::find_if(keyframe_order_.begin(), keyframe_order_.end(),
                           [id](const Frame::Ptr& f) { return f->id == id; });
    if (it != keyframe_order_.end()) keyframe_order_.erase(it);
}

Frame::Ptr Map::get_keyframe(long id) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    auto it = keyframes_.find(id);
    return (it != keyframes_.end()) ? it->second : nullptr;
}

std::vector<Frame::Ptr> Map::all_keyframes() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    return std::vector<Frame::Ptr>(keyframe_order_.begin(), keyframe_order_.end());
}

std::vector<Frame::Ptr> Map::local_window() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    int n = static_cast<int>(keyframe_order_.size());
    int start = std::max(0, n - kWindowSize);
    return std::vector<Frame::Ptr>(
        keyframe_order_.begin() + start, keyframe_order_.end());
}

std::vector<Frame::Ptr> Map::local_window(int size) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    int n = static_cast<int>(keyframe_order_.size());
    int start = std::max(0, n - size);
    return std::vector<Frame::Ptr>(
        keyframe_order_.begin() + start, keyframe_order_.end());
}

// ─── Map point management ─────────────────────────────────────────────────────

void Map::insert_map_point(MapPoint::Ptr mp)
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    map_points_[mp->id] = mp;
}

void Map::remove_map_point(long id)
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    map_points_.erase(id);
}

MapPoint::Ptr Map::get_map_point(long id) const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    auto it = map_points_.find(id);
    return (it != map_points_.end()) ? it->second : nullptr;
}

std::vector<MapPoint::Ptr> Map::all_map_points() const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    std::vector<MapPoint::Ptr> out;
    out.reserve(map_points_.size());
    for (auto& [id, mp] : map_points_) {
        if (!mp->is_bad) out.push_back(mp);
    }
    return out;
}

void Map::cleanup_bad_map_points()
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    for (auto it = map_points_.begin(); it != map_points_.end(); ) {
        if (it->second->is_bad) it = map_points_.erase(it);
        else                    ++it;
    }
}

size_t Map::num_keyframes() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    return keyframes_.size();
}

size_t Map::num_map_points() const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    return map_points_.size();
}

}  // namespace slam
