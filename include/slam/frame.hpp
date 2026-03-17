#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <atomic>
#include <cuda_fp16.h>

namespace slam {

class MapPoint;

// a single processed image frame
// pose convention: T_cw transforms world -> camera: X_c = R*X_w + t
// stored as Eigen::Isometry3d (4x4 SE3)
class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    static Ptr create(const cv::Mat& image, double timestamp, long id);

    long   id;
    double timestamp;
    cv::Mat image_gray;  // grayscale 8U

    // --- left image features ---
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;  // Nx32 CV_8U (ORB); dummy in hybrid mode

    // --- right image (stereo; empty if mono) ---
    cv::Mat                   image_right;
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat                   descriptors_right;

    std::vector<float> uR;  // right x-coord per left kp (-1 = no match)

    // map associations (one per kp; nullptr = unmatched)
    std::vector<std::shared_ptr<MapPoint>> map_points;

    Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();  // world -> camera

    bool is_keyframe = false;

    // --- deep frontend fields (hybrid XFeat mode) ---

    cv::Mat xfeat_descriptors;  // Nx64 CV_32F (promoted from FP16 for Ceres)

    // per-kp confidence [0.1, 1.0] from L2 ratio or LG probability
    // kept for future per-observation weighting; BA currently uses depth-based info
    std::vector<float> match_confidence;

    // XFeat feat maps at 1/8 res (CHW float32, host memory)
    // populated on KFs only for semi-dense disparity
    cv::Mat feat_map_left;
    cv::Mat feat_map_right;

    // --- helpers ---
    Eigen::Isometry3d T_wc() const { return T_cw.inverse(); }
    Eigen::Vector3d camera_center() const { return T_wc().translation(); }
    const uint8_t* desc_ptr() const { return descriptors.data; }
    int num_features() const { return static_cast<int>(keypoints.size()); }
    int num_tracked() const;  // count kps w/ valid map point

private:
    Frame() = default;
};

}  // namespace slam
