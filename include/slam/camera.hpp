#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <string>

namespace slam {

/// pinhole camera model with optional radial distortion.
struct Camera {
    double fx, fy;  // focal lengths (pixels)
    double cx, cy;  // principal point (pixels)
    double k1, k2;  // radial distortion (0 if undistorted)
    int    width, height;
    double baseline = 0.0;  // stereo baseline in metres (0 = monocular)
    bool is_stereo() const { return baseline > 0.0; }

    Camera() = default;
    Camera(double fx, double fy, double cx, double cy,
           int width = 0, int height = 0,
           double k1 = 0.0, double k2 = 0.0)
        : fx(fx), fy(fy), cx(cx), cy(cy),
          k1(k1), k2(k2), width(width), height(height) {}

    /// 3×3 intrinsic matrix K
    Eigen::Matrix3d K() const {
        Eigen::Matrix3d mat;
        mat << fx,  0, cx,
                0, fy, cy,
                0,  0,  1;
        return mat;
    }

    /// OpenCV intrinsic matrix (for solvePnP etc.)
    cv::Mat K_cv() const {
        cv::Mat mat = (cv::Mat_<double>(3, 3)
            << fx,  0, cx,
                0, fy, cy,
                0,  0,  1);
        return mat;
    }

    /// distortion coefficients [k1, k2, 0, 0] for OpenCV
    cv::Mat dist_cv() const {
        return (cv::Mat_<double>(1, 4) << k1, k2, 0.0, 0.0);
    }

    /// project a 3D point (camera frame) to image plane
    Eigen::Vector2d project(const Eigen::Vector3d& Xc) const {
        return {fx * Xc(0) / Xc(2) + cx,
                fy * Xc(1) / Xc(2) + cy};
    }

    /// back-project a pixel to a unit bearing vector (camera frame)
    Eigen::Vector3d unproject(double u, double v) const {
        return Eigen::Vector3d((u - cx) / fx, (v - cy) / fy, 1.0).normalized();
    }

    /// parse KITTI calib.txt and return the P0 camera parameters
    static Camera from_kitti_calib(const std::string& calib_file);
};

}  // namespace slam
