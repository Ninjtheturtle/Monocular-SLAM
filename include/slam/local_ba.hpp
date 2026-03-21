#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <memory>
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>

namespace slam {

// --- marginalization prior ---
// dense quadratic prior from Schur complement marginalization of oldest KF
// E_prior = 0.5 * ||S*delta + e0||^2
// delta = stacked tangent-space deviation from linearization pts
// S^T S = H* (marginalized Hessian), e0 = S^{-T} b* (gradient offset, ~0 at optimum)
// FEJ (first-estimate Jacobians) used for consistency

struct MarginalizationInfo {
    bool valid = false;

    Eigen::MatrixXd S;   // total_dim x total_dim; H* = S^T S
    Eigen::VectorXd e0;  // total_dim; residual offset

    struct PoseBlock {
        long frame_id;
        int  offset;  // offset in stacked tangent vec (6-aligned)
        std::array<double, 7> x0;  // ambient linearization pt [qx,qy,qz,qw,tx,ty,tz]
    };
    std::vector<PoseBlock> poses;
    int total_dim = 0;  // = poses.size() * 6
};

// dynamic-sized cost function injecting the marg prior
// one 7-DOF pose param block per kept pose; residual dim = total_dim
// r = S * delta + e0; J = FEJ ambient: S_i * J_pinv(x0_i)
class MarginalizationPriorCost : public ceres::CostFunction {
public:
    explicit MarginalizationPriorCost(const MarginalizationInfo& info);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

private:
    MarginalizationInfo info_;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 7>> J_amb_;  // precomputed FEJ per block
};

// --- LocalBA ---
// sliding-window BA w/ Ceres
// optimizes last N KF poses + all visible map pts
// when window is full, oldest KF is marginalized via Schur complement
// cost: stereo/mono reproj w/ Huber + marg prior
// parameterization: 7-DOF [qx,qy,qz,qw,tx,ty,tz] w/ EigenQuaternionManifold
// analytical Jacobians for reproj; FEJ for prior

class LocalBA {
public:
    struct Config {
        int    max_iterations    = 60;
        int    window_size       = Map::kWindowSize;
        double huber_delta       = 1.5;   // Huber kernel threshold (px)
        double sigma_px          = 1.0;   // pixel noise sigma
        double z_ref             = 15.0;  // depth ref for info attenuation (m)
        bool   verbose           = false;
    };

    using Ptr = std::shared_ptr<LocalBA>;
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    void optimize();  // run one BA iteration on current window

private:
    Camera   cam_;
    Map::Ptr map_;
    Config   cfg_;

    MarginalizationInfo marg_info_;  // prior from previously dropped KF

    // build marg prior after optimization; eliminates oldest KF from reduced Hessian
    void compute_marginalization_prior(
        const std::vector<Frame::Ptr>& window,
        const std::unordered_map<long, std::vector<double>>& pose_params,
        const std::unordered_map<long, std::array<double, 3>>& point_params);

    LocalBA() = default;
};

// --- analytical cost functions (quaternion parameterized) ---
// pose block: 7 doubles [qx,qy,qz,qw,tx,ty,tz] — Eigen storage (x,y,z,w)
// point block: 3 doubles [X,Y,Z] (world frame)
// hand-derived Jacobians from explicit R(q) formula
// dR/dq_i computed treating all 4 quat components as independent (ambient derivative)
// Ceres EigenQuaternionManifold projects onto 3-DOF tangent plane
//
// stereo info weighting:
//   Omega_uv = 1/sigma^2 (bearing — constant)
//   Omega_d  = Omega_uv * min(1, Z_ref^2/Z^2) (disparity — attenuated at depth)
//   residuals premultiplied by sqrt(Omega_i) -> Ceres minimizes sum(r_i^2) = Mahalanobis

// stereo: 3 residuals (u_L, v_L, u_R)
class StereoReprojCost final : public ceres::SizedCostFunction<3, 7, 3> {
public:
    StereoReprojCost(double obs_uL, double obs_vL, double obs_uR,
                     double fx, double fy, double cx, double cy,
                     double b, double info_uv, double info_disp);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

    static ceres::CostFunction* Create(double obs_uL, double obs_vL, double obs_uR,
                                        double fx, double fy, double cx, double cy,
                                        double b, double info_uv, double info_disp) {
        return new StereoReprojCost(obs_uL, obs_vL, obs_uR, fx, fy, cx, cy, b,
                                    info_uv, info_disp);
    }

private:
    double obs_uL_, obs_vL_, obs_uR_;
    double fx_, fy_, cx_, cy_, b_;
    double sqrt_info_uv_;  // sqrt(Omega) for bearing (rows 0,1)
    double sqrt_info_d_;   // sqrt(Omega) for disparity (row 2)
};

// mono: 2 residuals (u, v) — used when uR < 0 (no stereo match)
class MonoReprojCost final : public ceres::SizedCostFunction<2, 7, 3> {
public:
    MonoReprojCost(double obs_u, double obs_v,
                   double fx, double fy, double cx, double cy,
                   double info_uv);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

    static ceres::CostFunction* Create(double obs_u, double obs_v,
                                        double fx, double fy, double cx, double cy,
                                        double info_uv) {
        return new MonoReprojCost(obs_u, obs_v, fx, fy, cx, cy, info_uv);
    }

private:
    double obs_u_, obs_v_;
    double fx_, fy_, cx_, cy_;
    double sqrt_info_uv_;
};

}  // namespace slam
