// sliding-window bundle adjustment using Ceres (analytical Jacobians, SPARSE_SCHUR).

#include "slam/local_ba.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>  // AngleAxisRotatePoint, AngleAxisToRotationMatrix

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace slam {

// monocular reprojection cost (analytical Jacobians)

bool ReprojectionCost::operator()(const double* const pose,   // [ω₀,ω₁,ω₂, t₀,t₁,t₂]
                                  const double* const point,  // [X,Y,Z] world
                                  double* residuals, double** jacobians) const {
    // transform world point to camera frame: X_c = R(ω)*X_w + t
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    // residuals: projected - observed
    const double u_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_proj = fy * Xc[1] * inv_Zc + cy;
    residuals[0] = u_proj - u_obs;
    residuals[1] = v_proj - v_obs;

    if (!jacobians) return true;

    // J_proj rows: [fx/Zc, 0, -fx*Xc/Zc²] and [0, fy/Zc, -fy*Yc/Zc²]
    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;

    // Jacobian w.r.t. pose (2×6)
    if (jacobians[0]) {
        // ∂X_c/∂ω = -[X_c]×  (linearized around current ω)
        // dXc_dw = [ 0, Zc, -Yc ; -Zc, 0, Xc ; Yc, -Xc, 0 ]
        const double dXc_dw[3][3] = {
            {0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

        // ∂r/∂ω = J_proj * ∂X_c/∂ω  (2×3)
        double* j_pose = jacobians[0];  // row-major 2×6

        // column 0 (dω₀)
        j_pose[0] = jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0];  // ∂u/∂ω₀
        j_pose[6] = jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0];  // ∂v/∂ω₀
        // column 1 (dω₁)
        j_pose[1] = jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1];  // ∂u/∂ω₁
        j_pose[7] = jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1];  // ∂v/∂ω₁
        // column 2 (dω₂)
        j_pose[2] = jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2];  // ∂u/∂ω₂
        j_pose[8] = jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2];  // ∂v/∂ω₂

        // ∂X_c/∂t = I₃  → ∂r/∂t = J_proj (columns 3,4,5)
        j_pose[3] = jp00;   // ∂u/∂t₀
        j_pose[4] = 0.0;    // ∂u/∂t₁
        j_pose[5] = jp02;   // ∂u/∂t₂
        j_pose[9] = 0.0;    // ∂v/∂t₀
        j_pose[10] = jp11;  // ∂v/∂t₁
        j_pose[11] = jp12;  // ∂v/∂t₂
    }

    // Jacobian w.r.t. 3D point X_w (2×3): ∂r/∂X_w = J_proj * R
    if (jacobians[1]) {
        // ∂X_c/∂X_w = R(ω)
        double R[9];  // row-major 3×3
        ceres::AngleAxisToRotationMatrix(pose, R);

        double* j_point = jacobians[1];  // row-major 2×3

        // row 0 (∂u/∂X_w)
        j_point[0] = jp00 * R[0] + jp02 * R[6];  // ∂u/∂X
        j_point[1] = jp00 * R[1] + jp02 * R[7];  // ∂u/∂Y
        j_point[2] = jp00 * R[2] + jp02 * R[8];  // ∂u/∂Z
        // row 1 (∂v/∂X_w)
        j_point[3] = jp11 * R[3] + jp12 * R[6];  // ∂v/∂X
        j_point[4] = jp11 * R[4] + jp12 * R[7];  // ∂v/∂Y
        j_point[5] = jp11 * R[5] + jp12 * R[8];  // ∂v/∂Z
    }

    return true;
}

// Ceres wrapper adapting ReprojectionCost to SizedCostFunction interface

class AnalyticReprojectionCostFunction
    : public ceres::SizedCostFunction<ReprojectionCost::kNumResiduals,    // 2 residuals
                                      ReprojectionCost::kNumPoseParams,   // 6 pose params
                                      ReprojectionCost::kNumPointParams>  // 3 point params
{
   public:
    explicit AnalyticReprojectionCostFunction(const ReprojectionCost& cost) : cost_(cost) {}

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }

   private:
    ReprojectionCost cost_;
};

// stereo reprojection cost — 3 residuals: [u_L, v_L, u_R]. row 2 uses the right-camera projection.

bool StereoReprojectionCost::operator()(const double* const pose, const double* const point,
                                        double* residuals, double** jacobians) const {
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    const double u_L_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_L_proj = fy * Xc[1] * inv_Zc + cy;
    const double u_R_proj = fx * (Xc[0] - baseline) * inv_Zc + cx;

    residuals[0] = u_L_proj - u_L_obs;
    residuals[1] = v_L_proj - v_L_obs;
    residuals[2] = u_R_proj - u_R_obs;

    if (!jacobians) return true;

    // shared projection Jacobian terms
    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;
    // right-camera u Jacobian: [ fx/Zc, 0, -fx*(Xc[0]-b)/Zc² ]
    const double jr02 = -fx * (Xc[0] - baseline) * inv_Zc2;

    const double dXc_dw[3][3] = {{0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

    if (jacobians[0]) {
        double* j = jacobians[0];  // row-major 3×6
        // row 0 (∂u_L/∂pose) — same as monocular
        j[0] = jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0];
        j[1] = jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1];
        j[2] = jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2];
        j[3] = jp00;
        j[4] = 0.0;
        j[5] = jp02;
        // row 1 (∂v_L/∂pose)
        j[6] = jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0];
        j[7] = jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1];
        j[8] = jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2];
        j[9] = 0.0;
        j[10] = jp11;
        j[11] = jp12;
        // row 2 (∂u_R/∂pose) — same ∂X_c/∂ω, but right-camera proj uses jr02
        j[12] = jp00 * dXc_dw[0][0] + jr02 * dXc_dw[2][0];
        j[13] = jp00 * dXc_dw[0][1] + jr02 * dXc_dw[2][1];
        j[14] = jp00 * dXc_dw[0][2] + jr02 * dXc_dw[2][2];
        j[15] = jp00;
        j[16] = 0.0;
        j[17] = jr02;
    }

    if (jacobians[1]) {
        double R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        double* j = jacobians[1];  // row-major 3×3
        // row 0 (∂u_L/∂X_w)
        j[0] = jp00 * R[0] + jp02 * R[6];
        j[1] = jp00 * R[1] + jp02 * R[7];
        j[2] = jp00 * R[2] + jp02 * R[8];
        // row 1 (∂v_L/∂X_w)
        j[3] = jp11 * R[3] + jp12 * R[6];
        j[4] = jp11 * R[4] + jp12 * R[7];
        j[5] = jp11 * R[5] + jp12 * R[8];
        // row 2 (∂u_R/∂X_w) — uses jr02 instead of jp02
        j[6] = jp00 * R[0] + jr02 * R[6];
        j[7] = jp00 * R[1] + jr02 * R[7];
        j[8] = jp00 * R[2] + jr02 * R[8];
    }

    return true;
}

class AnalyticStereoReprojectionCostFunction
    : public ceres::SizedCostFunction<StereoReprojectionCost::kNumResiduals,    // 3 residuals
                                      StereoReprojectionCost::kNumPoseParams,   // 6 pose params
                                      StereoReprojectionCost::kNumPointParams>  // 3 point params
{
   public:
    explicit AnalyticStereoReprojectionCostFunction(const StereoReprojectionCost& cost)
        : cost_(cost) {}

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }

   private:
    StereoReprojectionCost cost_;
};

// confidence-weighted monocular cost — identical to ReprojectionCost but residuals/Jacobians
// are premultiplied by sqrt_w (equivalent to weighting the information matrix by w).

bool ConfidenceWeightedReprojectionCost::operator()(const double* const pose,
                                                     const double* const point,
                                                     double* residuals,
                                                     double** jacobians) const {
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc  = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    const double u_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_proj = fy * Xc[1] * inv_Zc + cy;
    residuals[0] = sqrt_w * (u_proj - u_obs);
    residuals[1] = sqrt_w * (v_proj - v_obs);

    if (!jacobians) return true;

    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;

    const double dXc_dw[3][3] = {
        {0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

    if (jacobians[0]) {
        double* j = jacobians[0];
        j[0]  = sqrt_w * (jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0]);
        j[1]  = sqrt_w * (jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1]);
        j[2]  = sqrt_w * (jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2]);
        j[3]  = sqrt_w * jp00;
        j[4]  = 0.0;
        j[5]  = sqrt_w * jp02;
        j[6]  = sqrt_w * (jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0]);
        j[7]  = sqrt_w * (jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1]);
        j[8]  = sqrt_w * (jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2]);
        j[9]  = 0.0;
        j[10] = sqrt_w * jp11;
        j[11] = sqrt_w * jp12;
    }

    if (jacobians[1]) {
        double R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        double* j = jacobians[1];
        j[0] = sqrt_w * (jp00 * R[0] + jp02 * R[6]);
        j[1] = sqrt_w * (jp00 * R[1] + jp02 * R[7]);
        j[2] = sqrt_w * (jp00 * R[2] + jp02 * R[8]);
        j[3] = sqrt_w * (jp11 * R[3] + jp12 * R[6]);
        j[4] = sqrt_w * (jp11 * R[4] + jp12 * R[7]);
        j[5] = sqrt_w * (jp11 * R[5] + jp12 * R[8]);
    }

    return true;
}

class AnalyticConfReprojCostFunction
    : public ceres::SizedCostFunction<2, 6, 3>  // kNumResiduals=2, kNumPoseParams=6, kNumPointParams=3
{
   public:
    explicit AnalyticConfReprojCostFunction(const ConfidenceWeightedReprojectionCost& cost)
        : cost_(cost) {}
    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }
   private:
    ConfidenceWeightedReprojectionCost cost_;
};

// confidence-weighted stereo cost

bool ConfidenceWeightedStereoCost::operator()(const double* const pose,
                                               const double* const point,
                                               double* residuals,
                                               double** jacobians) const {
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc  = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    const double u_L_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_L_proj = fy * Xc[1] * inv_Zc + cy;
    const double u_R_proj = fx * (Xc[0] - baseline) * inv_Zc + cx;

    residuals[0] = sqrt_w * (u_L_proj - u_L_obs);
    residuals[1] = sqrt_w * (v_L_proj - v_L_obs);
    residuals[2] = sqrt_w * (u_R_proj - u_R_obs);

    if (!jacobians) return true;

    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;
    const double jr02 = -fx * (Xc[0] - baseline) * inv_Zc2;

    const double dXc_dw[3][3] = {{0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

    if (jacobians[0]) {
        double* j = jacobians[0];  // row-major 3×6
        j[0]  = sqrt_w * (jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0]);
        j[1]  = sqrt_w * (jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1]);
        j[2]  = sqrt_w * (jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2]);
        j[3]  = sqrt_w * jp00;
        j[4]  = 0.0;
        j[5]  = sqrt_w * jp02;
        j[6]  = sqrt_w * (jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0]);
        j[7]  = sqrt_w * (jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1]);
        j[8]  = sqrt_w * (jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2]);
        j[9]  = 0.0;
        j[10] = sqrt_w * jp11;
        j[11] = sqrt_w * jp12;
        j[12] = sqrt_w * (jp00 * dXc_dw[0][0] + jr02 * dXc_dw[2][0]);
        j[13] = sqrt_w * (jp00 * dXc_dw[0][1] + jr02 * dXc_dw[2][1]);
        j[14] = sqrt_w * (jp00 * dXc_dw[0][2] + jr02 * dXc_dw[2][2]);
        j[15] = sqrt_w * jp00;
        j[16] = 0.0;
        j[17] = sqrt_w * jr02;
    }

    if (jacobians[1]) {
        double R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        double* j = jacobians[1];  // row-major 3×3
        j[0] = sqrt_w * (jp00 * R[0] + jp02 * R[6]);
        j[1] = sqrt_w * (jp00 * R[1] + jp02 * R[7]);
        j[2] = sqrt_w * (jp00 * R[2] + jp02 * R[8]);
        j[3] = sqrt_w * (jp11 * R[3] + jp12 * R[6]);
        j[4] = sqrt_w * (jp11 * R[4] + jp12 * R[7]);
        j[5] = sqrt_w * (jp11 * R[5] + jp12 * R[8]);
        j[6] = sqrt_w * (jp00 * R[0] + jr02 * R[6]);
        j[7] = sqrt_w * (jp00 * R[1] + jr02 * R[7]);
        j[8] = sqrt_w * (jp00 * R[2] + jr02 * R[8]);
    }

    return true;
}

class AnalyticConfStereoCostFunction
    : public ceres::SizedCostFunction<3, 6, 3>  // kNumResiduals=3, kNumPoseParams=6, kNumPointParams=3
{
   public:
    explicit AnalyticConfStereoCostFunction(const ConfidenceWeightedStereoCost& cost)
        : cost_(cost) {}
    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }
   private:
    ConfidenceWeightedStereoCost cost_;
};

// pitch/roll soft constraint — penalizes R[3] and R[5] (zero when level). no height ref so no
// drift.
struct PitchRollCost {
    double w_rp;
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        T R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        residuals[0] = T(w_rp) * R[3];  // R(1,0) — roll proxy  → 0 when level
        residuals[1] = T(w_rp) * R[5];  // R(1,2) — pitch proxy → 0 when level
        return true;
    }
    static ceres::CostFunction* Create(double w_rp) {
        return new ceres::AutoDiffCostFunction<PitchRollCost, 2, 6>(new PitchRollCost{w_rp});
    }
};

/// Ground-plane height constraint: penalizes camera Y deviating from reference height.
/// For a car on flat ground, camera height should remain approximately constant.
struct GroundHeightCost {
    double y_ref;   // reference camera-center Y (world frame)
    double w_h;     // weight

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Camera center in world frame: C = -R^T * t
        // Ceres AngleAxisToRotationMatrix outputs column-major R:
        //   R[0..2] = col0, R[3..5] = col1, R[6..8] = col2
        // Camera center Y = -(R[1]*t0 + R[4]*t1 + R[7]*t2)
        T R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        T cy = -(R[1] * pose[3] + R[4] * pose[4] + R[7] * pose[5]);
        residuals[0] = T(w_h) * (cy - T(y_ref));
        return true;
    }

    static ceres::CostFunction* Create(double y_ref, double w_h) {
        auto* c = new GroundHeightCost{y_ref, w_h};
        return new ceres::AutoDiffCostFunction<GroundHeightCost, 1, 6>(c);
    }
};

// pose prior — soft anchor to the pre-BA PnP estimate so sparse KFs don't drift.
struct PosePriorCost {
    double prior[6];
    double w_r, w_t;

    template <typename T>
    bool operator()(const T* const pose, T* res) const {
        for (int i = 0; i < 3; ++i) res[i] = T(w_r) * (pose[i] - T(prior[i]));
        for (int i = 0; i < 3; ++i) res[i + 3] = T(w_t) * (pose[i + 3] - T(prior[i + 3]));
        return true;
    }

    static ceres::CostFunction* Create(const double* p, double w_r, double w_t) {
        auto* c = new PosePriorCost;
        std::copy(p, p + 6, c->prior);
        c->w_r = w_r;
        c->w_t = w_t;
        return new ceres::AutoDiffCostFunction<PosePriorCost, 6, 6>(c);
    }
};

// pose ↔ Isometry3d conversion helpers

/// Eigen::Isometry3d → 6-DOF pose vector [ω, t]
static void isometry_to_pose(const Eigen::Isometry3d& T, double* pose) {
    // angle-axis from rotation matrix
    Eigen::AngleAxisd aa(T.rotation());
    Eigen::Vector3d omega = aa.angle() * aa.axis();
    pose[0] = omega.x();
    pose[1] = omega.y();
    pose[2] = omega.z();
    pose[3] = T.translation().x();
    pose[4] = T.translation().y();
    pose[5] = T.translation().z();
}

/// 6-DOF pose vector [ω, t] → Eigen::Isometry3d
static Eigen::Isometry3d pose_to_isometry(const double* pose) {
    Eigen::Vector3d omega(pose[0], pose[1], pose[2]);
    double angle = omega.norm();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    if (angle > 1e-9) {
        T.linear() = Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix();
    }
    T.translation() << pose[3], pose[4], pose[5];
    return T;
}

// LocalBA factory + optimize()

LocalBA::Ptr LocalBA::create(const Camera& cam, Map::Ptr map, const Config& cfg) {
    auto ba = std::shared_ptr<LocalBA>(new LocalBA());
    ba->cam_ = cam;
    ba->map_ = map;
    ba->cfg_ = cfg;
    return ba;
}

void LocalBA::optimize() {
    // 1. gather the sliding window of keyframes
    auto window = map_->local_window();
    if (window.size() < 2) return;

    // 2. collect all map points visible in the window
    std::unordered_map<long, MapPoint::Ptr> active_points;
    for (auto& kf : window) {
        for (auto& mp : kf->map_points) {
            if (mp && !mp->is_bad) {
                active_points[mp->id] = mp;
            }
        }
    }
    if (active_points.empty()) return;

    // 3. allocate parameter blocks
    // poses: window_size × 6 doubles
    std::unordered_map<long, std::vector<double>> pose_params;
    for (auto& kf : window) {
        pose_params[kf->id].resize(6);
        isometry_to_pose(kf->T_cw, pose_params[kf->id].data());
    }

    // points: map_point_id × 3 doubles
    std::unordered_map<long, std::array<double, 3>> point_params;
    for (auto& [id, mp] : active_points) {
        point_params[id] = {mp->position.x(), mp->position.y(), mp->position.z()};
    }

    // record pre-BA poses for the pose prior soft anchor
    std::unordered_map<long, std::array<double, 6>> prior_poses;
    for (auto& kf : window) {
        prior_poses[kf->id] = {};
        std::copy(pose_params[kf->id].begin(), pose_params[kf->id].end(),
                  prior_poses[kf->id].begin());
    }

    // 4. build Ceres problem
    double effective_huber = cfg_.huber_delta;

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(effective_huber);

    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();

        for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
            auto& mp = kf->map_points[kp_idx];
            if (!mp || mp->is_bad) continue;

            auto pit = point_params.find(mp->id);
            if (pit == point_params.end()) continue;

            double* pt = pit->second.data();
            const cv::Point2f& obs = kf->keypoints[kp_idx].pt;

            // Per-keypoint confidence weight (1.0 for ORB mode; L2/LG-derived in hybrid)
            double w = 1.0;
            if (kp_idx < (int)kf->match_confidence.size())
                w = std::max(0.01, (double)kf->match_confidence[kp_idx]);

            // Distant points anchor yaw; floor their weight so the optimizer
            // cannot down-weight them to near-zero even if L2 confidence is low.
            {
                double Xc[3];
                ceres::AngleAxisRotatePoint(pose, pt, Xc);
                Xc[0] += pose[3]; Xc[1] += pose[4]; Xc[2] += pose[5];
                if (Xc[2] > 40.0)
                    w = std::max(w, 0.5);
            }

            // use stereo cost (3 residuals) when a valid right-image observation exists
            if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                ConfidenceWeightedStereoCost cost(obs.x, obs.y, kf->uR[kp_idx], cam_.fx, cam_.fy,
                                                  cam_.cx, cam_.cy, cam_.baseline, w);
                problem.AddResidualBlock(new AnalyticConfStereoCostFunction(cost), loss, pose, pt);
            } else {
                ConfidenceWeightedReprojectionCost cost(obs.x, obs.y, cam_.fx, cam_.fy, cam_.cx,
                                                        cam_.cy, w);
                problem.AddResidualBlock(new AnalyticConfReprojCostFunction(cost), loss, pose, pt);
            }
        }

        problem.AddParameterBlock(pose, 6);
    }

    // add point parameter blocks
    for (auto& [id, pt] : point_params) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    // fix oldest KF pose to remove gauge freedom
    if (!window.empty()) {
        double* oldest_pose = pose_params[window.front()->id].data();
        problem.SetParameterBlockConstant(oldest_pose);
    }

    // pitch/roll soft constraint (stereo only)
    // w_rp=100: strong pitch/roll constraint (gravity-aligned); yaw left free for BA
    if (cam_.is_stereo()) {
        for (auto& kf : window) {
            if (kf == window.front()) continue;
            problem.AddResidualBlock(PitchRollCost::Create(50.0), nullptr,
                                     pose_params[kf->id].data());
        }
    }

    // ground-height constraint DISABLED — KITTI 00 has 25m elevation change;
    // absolute height constraint distorts BA and causes Y-drift instability.
    // Stereo reprojection residuals already constrain Y via depth observations.

    // pose prior — translation-only anchor; w_r=0.0 (no rotation prior) frees yaw for BA
    for (auto& kf : window) {
        if (kf == window.front()) continue;
        auto it = prior_poses.find(kf->id);
        if (it == prior_poses.end()) continue;
        problem.AddResidualBlock(PosePriorCost::Create(it->second.data(), 0.0, 2.0), nullptr,
                                 pose_params[kf->id].data());
    }

    // 5. solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = cfg_.max_iterations;
    options.minimizer_progress_to_stdout = cfg_.verbose;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cfg_.verbose) {
        std::cout << summary.BriefReport() << "\n";
    }

    // 6. write back optimized poses + log yaw correction
    for (auto& kf : window) {
        Eigen::Isometry3d T_old = kf->T_cw;
        kf->T_cw = pose_to_isometry(pose_params[kf->id].data());
        // diagnostic: BA yaw correction magnitude
        Eigen::Matrix3d R_wc_old = T_old.inverse().rotation();
        Eigen::Matrix3d R_wc_new = kf->T_cw.inverse().rotation();
        double yaw_old = std::atan2(R_wc_old(0, 2), R_wc_old(0, 0)) * 180.0 / 3.14159265358979323846;
        double yaw_new = std::atan2(R_wc_new(0, 2), R_wc_new(0, 0)) * 180.0 / 3.14159265358979323846;
        double delta_yaw = yaw_new - yaw_old;
        while (delta_yaw >  180.0) delta_yaw -= 360.0;
        while (delta_yaw < -180.0) delta_yaw += 360.0;
        if (std::abs(delta_yaw) > 0.01)
            fprintf(stderr, "[BA-DIAG] kf=%ld delta_yaw=%.4f deg\n", kf->id, delta_yaw);
    }

    // 7. write back optimized 3D point positions
    for (auto& [id, mp] : active_points) {
        auto& pt = point_params[id];
        mp->position = Eigen::Vector3d(pt[0], pt[1], pt[2]);
    }

    // 8. post-BA culling — mark points with >6px reprojection error as bad
    {
        const double cull_thresh2 = 36.0;  // 6 px²
        for (auto& kf : window) {
            const double* pose = pose_params.at(kf->id).data();
            for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
                auto& mp = kf->map_points[kp_idx];
                if (!mp || mp->is_bad) continue;
                auto pit = point_params.find(mp->id);
                if (pit == point_params.end()) continue;
                const double* pt = pit->second.data();

                double Xc[3];
                ceres::AngleAxisRotatePoint(pose, pt, Xc);
                Xc[0] += pose[3];
                Xc[1] += pose[4];
                Xc[2] += pose[5];
                if (Xc[2] <= 0.0) {
                    mp->is_bad = true;
                    continue;
                }

                // cull stereo points beyond 80m — depth is unreliable at that range
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f &&
                    Xc[2] > 80.0) {
                    mp->is_bad = true;
                    continue;
                }

                double u = cam_.fx * Xc[0] / Xc[2] + cam_.cx;
                double v = cam_.fy * Xc[1] / Xc[2] + cam_.cy;
                double du = u - kf->keypoints[kp_idx].pt.x;
                double dv = v - kf->keypoints[kp_idx].pt.y;
                if (du * du + dv * dv > cull_thresh2) {
                    mp->is_bad = true;
                    continue;
                }

                // also cull if right-camera reprojection exceeds threshold
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                    double u_R = cam_.fx * (Xc[0] - cam_.baseline) / Xc[2] + cam_.cx;
                    double dur = u_R - kf->uR[kp_idx];
                    if (dur * dur > cull_thresh2) mp->is_bad = true;
                }
            }
        }
    }

    // remove map points flagged as bad by other threads
    map_->cleanup_bad_map_points();
}

}  // namespace slam
