// sliding-window bundle adjustment w/ Ceres
// quaternion+translation parameterization (7-DOF), analytical Jacobians
// information-weighted stereo residuals w/ depth-attenuated disparity term
// Schur complement marginalization of oldest KF -> dense prior on connected poses

#include "slam/local_ba.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace slam {

// StereoReprojCost: 3 residuals, analytical Jacobians
// SizedCostFunction<3, 7, 3>

StereoReprojCost::StereoReprojCost(double obs_uL, double obs_vL, double obs_uR, double fx,
                                   double fy, double cx, double cy, double b,
                                   double info_uv, double info_disp)
    : obs_uL_(obs_uL),
      obs_vL_(obs_vL),
      obs_uR_(obs_uR),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      b_(b),
      sqrt_info_uv_(std::sqrt(std::max(info_uv, 1e-6))),
      sqrt_info_d_(std::sqrt(std::max(info_disp, 1e-6))) {}

bool StereoReprojCost::Evaluate(double const* const* parameters, double* residuals,
                                double** jacobians) const {
    // params[0] = pose [qx,qy,qz,qw, tx,ty,tz] — Eigen stores (x,y,z,w) internally
    // params[1] = point [X,Y,Z]
    Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);

    // forward pass: Pc = R(q)*Pw + t
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const Eigen::Vector3d Pc = R * Pw + t;

    const double Xc = Pc.x(), Yc = Pc.y(), Zc = Pc.z();
    const double inv_z = 1.0 / Zc;
    const double inv_z2 = inv_z * inv_z;

    // information-weighted residuals
    //   r0 = sqrt_Omega_uv * (fx*Xc/Zc + cx - uL)
    //   r1 = sqrt_Omega_uv * (fy*Yc/Zc + cy - vL)
    //   r2 = sqrt_Omega_d  * (fx*(Xc-b)/Zc + cx - uR)
    residuals[0] = sqrt_info_uv_ * (fx_ * Xc * inv_z + cx_ - obs_uL_);
    residuals[1] = sqrt_info_uv_ * (fy_ * Yc * inv_z + cy_ - obs_vL_);
    residuals[2] = sqrt_info_d_  * (fx_ * (Xc - b_) * inv_z + cx_ - obs_uR_);

    if (!jacobians) return true;

    // projection Jacobian J_proj (3x3) — sparse, only 5 nonzero entries
    const double j00 = fx_ * inv_z;
    const double j02 = -fx_ * Xc * inv_z2;
    const double j11 = fy_ * inv_z;
    const double j12 = -fy_ * Yc * inv_z2;
    const double j22 = -fx_ * (Xc - b_) * inv_z2;

    // pose Jacobian (3x7): chain rule through dR/dq_i * Pw
    if (jacobians[0]) {
        // each dR/dq_i is a 3x3 matrix; we only need its product w/ Pw
        // these are the four d(R*Pw)/dq_i 3-vectors

        const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
        const double X = Pw.x(), Y = Pw.y(), Z = Pw.z();

        const double d0_qx = 2.0 * (qy * Y + qz * Z);
        const double d1_qx = 2.0 * (qy * X - 2.0 * qx * Y - qw * Z);
        const double d2_qx = 2.0 * (qz * X + qw * Y - 2.0 * qx * Z);

        const double d0_qy = 2.0 * (-2.0 * qy * X + qx * Y + qw * Z);
        const double d1_qy = 2.0 * (qx * X + qz * Z);
        const double d2_qy = 2.0 * (-qw * X + qz * Y - 2.0 * qy * Z);

        const double d0_qz = 2.0 * (-2.0 * qz * X - qw * Y + qx * Z);
        const double d1_qz = 2.0 * (qw * X - 2.0 * qz * Y + qy * Z);
        const double d2_qz = 2.0 * (qx * X + qy * Y);

        const double d0_qw = 2.0 * (-qz * Y + qy * Z);
        const double d1_qw = 2.0 * (qz * X - qx * Z);
        const double d2_qw = 2.0 * (-qy * X + qx * Y);

        // J[row][col] = sqrt_info * J_proj[row] dot dPc_dq[col]
        double* J = jacobians[0];  // row-major 3x7

        // row 0 (u_L)
        J[0] = sqrt_info_uv_ * (j00 * d0_qx + j02 * d2_qx);
        J[1] = sqrt_info_uv_ * (j00 * d0_qy + j02 * d2_qy);
        J[2] = sqrt_info_uv_ * (j00 * d0_qz + j02 * d2_qz);
        J[3] = sqrt_info_uv_ * (j00 * d0_qw + j02 * d2_qw);

        // row 1 (v_L)
        J[7]  = sqrt_info_uv_ * (j11 * d1_qx + j12 * d2_qx);
        J[8]  = sqrt_info_uv_ * (j11 * d1_qy + j12 * d2_qy);
        J[9]  = sqrt_info_uv_ * (j11 * d1_qz + j12 * d2_qz);
        J[10] = sqrt_info_uv_ * (j11 * d1_qw + j12 * d2_qw);

        // row 2 (u_R)
        J[14] = sqrt_info_d_ * (j00 * d0_qx + j22 * d2_qx);
        J[15] = sqrt_info_d_ * (j00 * d0_qy + j22 * d2_qy);
        J[16] = sqrt_info_d_ * (j00 * d0_qz + j22 * d2_qz);
        J[17] = sqrt_info_d_ * (j00 * d0_qw + j22 * d2_qw);

        // translation cols 4-6: dPc/dt = I3, so just J_proj directly
        J[4]  = sqrt_info_uv_ * j00;
        J[5]  = 0.0;
        J[6]  = sqrt_info_uv_ * j02;
        J[11] = 0.0;
        J[12] = sqrt_info_uv_ * j11;
        J[13] = sqrt_info_uv_ * j12;
        J[18] = sqrt_info_d_ * j00;
        J[19] = 0.0;
        J[20] = sqrt_info_d_ * j22;
    }

    // point Jacobian (3x3): dPc/dPw = R, so J_point = sqrt_info * J_proj * R
    if (jacobians[1]) {
        double* J = jacobians[1];  // row-major 3x3
        for (int c = 0; c < 3; ++c) {
            J[c]     = sqrt_info_uv_ * (j00 * R(0, c) + j02 * R(2, c));
            J[3 + c] = sqrt_info_uv_ * (j11 * R(1, c) + j12 * R(2, c));
            J[6 + c] = sqrt_info_d_  * (j00 * R(0, c) + j22 * R(2, c));
        }
    }

    return true;
}

// MonoReprojCost: 2 residuals, analytical Jacobians
// SizedCostFunction<2, 7, 3>
// same math as stereo but without the right-camera residual

MonoReprojCost::MonoReprojCost(double obs_u, double obs_v, double fx, double fy, double cx,
                               double cy, double info_uv)
    : obs_u_(obs_u),
      obs_v_(obs_v),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      sqrt_info_uv_(std::sqrt(std::max(info_uv, 1e-6))) {}

bool MonoReprojCost::Evaluate(double const* const* parameters, double* residuals,
                              double** jacobians) const {
    Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);

    const Eigen::Matrix3d R = q.toRotationMatrix();
    const Eigen::Vector3d Pc = R * Pw + t;

    const double Xc = Pc.x(), Yc = Pc.y(), Zc = Pc.z();
    const double inv_z = 1.0 / Zc;
    const double inv_z2 = inv_z * inv_z;

    residuals[0] = sqrt_info_uv_ * (fx_ * Xc * inv_z + cx_ - obs_u_);
    residuals[1] = sqrt_info_uv_ * (fy_ * Yc * inv_z + cy_ - obs_v_);

    if (!jacobians) return true;

    const double j00 = fx_ * inv_z;
    const double j02 = -fx_ * Xc * inv_z2;
    const double j11 = fy_ * inv_z;
    const double j12 = -fy_ * Yc * inv_z2;

    if (jacobians[0]) {
        const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
        const double X = Pw.x(), Y = Pw.y(), Z = Pw.z();

        const double d0_qx = 2.0 * (qy * Y + qz * Z);
        const double d1_qx = 2.0 * (qy * X - 2.0 * qx * Y - qw * Z);
        const double d2_qx = 2.0 * (qz * X + qw * Y - 2.0 * qx * Z);

        const double d0_qy = 2.0 * (-2.0 * qy * X + qx * Y + qw * Z);
        const double d1_qy = 2.0 * (qx * X + qz * Z);
        const double d2_qy = 2.0 * (-qw * X + qz * Y - 2.0 * qy * Z);

        const double d0_qz = 2.0 * (-2.0 * qz * X - qw * Y + qx * Z);
        const double d1_qz = 2.0 * (qw * X - 2.0 * qz * Y + qy * Z);
        const double d2_qz = 2.0 * (qx * X + qy * Y);

        const double d0_qw = 2.0 * (-qz * Y + qy * Z);
        const double d1_qw = 2.0 * (qz * X - qx * Z);
        const double d2_qw = 2.0 * (-qy * X + qx * Y);

        double* J = jacobians[0];  // row-major 2x7

        J[0]  = sqrt_info_uv_ * (j00 * d0_qx + j02 * d2_qx);
        J[1]  = sqrt_info_uv_ * (j00 * d0_qy + j02 * d2_qy);
        J[2]  = sqrt_info_uv_ * (j00 * d0_qz + j02 * d2_qz);
        J[3]  = sqrt_info_uv_ * (j00 * d0_qw + j02 * d2_qw);

        J[7]  = sqrt_info_uv_ * (j11 * d1_qx + j12 * d2_qx);
        J[8]  = sqrt_info_uv_ * (j11 * d1_qy + j12 * d2_qy);
        J[9]  = sqrt_info_uv_ * (j11 * d1_qz + j12 * d2_qz);
        J[10] = sqrt_info_uv_ * (j11 * d1_qw + j12 * d2_qw);

        J[4]  = sqrt_info_uv_ * j00;
        J[5]  = 0.0;
        J[6]  = sqrt_info_uv_ * j02;
        J[11] = 0.0;
        J[12] = sqrt_info_uv_ * j11;
        J[13] = sqrt_info_uv_ * j12;
    }

    if (jacobians[1]) {
        double* J = jacobians[1];  // row-major 2x3
        for (int c = 0; c < 3; ++c) {
            J[c]     = sqrt_info_uv_ * (j00 * R(0, c) + j02 * R(2, c));
            J[3 + c] = sqrt_info_uv_ * (j11 * R(1, c) + j12 * R(2, c));
        }
    }

    return true;
}

// pose <-> Isometry3d conversion (quaternion 7-DOF)
// NOTE: this is different from pose_graph.cpp which uses angle-axis 6-DOF

static void isometry_to_pose(const Eigen::Isometry3d& T, double* pose) {
    Eigen::Quaterniond q(T.rotation());
    q.normalize();
    // Eigen coeffs() = (x,y,z,w)
    pose[0] = q.x();
    pose[1] = q.y();
    pose[2] = q.z();
    pose[3] = q.w();
    pose[4] = T.translation().x();
    pose[5] = T.translation().y();
    pose[6] = T.translation().z();
}

static Eigen::Isometry3d pose_to_isometry(const double* pose) {
    Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);  // ctor is (w,x,y,z)
    q.normalize();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() << pose[4], pose[5], pose[6];
    return T;
}

// quaternion point rotation (for post-BA culling)
// Hamilton double-cross: result = q * p * q^{-1}
static void quat_rotate_point(const double* q, const double* p, double* result) {
    const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    // t = 2*(v x p)
    const double tx = 2.0 * (qy * p[2] - qz * p[1]);
    const double ty = 2.0 * (qz * p[0] - qx * p[2]);
    const double tz = 2.0 * (qx * p[1] - qy * p[0]);
    // result = p + w*t + v x t
    result[0] = p[0] + qw * tx + qy * tz - qz * ty;
    result[1] = p[1] + qw * ty + qz * tx - qx * tz;
    result[2] = p[2] + qw * tz + qx * ty - qy * tx;
}

// marginalization helpers

// 7x6 PlusJacobian of ProductManifold(EigenQuaternion, Euclidean<3>)
// J_plus maps tangent delta (R^6) to ambient perturbation (R^7)
// layout: [J_q (4x3) | 0] / [0 | I3 (3x3)]
// J_q = 0.5 * [[w,-z,y],[z,w,-x],[-y,x,w],[-x,-y,-z]]
static Eigen::Matrix<double, 7, 6> pose_plus_jacobian(const double* pose) {
    const double qx = pose[0], qy = pose[1], qz = pose[2], qw = pose[3];
    Eigen::Matrix<double, 7, 6> J = Eigen::Matrix<double, 7, 6>::Zero();
    J(0, 0) =  0.5 * qw;  J(0, 1) = -0.5 * qz;  J(0, 2) =  0.5 * qy;
    J(1, 0) =  0.5 * qz;  J(1, 1) =  0.5 * qw;  J(1, 2) = -0.5 * qx;
    J(2, 0) = -0.5 * qy;  J(2, 1) =  0.5 * qx;  J(2, 2) =  0.5 * qw;
    J(3, 0) = -0.5 * qx;  J(3, 1) = -0.5 * qy;  J(3, 2) = -0.5 * qz;
    J(4, 3) = 1.0;  J(5, 4) = 1.0;  J(6, 5) = 1.0;
    return J;
}

// pseudo-inverse of PlusJacobian: J_pinv (6x7)
// J^T J = diag(0.25*I3, I3), so J_pinv = diag(4*I3, I3) * J^T
// i.e. quaternion rows scaled by 2 (4*0.5), translation by 1
static Eigen::Matrix<double, 6, 7> pose_plus_jacobian_pinv(const double* pose) {
    Eigen::Matrix<double, 7, 6> J_plus = pose_plus_jacobian(pose);
    Eigen::Matrix<double, 6, 7> J_pinv = Eigen::Matrix<double, 6, 7>::Zero();
    J_pinv.block<3, 4>(0, 0) = 4.0 * J_plus.block<4, 3>(0, 0).transpose();
    J_pinv(3, 4) = 1.0;  J_pinv(4, 5) = 1.0;  J_pinv(5, 6) = 1.0;
    return J_pinv;
}

// quaternion log: unit quat (x,y,z,w) -> 3-vector delta s.t. q = exp(delta/2)
static Eigen::Vector3d quaternion_log(const double* q_data) {
    double qx = q_data[0], qy = q_data[1], qz = q_data[2], qw = q_data[3];
    if (qw < 0.0) { qx = -qx; qy = -qy; qz = -qz; qw = -qw; }  // positive hemisphere

    Eigen::Vector3d v(qx, qy, qz);
    double sin_half = v.norm();

    if (sin_half < 1e-10) {
        return 2.0 * v;  // small angle: theta ~ 2*sin(theta/2)
    }

    double half_angle = std::atan2(sin_half, qw);
    double angle = 2.0 * half_angle;
    return (angle / sin_half) * v;
}

// tangent-space delta between two poses: delta = Minus(x, x0)
// delta = [delta_rot(3); delta_trans(3)] where delta_rot = log(q * q0^{-1})
static Eigen::Matrix<double, 6, 1> pose_tangent_delta(const double* x, const double* x0) {
    Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);
    Eigen::Quaterniond q0(x0[3], x0[0], x0[1], x0[2]);
    q.normalize();
    q0.normalize();
    Eigen::Quaterniond q_rel = q * q0.conjugate();

    double q_rel_data[4] = {q_rel.x(), q_rel.y(), q_rel.z(), q_rel.w()};
    Eigen::Vector3d delta_rot = quaternion_log(q_rel_data);

    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = delta_rot;
    delta(3) = x[4] - x0[4];
    delta(4) = x[5] - x0[5];
    delta(5) = x[6] - x0[6];
    return delta;
}

// MarginalizationPriorCost

MarginalizationPriorCost::MarginalizationPriorCost(const MarginalizationInfo& info)
    : info_(info)
{
    set_num_residuals(info.total_dim);
    for (size_t i = 0; i < info.poses.size(); ++i) {
        mutable_parameter_block_sizes()->push_back(7);
    }

    // precompute FEJ ambient Jacobians: J_amb[i] = S_cols_i * J_pinv(x0_i)
    J_amb_.resize(info.poses.size());
    for (size_t i = 0; i < info.poses.size(); ++i) {
        int off = info.poses[i].offset;
        Eigen::MatrixXd S_i = info.S.block(0, off, info.total_dim, 6);
        Eigen::Matrix<double, 6, 7> J_pinv = pose_plus_jacobian_pinv(info.poses[i].x0.data());
        J_amb_[i] = S_i * J_pinv;  // total_dim x 7
    }
}

bool MarginalizationPriorCost::Evaluate(double const* const* parameters,
                                         double* residuals,
                                         double** jacobians) const {
    const int n = info_.total_dim;

    // stacked tangent-space delta from each pose's linearization point
    Eigen::VectorXd delta(n);
    for (size_t i = 0; i < info_.poses.size(); ++i) {
        int off = info_.poses[i].offset;
        delta.segment<6>(off) = pose_tangent_delta(parameters[i], info_.poses[i].x0.data());
    }

    // r = S * delta + e0
    Eigen::Map<Eigen::VectorXd> r(residuals, n);
    r = info_.S * delta + info_.e0;

    if (!jacobians) return true;

    // FEJ Jacobians are constant (precomputed in ctor)
    for (size_t i = 0; i < info_.poses.size(); ++i) {
        if (jacobians[i]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
                J_out(jacobians[i], n, 7);
            J_out = J_amb_[i];
        }
    }

    return true;
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
    auto window = map_->local_window();
    if (window.size() < 2) return;

    // collect all map pts visible in window
    std::unordered_map<long, MapPoint::Ptr> active_points;
    for (auto& kf : window) {
        for (auto& mp : kf->map_points) {
            if (mp && !mp->is_bad) {
                active_points[mp->id] = mp;
            }
        }
    }
    if (active_points.empty()) return;

    // 7 doubles per pose: [qx,qy,qz,qw, tx,ty,tz]
    std::unordered_map<long, std::vector<double>> pose_params;
    for (auto& kf : window) {
        pose_params[kf->id].resize(7);
        isometry_to_pose(kf->T_cw, pose_params[kf->id].data());
    }

    std::unordered_map<long, std::array<double, 3>> point_params;
    for (auto& [id, mp] : active_points) {
        point_params[id] = {mp->position.x(), mp->position.y(), mp->position.z()};
    }

    // build Ceres problem
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(cfg_.huber_delta);

    // stereo information model:
    //   Omega_uv = 1/sigma^2                    (bearing — constant)
    //   Omega_d  = (1/sigma^2) * min(1, Z_ref^2/Z^2)  (disparity — attenuated at depth)
    // sigma_Z = Z^2*sigma_d/(fx*b) grows quadratic w/ depth
    const double sigma2 = cfg_.sigma_px * cfg_.sigma_px;
    const double info_uv = 1.0 / sigma2;
    const double z_ref2  = cfg_.z_ref * cfg_.z_ref;

    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();

        Eigen::Quaterniond q_kf(pose[3], pose[0], pose[1], pose[2]);
        q_kf.normalize();
        const Eigen::Matrix3d R_kf = q_kf.toRotationMatrix();
        const Eigen::Vector3d t_kf(pose[4], pose[5], pose[6]);

        for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
            auto& mp = kf->map_points[kp_idx];
            if (!mp || mp->is_bad) continue;

            auto pit = point_params.find(mp->id);
            if (pit == point_params.end()) continue;

            double* pt = pit->second.data();
            const cv::Point2f& obs = kf->keypoints[kp_idx].pt;

            // depth at current linearization pt — used for disparity info weighting
            const Eigen::Vector3d Pw(pt[0], pt[1], pt[2]);
            const double Zc = (R_kf * Pw + t_kf).z();
            if (Zc <= 0.01) continue;  // behind camera

            if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                const double depth_factor = std::min(1.0, z_ref2 / (Zc * Zc));
                const double info_disp = info_uv * depth_factor;

                problem.AddResidualBlock(
                    StereoReprojCost::Create(obs.x, obs.y, kf->uR[kp_idx], cam_.fx, cam_.fy,
                                             cam_.cx, cam_.cy, cam_.baseline,
                                             info_uv, info_disp),
                    loss, pose, pt);
            } else {
                problem.AddResidualBlock(
                    MonoReprojCost::Create(obs.x, obs.y, cam_.fx, cam_.fy, cam_.cx, cam_.cy,
                                           info_uv),
                    loss, pose, pt);
            }
        }
    }

    // register quaternion manifold on each pose block
    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();
        problem.AddParameterBlock(pose, 7,
                                  new ceres::ProductManifold(new ceres::EigenQuaternionManifold,
                                                             new ceres::EuclideanManifold<3>));
    }

    for (auto& [id, pt] : point_params) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    // gauge freedom: marg prior or fix oldest KF
    // if we have a valid marg prior w/ all poses in window, inject it (provides gauge)
    // otherwise fall back to fixing oldest KF constant
    bool prior_injected = false;
    if (marg_info_.valid && !marg_info_.poses.empty()) {
        bool all_present = true;
        for (auto& pb : marg_info_.poses) {
            if (pose_params.find(pb.frame_id) == pose_params.end()) {
                all_present = false;
                break;
            }
        }
        if (all_present) {
            auto* prior_cost = new MarginalizationPriorCost(marg_info_);
            std::vector<double*> prior_param_ptrs;
            for (auto& pb : marg_info_.poses) {
                prior_param_ptrs.push_back(pose_params[pb.frame_id].data());
            }
            problem.AddResidualBlock(prior_cost, nullptr,  // no loss on prior
                                     prior_param_ptrs);
            prior_injected = true;
            if (cfg_.verbose) {
                fprintf(stderr, "[BA] Injected marginalization prior (%d poses, dim=%d)\n",
                        (int)marg_info_.poses.size(), marg_info_.total_dim);
            }
        } else {
            // prior references stale KFs (e.g. after map reset) — invalidate
            marg_info_.valid = false;
        }
    }

    if (!prior_injected && !window.empty()) {
        problem.SetParameterBlockConstant(pose_params[window.front()->id].data());  // fix oldest
    }

    // solve
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

    // write back optimised poses + log yaw correction for diagnostics
    for (auto& kf : window) {
        Eigen::Isometry3d T_old = kf->T_cw;
        kf->T_cw = pose_to_isometry(pose_params[kf->id].data());
        Eigen::Matrix3d R_wc_old = T_old.inverse().rotation();
        Eigen::Matrix3d R_wc_new = kf->T_cw.inverse().rotation();
        double yaw_old =
            std::atan2(R_wc_old(0, 2), R_wc_old(0, 0)) * (180.0 / 3.14159265358979323846);
        double yaw_new =
            std::atan2(R_wc_new(0, 2), R_wc_new(0, 0)) * (180.0 / 3.14159265358979323846);
        double delta_yaw = yaw_new - yaw_old;
        while (delta_yaw > 180.0) delta_yaw -= 360.0;
        while (delta_yaw < -180.0) delta_yaw += 360.0;
        if (std::abs(delta_yaw) > 0.01)
            fprintf(stderr, "[BA-DIAG] kf=%ld delta_yaw=%.4f deg\n", kf->id, delta_yaw);
    }

    // write back optimised 3D positions
    for (auto& [id, mp] : active_points) {
        auto& pt = point_params[id];
        mp->position = Eigen::Vector3d(pt[0], pt[1], pt[2]);
    }

    // post-BA culling
    {
        const double cull_thresh2 = 49.0;  // 7px^2
        for (auto& kf : window) {
            const double* pose = pose_params.at(kf->id).data();
            for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
                auto& mp = kf->map_points[kp_idx];
                if (!mp || mp->is_bad) continue;
                auto pit = point_params.find(mp->id);
                if (pit == point_params.end()) continue;
                const double* pt = pit->second.data();

                double Xc[3];
                quat_rotate_point(pose, pt, Xc);
                Xc[0] += pose[4];
                Xc[1] += pose[5];
                Xc[2] += pose[6];

                if (Xc[2] <= 0.0) {
                    mp->is_bad = true;
                    continue;
                }

                // cull stereo pts beyond 50m — depth uncertainty too high
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f &&
                    Xc[2] > 50.0) {
                    mp->is_bad = true;
                    continue;
                }

                // depth-adaptive: tighter threshold at distance
                double depth_scale = std::max(1.0, Xc[2] / 20.0);
                double adaptive_cull2 = cull_thresh2 / (depth_scale * depth_scale);

                double u = cam_.fx * Xc[0] / Xc[2] + cam_.cx;
                double v = cam_.fy * Xc[1] / Xc[2] + cam_.cy;
                double du = u - kf->keypoints[kp_idx].pt.x;
                double dv = v - kf->keypoints[kp_idx].pt.y;
                if (du * du + dv * dv > adaptive_cull2) {
                    mp->is_bad = true;
                    continue;
                }

                // right-camera reproj check
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                    double u_R = cam_.fx * (Xc[0] - cam_.baseline) / Xc[2] + cam_.cx;
                    double dur = u_R - kf->uR[kp_idx];
                    if (dur * dur > adaptive_cull2) mp->is_bad = true;
                }
            }
        }
    }

    // observation-ratio culling: pts frequently visible but rarely matched
    // probably dynamic objects or unstable features
    {
        const int   min_visible = 10;
        const float min_ratio   = 0.25f;  // need >= 25% match rate
        for (auto& [id, mp] : active_points) {
            if (mp->is_bad) continue;
            if (mp->visible_times >= min_visible) {
                float ratio = (float)mp->observed_times / (float)mp->visible_times;
                if (ratio < min_ratio) {
                    mp->is_bad = true;
                }
            }
        }
    }

    map_->cleanup_bad_map_points();

    // marginalization: capture info before oldest KF drops out
    if ((int)window.size() >= cfg_.window_size) {
        compute_marginalization_prior(window, pose_params, point_params);
    }
}

// compute_marginalization_prior()
// builds reduced camera Hessian via Schur complement:
// 1. identify oldest KF + connected poses (shared map pts)
// 2. for each map pt seen by oldest KF: accumulate per-point Hessian, Schur out
// 3. add old prior contribution (chain from previous marg)
// 4. marginalize oldest pose from pose-only Hessian
// 5. eigendecompose, clamp eigenvalues, build S & e0

void LocalBA::compute_marginalization_prior(
    const std::vector<Frame::Ptr>& window,
    const std::unordered_map<long, std::vector<double>>& pose_params,
    const std::unordered_map<long, std::array<double, 3>>& point_params)
{
    if (window.size() < 2) { marg_info_.valid = false; return; }

    Frame::Ptr oldest = window.front();

    // inverse index: point_id -> list of (kf, kp_idx)
    struct Observer { Frame::Ptr kf; int kp_idx; };
    std::unordered_map<long, std::vector<Observer>> point_observers;
    for (auto& kf : window) {
        for (int kp = 0; kp < (int)kf->map_points.size(); ++kp) {
            auto& mp = kf->map_points[kp];
            if (mp && !mp->is_bad && point_params.count(mp->id)) {
                point_observers[mp->id].push_back({kf, kp});
            }
        }
    }

    // find connected poses (share map pts w/ oldest KF)
    // pose_index: kf_id -> local index; 0 = oldest (will be marginalized)
    std::unordered_map<long, int> pose_index;
    pose_index[oldest->id] = 0;
    int next_idx = 1;

    std::vector<long> oldest_point_ids;
    for (int kp = 0; kp < (int)oldest->map_points.size(); ++kp) {
        auto& mp = oldest->map_points[kp];
        if (!mp || mp->is_bad) continue;
        if (!point_params.count(mp->id)) continue;
        oldest_point_ids.push_back(mp->id);

        auto it = point_observers.find(mp->id);
        if (it == point_observers.end()) continue;
        for (auto& obs : it->second) {
            if (pose_index.find(obs.kf->id) == pose_index.end()) {
                pose_index[obs.kf->id] = next_idx++;
            }
        }
    }

    int n_poses = next_idx;
    if (n_poses < 2) {
        marg_info_.valid = false;  // no connected poses — prior would be empty
        return;
    }

    int H_dim = n_poses * 6;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(H_dim, H_dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(H_dim);

    const double sigma2  = cfg_.sigma_px * cfg_.sigma_px;
    const double info_uv = 1.0 / sigma2;
    const double z_ref2  = cfg_.z_ref * cfg_.z_ref;

    // process each map pt observed by oldest KF
    for (long pt_id : oldest_point_ids) {
        auto obs_it = point_observers.find(pt_id);
        if (obs_it == point_observers.end()) continue;

        auto pt_it = point_params.find(pt_id);
        if (pt_it == point_params.end()) continue;
        const double* pt = pt_it->second.data();

        // per-point Hessian accumulators
        Eigen::Matrix3d H_PP = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b_P  = Eigen::Vector3d::Zero();

        struct PoseTerm {
            Eigen::Matrix<double, 6, 3> H_pP = Eigen::Matrix<double, 6, 3>::Zero();
            Eigen::Matrix<double, 6, 6> H_pp = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b_p  = Eigen::Matrix<double, 6, 1>::Zero();
        };
        std::unordered_map<int, PoseTerm> pose_terms;

        for (auto& obs : obs_it->second) {
            auto pidx_it = pose_index.find(obs.kf->id);
            if (pidx_it == pose_index.end()) continue;
            int pidx = pidx_it->second;

            auto pp_it = pose_params.find(obs.kf->id);
            if (pp_it == pose_params.end()) continue;
            const double* pose = pp_it->second.data();

            const cv::Point2f& kp = obs.kf->keypoints[obs.kp_idx].pt;
            bool is_stereo = cam_.is_stereo() &&
                             obs.kp_idx < (int)obs.kf->uR.size() &&
                             obs.kf->uR[obs.kp_idx] >= 0.0f;

            Eigen::Quaterniond q_kf(pose[3], pose[0], pose[1], pose[2]);
            q_kf.normalize();
            Eigen::Matrix3d R_kf = q_kf.toRotationMatrix();
            Eigen::Vector3d t_kf(pose[4], pose[5], pose[6]);
            Eigen::Vector3d Pw(pt[0], pt[1], pt[2]);
            double Zc = (R_kf * Pw + t_kf).z();
            if (Zc <= 0.01) continue;

            // evaluate ambient Jacobians via our cost functions directly
            const double* params[2] = {pose, pt};

            if (is_stereo) {
                double depth_factor = std::min(1.0, z_ref2 / (Zc * Zc));
                double info_d = info_uv * depth_factor;
                StereoReprojCost cost_fn(kp.x, kp.y, obs.kf->uR[obs.kp_idx],
                                         cam_.fx, cam_.fy, cam_.cx, cam_.cy,
                                         cam_.baseline, info_uv, info_d);

                double residual[3];
                double J_pose_amb[3 * 7];
                double J_point_amb[3 * 3];
                double* jacs[2] = {J_pose_amb, J_point_amb};
                cost_fn.Evaluate(params, residual, jacs);

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jp_amb(J_pose_amb);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpt_amb(J_point_amb);
                Eigen::Map<Eigen::Vector3d> r(residual);

                // project pose Jacobian to tangent space: J_tan = J_amb * J_plus
                Eigen::Matrix<double, 7, 6> J_plus = pose_plus_jacobian(pose);
                Eigen::Matrix<double, 3, 6> Jp_tan = Jp_amb * J_plus;
                Eigen::Matrix<double, 3, 3> Jpt_tan = Jpt_amb;  // already tangent (Euclidean)

                H_PP += Jpt_tan.transpose() * Jpt_tan;
                b_P  += Jpt_tan.transpose() * r;

                auto& pt_term = pose_terms[pidx];
                pt_term.H_pP += Jp_tan.transpose() * Jpt_tan;
                pt_term.H_pp += Jp_tan.transpose() * Jp_tan;
                pt_term.b_p  += Jp_tan.transpose() * r;
            } else {
                MonoReprojCost cost_fn(kp.x, kp.y, cam_.fx, cam_.fy, cam_.cx, cam_.cy,
                                       info_uv);

                double residual[2];
                double J_pose_amb[2 * 7];
                double J_point_amb[2 * 3];
                double* jacs[2] = {J_pose_amb, J_point_amb};
                cost_fn.Evaluate(params, residual, jacs);

                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jp_amb(J_pose_amb);
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jpt_amb(J_point_amb);
                Eigen::Map<Eigen::Vector2d> r(residual);

                Eigen::Matrix<double, 7, 6> J_plus = pose_plus_jacobian(pose);
                Eigen::Matrix<double, 2, 6> Jp_tan = Jp_amb * J_plus;
                Eigen::Matrix<double, 2, 3> Jpt_tan = Jpt_amb;

                H_PP += Jpt_tan.transpose() * Jpt_tan;
                b_P  += Jpt_tan.transpose() * r;

                auto& pt_term = pose_terms[pidx];
                pt_term.H_pP += Jp_tan.transpose() * Jpt_tan;
                pt_term.H_pp += Jp_tan.transpose() * Jp_tan;
                pt_term.b_p  += Jp_tan.transpose() * r;
            }
        }

        // Schur complement out this point
        double det = H_PP.determinant();
        if (std::abs(det) < 1e-12) continue;  // ill-conditioned, skip
        Eigen::Matrix3d H_PP_inv = H_PP.inverse();

        // direct contributions
        for (auto& [pidx_i, term_i] : pose_terms) {
            int off_i = pidx_i * 6;
            H.block<6, 6>(off_i, off_i) += term_i.H_pp;
            b.segment<6>(off_i)         += term_i.b_p;
        }

        // fill-in from point elimination: H -= H_{poses,P} * H_PP^{-1} * H_{P,poses}
        for (auto& [pidx_i, term_i] : pose_terms) {
            int off_i = pidx_i * 6;
            Eigen::Matrix<double, 6, 3> A_i = term_i.H_pP * H_PP_inv;

            for (auto& [pidx_j, term_j] : pose_terms) {
                int off_j = pidx_j * 6;
                H.block<6, 6>(off_i, off_j) -= A_i * term_j.H_pP.transpose();
            }
            b.segment<6>(off_i) -= A_i * b_P;
        }
    }

    // chain old prior contribution into the Hessian
    if (marg_info_.valid) {
        std::vector<const double*> prior_params;
        std::vector<int> prior_pose_indices;
        bool all_mapped = true;

        for (auto& pb : marg_info_.poses) {
            auto pidx_it = pose_index.find(pb.frame_id);
            if (pidx_it == pose_index.end()) { all_mapped = false; break; }
            prior_pose_indices.push_back(pidx_it->second);
            auto pp_it = pose_params.find(pb.frame_id);
            if (pp_it == pose_params.end()) { all_mapped = false; break; }
            prior_params.push_back(pp_it->second.data());
        }

        if (all_mapped && !prior_params.empty()) {
            MarginalizationPriorCost prior_cost(marg_info_);
            int n_r = marg_info_.total_dim;

            std::vector<double> residual(n_r);
            std::vector<std::vector<double>> jac_storage(prior_params.size());
            std::vector<double*> jacs(prior_params.size());
            for (size_t i = 0; i < prior_params.size(); ++i) {
                jac_storage[i].resize(n_r * 7);
                jacs[i] = jac_storage[i].data();
            }

            prior_cost.Evaluate(prior_params.data(), residual.data(), jacs.data());

            Eigen::Map<Eigen::VectorXd> r_prior(residual.data(), n_r);

            for (size_t i = 0; i < prior_params.size(); ++i) {
                int pidx_i = prior_pose_indices[i];
                int off_i = pidx_i * 6;
                const double* pose_i = prior_params[i];

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
                    J_amb_i(jacs[i], n_r, 7);
                Eigen::Matrix<double, 7, 6> J_plus_i = pose_plus_jacobian(pose_i);
                Eigen::MatrixXd J_tan_i = J_amb_i * J_plus_i;

                H.block<6, 6>(off_i, off_i) += J_tan_i.transpose() * J_tan_i;
                b.segment<6>(off_i)         += J_tan_i.transpose() * r_prior;

                for (size_t j = i + 1; j < prior_params.size(); ++j) {
                    int pidx_j = prior_pose_indices[j];
                    int off_j = pidx_j * 6;
                    const double* pose_j = prior_params[j];

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
                        J_amb_j(jacs[j], n_r, 7);
                    Eigen::Matrix<double, 7, 6> J_plus_j = pose_plus_jacobian(pose_j);
                    Eigen::MatrixXd J_tan_j = J_amb_j * J_plus_j;

                    Eigen::Matrix<double, 6, 6> cross = J_tan_i.transpose() * J_tan_j;
                    H.block<6, 6>(off_i, off_j) += cross;
                    H.block<6, 6>(off_j, off_i) += cross.transpose();
                }
            }
        }
    }

    // marginalize oldest pose (index 0)
    const int marg_dim = 6;
    const int keep_dim = H_dim - marg_dim;

    if (keep_dim <= 0) { marg_info_.valid = false; return; }

    Eigen::MatrixXd H_mm = H.block(0, 0, marg_dim, marg_dim);
    Eigen::MatrixXd H_mk = H.block(0, marg_dim, marg_dim, keep_dim);
    Eigen::MatrixXd H_km = H.block(marg_dim, 0, keep_dim, marg_dim);
    Eigen::MatrixXd H_kk = H.block(marg_dim, marg_dim, keep_dim, keep_dim);
    Eigen::VectorXd b_m  = b.segment(0, marg_dim);
    Eigen::VectorXd b_k  = b.segment(marg_dim, keep_dim);

    // LDLT inversion of H_mm (symmetric PSD)
    Eigen::LDLT<Eigen::MatrixXd> ldlt_mm(H_mm);
    if (ldlt_mm.info() != Eigen::Success || !ldlt_mm.isPositive()) {
        // not well-conditioned — add small regularization & retry
        H_mm += 1e-4 * Eigen::MatrixXd::Identity(marg_dim, marg_dim);
        ldlt_mm.compute(H_mm);
        if (ldlt_mm.info() != Eigen::Success) {
            fprintf(stderr, "[BA-MARG] H_mm decomposition failed — skipping marginalization\n");
            marg_info_.valid = false;
            return;
        }
    }

    Eigen::MatrixXd H_star = H_kk - H_km * ldlt_mm.solve(H_mk);
    Eigen::VectorXd b_star = b_k  - H_km * ldlt_mm.solve(b_m);

    H_star = 0.5 * (H_star + H_star.transpose());  // symmetrize (numerical)

    // eigendecompose, clamp, build S & e0
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H_star);
    if (eig.info() != Eigen::Success) {
        fprintf(stderr, "[BA-MARG] Eigendecomposition failed — skipping\n");
        marg_info_.valid = false;
        return;
    }

    Eigen::VectorXd D = eig.eigenvalues();
    Eigen::MatrixXd V = eig.eigenvectors();

    // clamp small/negative eigenvalues for numerical stability
    const double eig_min = 1e-6;
    int valid_dims = 0;
    for (int i = 0; i < D.size(); ++i) {
        if (D(i) > eig_min) ++valid_dims;
        D(i) = std::max(D(i), eig_min);
    }

    // S = sqrt(D) * V^T so that S^T S = V D V^T = H*
    Eigen::VectorXd sqrt_D = D.cwiseSqrt();
    Eigen::MatrixXd S = sqrt_D.asDiagonal() * V.transpose();

    // e0 = S^{-T} * b* = sqrt(D)^{-1} * V^T * b*
    Eigen::VectorXd inv_sqrt_D = sqrt_D.cwiseInverse();
    Eigen::VectorXd e0 = inv_sqrt_D.asDiagonal() * V.transpose() * b_star;

    // store new prior
    MarginalizationInfo new_info;
    new_info.valid = true;
    new_info.S  = S;
    new_info.e0 = e0;
    new_info.total_dim = keep_dim;

    // kept poses: all except index 0 (oldest), ordered by local index
    std::vector<std::pair<long, int>> sorted_poses;
    for (auto& [kf_id, pidx] : pose_index) {
        if (pidx == 0) continue;  // skip marginalized
        sorted_poses.push_back({kf_id, pidx});
    }
    std::sort(sorted_poses.begin(), sorted_poses.end(),
              [](auto& a, auto& b) { return a.second < b.second; });

    for (auto& [kf_id, pidx] : sorted_poses) {
        MarginalizationInfo::PoseBlock pb;
        pb.frame_id = kf_id;
        pb.offset   = (pidx - 1) * 6;  // offset in kept tangent vec
        auto pp_it = pose_params.find(kf_id);
        if (pp_it != pose_params.end()) {
            std::copy(pp_it->second.begin(), pp_it->second.end(), pb.x0.begin());
        }
        new_info.poses.push_back(pb);
    }

    marg_info_ = std::move(new_info);

    if (cfg_.verbose) {
        fprintf(stderr, "[BA-MARG] Marginalized KF %ld → prior on %d poses (dim=%d, "
                "valid_eigs=%d/%d, |b*|=%.4e)\n",
                oldest->id, (int)marg_info_.poses.size(), keep_dim,
                valid_dims, (int)D.size(), b_star.norm());
    }
}

}  // namespace slam
