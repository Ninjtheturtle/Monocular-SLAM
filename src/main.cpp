// entry point — parses args, loads kitti seq, runs the main loop
// usage: vslam.exe --sequence <path> [--start N] [--end N] [--no-viz] [--hybrid] [--xfeat <model>] [--lg <model>]

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/local_ba.hpp"
#include "slam/map.hpp"
#include "slam/pose_graph.hpp"
#include "slam/tracker.hpp"
#include "slam/visualizer.hpp"

// Deep frontend (only compiled when ENABLE_DEEP_FRONTEND is defined in CMake)
#ifdef ENABLE_DEEP_FRONTEND
#include "deep/lighterglue_async.hpp"
#include "deep/semi_dense_disparity.hpp"
#include "deep/ttt_autoencoder.hpp"
#include "deep/xfeat_extractor.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// --- kitti loader ---

struct KittiSequence {
    std::string sequence_path;
    std::vector<std::string> image_paths;        // left  (image_0/)
    std::vector<std::string> image_right_paths;  // right (image_1/); empty if not found
    std::vector<double> timestamps;
    slam::Camera camera;

    static KittiSequence load(const std::string& seq_path) {
        KittiSequence seq;
        seq.sequence_path = seq_path;

        seq.camera = slam::Camera::from_kitti_calib(seq_path + "/calib.txt");

        std::ifstream tf(seq_path + "/times.txt");
        if (!tf.is_open()) throw std::runtime_error("Cannot open times.txt in " + seq_path);
        double t;
        while (tf >> t) seq.timestamps.push_back(t);

        fs::path img_dir = fs::path(seq_path) / "image_0";
        if (!fs::exists(img_dir)) throw std::runtime_error("image_0/ not found in " + seq_path);

        std::vector<fs::path> paths;
        for (auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".png") paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        for (auto& p : paths) seq.image_paths.push_back(p.string());

        if (seq.image_paths.empty())
            throw std::runtime_error("No .png images found in " + img_dir.string());

        // optional right cam — enables stereo mode
        fs::path img_dir_r = fs::path(seq_path) / "image_1";
        if (fs::exists(img_dir_r)) {
            std::vector<fs::path> rpaths;
            for (auto& entry : fs::directory_iterator(img_dir_r)) {
                if (entry.path().extension() == ".png") rpaths.push_back(entry.path());
            }
            std::sort(rpaths.begin(), rpaths.end());
            for (auto& p : rpaths) seq.image_right_paths.push_back(p.string());
        }

        std::cout << "[KITTI] Loaded " << seq.image_paths.size() << " frames from " << seq_path;
        if (!seq.image_right_paths.empty())
            std::cout << " (stereo, b=" << seq.camera.baseline << " m)";
        std::cout << "\n";

        std::cout << "[KITTI] Intrinsics: fx=" << seq.camera.fx << "  fy=" << seq.camera.fy
                  << "  cx=" << seq.camera.cx << "  cy=" << seq.camera.cy << "\n";
        if (seq.camera.is_stereo()) {
            std::cout << "[KITTI] Stereo baseline: " << seq.camera.baseline << " m";
            if (seq.camera.baseline < 0.3 || seq.camera.baseline > 0.8)
                std::cout << "  *** WARNING: outside expected range [0.30, 0.80] m"
                             " — verify calib.txt uses P0/P1 (grayscale), not P2/P3 (color)";
            std::cout << "\n";
        }

        return seq;
    }
};

// --- arg parsing ---

struct Args {
    std::string sequence_path;
    int start_idx = 0;
    int end_idx = -1;  // -1 = all frames
    bool no_viz = false;
    bool hybrid = false;
    std::string xfeat_engine;
    std::string lg_engine;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sequence" && i + 1 < argc) {
            args.sequence_path = argv[++i];
        } else if (a == "--start" && i + 1 < argc) {
            args.start_idx = std::stoi(argv[++i]);
        } else if (a == "--end" && i + 1 < argc) {
            args.end_idx = std::stoi(argv[++i]);
        } else if (a == "--no-viz") {
            args.no_viz = true;
        } else if (a == "--hybrid") {
            args.hybrid = true;
        } else if (a == "--xfeat" && i + 1 < argc) {
            args.xfeat_engine = argv[++i];
        } else if (a == "--lg" && i + 1 < argc) {
            args.lg_engine = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: vslam.exe --sequence <path> [--start N] [--end N] [--no-viz]\n";
            exit(0);
        }
    }
    if (args.sequence_path.empty()) {
        std::cerr << "Error: --sequence <path> is required\n";
        exit(1);
    }
    return args;
}

// hacky path math — assumes kitti dir layout: .../sequences/00 -> .../poses/00.txt
static std::string derive_gt_path(const std::string& seq_path) {
    std::string p = seq_path;
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    size_t s1 = p.find_last_of("/\\");
    std::string seq_id = (s1 == std::string::npos) ? p : p.substr(s1 + 1);
    std::string up1 = (s1 == std::string::npos) ? "." : p.substr(0, s1);
    size_t s2 = up1.find_last_of("/\\");
    std::string base = (s2 == std::string::npos) ? "." : up1.substr(0, s2);
    return base + "/poses/" + seq_id + ".txt";
}

// returns (tx, ty, tz) camera centers per frame — for viz & ATE
static std::vector<std::array<float, 3>> load_gt_centers(const std::string& path) {
    std::vector<std::array<float, 3>> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        double v[12];
        for (int i = 0; i < 12; ++i) ss >> v[i];
        out.push_back({(float)v[3], (float)v[7], (float)v[11]}); // indices 3,7,11 = tx,ty,tz in row-major 3x4
    }
    return out;
}

// full T_wc per frame — only needed for yaw diagnostics
static std::vector<Eigen::Isometry3d> load_gt_poses(const std::string& path) {
    std::vector<Eigen::Isometry3d> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        double v[12];
        for (int i = 0; i < 12; ++i) ss >> v[i];
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() << v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10];
        T.translation() << v[3], v[7], v[11];
        out.push_back(T);
    }
    return out;
}

// KITTI cam: X=right, Y=down, Z=fwd — yaw rotates around Y
static double extract_yaw(const Eigen::Matrix3d& R) { return std::atan2(R(0, 2), R(0, 0)); }

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    cv::setRNGSeed(42); // reproducible RANSAC results

    KittiSequence seq = KittiSequence::load(args.sequence_path);

    // hacky: calib.txt doesn't include image dims so we peek at the first frame
    {
        cv::Mat img = cv::imread(seq.image_paths[0], cv::IMREAD_GRAYSCALE);
        seq.camera.width = img.cols;
        seq.camera.height = img.rows;
    }

    // --- init SLAM components ---
    auto map = slam::Map::create();
    auto local_ba = slam::LocalBA::create(seq.camera, map);
    auto pose_graph = slam::PoseGraph::create(map, seq.camera);
    slam::Visualizer::Ptr viz;

    slam::Tracker::Ptr tracker;

#ifdef ENABLE_DEEP_FRONTEND
    if (args.hybrid) {
        std::string xfeat_path =
            args.xfeat_engine.empty() ? "models/xfeat_fp16.engine" : args.xfeat_engine;
        std::string lg_path =
            args.lg_engine.empty() ? "models/lighterglue_fp16.engine" : args.lg_engine;

        deep::XFeatExtractor::Config xfeat_cfg;
        xfeat_cfg.engine_path = xfeat_path;
        xfeat_cfg.img_width = seq.camera.width;
        xfeat_cfg.img_height = seq.camera.height;
        auto xfeat = deep::XFeatExtractor::create(xfeat_cfg);

        deep::LighterGlueAsync::Config lg_cfg;
        lg_cfg.engine_path = lg_path;
        auto lg = deep::LighterGlueAsync::create(lg_cfg);

        deep::TTTLoopDetector::Config ttt_cfg;
        auto ttt = deep::TTTLoopDetector::create(ttt_cfg);

        tracker = slam::Tracker::create_hybrid(seq.camera, map, std::move(xfeat), std::move(lg),
                                               std::move(ttt));
        std::cout << "[VSLAM] Hybrid deep-geometric mode enabled\n";
    } else
#endif
    {
        tracker = slam::Tracker::create(seq.camera, map);
    }

    std::string gt_path = derive_gt_path(args.sequence_path);

    if (!args.no_viz) {
        viz = slam::Visualizer::create();
        viz->log_pinhole(seq.camera);

        auto gt_centers_viz = load_gt_centers(gt_path);
        if (!gt_centers_viz.empty()) {
            viz->log_ground_truth(gt_centers_viz);
        }
    }

    int n_frames = static_cast<int>(seq.image_paths.size());
    int start_idx = std::max(0, args.start_idx);
    int end_idx = (args.end_idx < 0) ? n_frames : std::min(args.end_idx, n_frames);

    auto gt_centers_metrics = load_gt_centers(gt_path);
    auto gt_poses = load_gt_poses(gt_path);

    std::vector<std::array<float, 3>> est_centers(n_frames, {0.f, 0.f, 0.f});
    std::vector<bool> est_valid(n_frames, false);

    // --- per-frame metric accumulators ---
    double prev_yaw_gt = 0.0, prev_yaw_est = 0.0;
    bool prev_yaw_valid = false;
    double ate_turn_sum2 = 0.0, ate_straight_sum2 = 0.0;
    int ate_turn_n = 0, ate_straight_n = 0;
    double max_yaw_err = 0.0;
    double final_yaw_err = 0.0;
    int lost_count = 0;

    // --- main tracking loop ---
#ifdef ENABLE_DEEP_FRONTEND
    deep::SemiDenseDisparity::Config sd_cfg;
    sd_cfg.baseline = (float)seq.camera.baseline;
    sd_cfg.fx       = (float)seq.camera.fx;
    sd_cfg.cx       = (float)seq.camera.cx;
    sd_cfg.cy       = (float)seq.camera.cy;
    deep::SemiDenseDisparity semi_dense(sd_cfg);
#endif

    long frame_count = 0;
    auto t_start_wall = std::chrono::steady_clock::now();

    for (int i = start_idx; i < end_idx; ++i) {
        cv::Mat img = cv::imread(seq.image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VSLAM] Failed to load: " << seq.image_paths[i] << "\n";
            continue;
        }

        double ts = (i < (int)seq.timestamps.size()) ? seq.timestamps[i] : (double)i;
        auto frame = slam::Frame::create(img, ts, i);

        if (i < (int)seq.image_right_paths.size()) {
            frame->image_right = cv::imread(seq.image_right_paths[i], cv::IMREAD_GRAYSCALE);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        bool ok = tracker->track(frame);

        // trajectory segmentation is handled automatically in log_trajectory() via gap detection

        auto t1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // BA + PGO after each KF insertion
        if (frame->is_keyframe && map->num_keyframes() >= 2) {
            local_ba->optimize();
            tracker->notify_ba_update();

            pose_graph->add_keyframe(frame);
            if (map->num_keyframes() % 5 == 0) {
                pose_graph->detect_and_add_loops();
                if (pose_graph->has_new_loops()) {
                    pose_graph->optimize();
                    tracker->notify_ba_update();
                }
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double ba_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (viz) {
            viz->log_frame(frame);
            viz->log_trajectory(map, frame, ts);
            if (frame->is_keyframe) {
                viz->log_map(map, ts);

#ifdef ENABLE_DEEP_FRONTEND
                // semi-dense disparity — viz only, never touches the map
                if (args.hybrid && !frame->feat_map_left.empty() &&
                    !frame->feat_map_right.empty()) {
                    auto sd_pts = semi_dense.compute(frame->feat_map_left, frame->feat_map_right,
                                                     frame->T_wc());
                    viz->log_semi_dense(sd_pts, ts);

                    frame->feat_map_left.release();  // free after use — these are ~3MB each
                    frame->feat_map_right.release();
                }
#endif
            }
        }

        ++frame_count;

        Eigen::Vector3d pos = frame->camera_center();
        est_centers[i] = {(float)pos.x(), (float)pos.y(), (float)pos.z()};
        est_valid[i] = true;

        // --- per-frame yaw & ATE diagnostics vs GT ---
        if (i < (int)gt_poses.size()) {
            Eigen::Matrix3d R_wc_est = frame->T_wc().rotation();
            Eigen::Matrix3d R_wc_gt = gt_poses[i].rotation();
            double yaw_est = extract_yaw(R_wc_est) * 180.0 / 3.14159265358979323846;
            double yaw_gt = extract_yaw(R_wc_gt) * 180.0 / 3.14159265358979323846;
            double yaw_err = yaw_est - yaw_gt;
            while (yaw_err > 180.0) yaw_err -= 360.0;
            while (yaw_err < -180.0) yaw_err += 360.0;

            // turn = GT yaw rate > 0.5 deg/frame
            bool is_turn = false;
            if (prev_yaw_valid) {
                double yaw_rate_gt = yaw_gt - prev_yaw_gt;
                while (yaw_rate_gt > 180.0) yaw_rate_gt -= 360.0;
                while (yaw_rate_gt < -180.0) yaw_rate_gt += 360.0;
                is_turn = std::abs(yaw_rate_gt) > 0.5;
            }
            prev_yaw_gt = yaw_gt;
            prev_yaw_est = yaw_est;
            prev_yaw_valid = true;

            double frame_ate = 0.0;
            if (i < (int)gt_centers_metrics.size()) {
                double dx = pos.x() - gt_centers_metrics[i][0];
                double dy = pos.y() - gt_centers_metrics[i][1];
                double dz = pos.z() - gt_centers_metrics[i][2];
                frame_ate = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (is_turn) {
                    ate_turn_sum2 += frame_ate * frame_ate;
                    ++ate_turn_n;
                } else {
                    ate_straight_sum2 += frame_ate * frame_ate;
                    ++ate_straight_n;
                }
            }

            if (std::abs(yaw_err) > std::abs(max_yaw_err)) max_yaw_err = yaw_err;
            final_yaw_err = yaw_err;

            double scale_ratio = 0.0;
            if (i > start_idx && i < (int)gt_centers_metrics.size() && est_valid[i - 1]) {
                double e_dx = est_centers[i][0] - est_centers[i - 1][0];
                double e_dy = est_centers[i][1] - est_centers[i - 1][1];
                double e_dz = est_centers[i][2] - est_centers[i - 1][2];
                double e_len = std::sqrt(e_dx * e_dx + e_dy * e_dy + e_dz * e_dz);
                double g_dx = gt_centers_metrics[i][0] - gt_centers_metrics[i - 1][0];
                double g_dy = gt_centers_metrics[i][1] - gt_centers_metrics[i - 1][1];
                double g_dz = gt_centers_metrics[i][2] - gt_centers_metrics[i - 1][2];
                double g_len = std::sqrt(g_dx * g_dx + g_dy * g_dy + g_dz * g_dz);
                if (g_len > 0.01) scale_ratio = e_len / g_len;
            }

            fprintf(stderr,
                    "[DIAG] frame=%d yaw_est=%.2f yaw_gt=%.2f err=%.2f ate=%.2f scale=%.4f %s\n", i,
                    yaw_est, yaw_gt, yaw_err, frame_ate, scale_ratio, is_turn ? "TURN" : "");
        }

        if (!ok) ++lost_count;

        fprintf(stderr,
                "[%05d] track=%.1fms ba=%.1fms tracked=%3d kf=%zu pts=%zu "
                "pos=(%.2f,%.2f,%.2f) %s\n",
                i, track_ms, ba_ms, frame->num_tracked(), map->num_keyframes(),
                map->num_map_points(), pos.x(), pos.y(), pos.z(), ok ? "OK" : "LOST");
    }

    // --- summary ---
    auto t_end_wall = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end_wall - t_start_wall).count();
    double fps = frame_count / elapsed_s;

    std::cout << "\n[VSLAM] Done. " << frame_count << " frames in " << elapsed_s << "s = " << fps
              << " FPS\n"
              << "  Keyframes : " << map->num_keyframes() << "\n"
              << "  Map points: " << map->num_map_points() << "\n";

    // --- metrics: ATE & RPE ---
    if (!gt_centers_metrics.empty()) {
        double ate_sum2 = 0.0;
        double y_max_dev = 0.0;
        int ate_count = 0;
        double y_ref = gt_centers_metrics.empty() ? 0.0 : gt_centers_metrics[0][1];

        for (int i = start_idx; i < end_idx && i < (int)gt_centers_metrics.size(); ++i) {
            if (!est_valid[i]) continue;
            double dx = est_centers[i][0] - gt_centers_metrics[i][0];
            double dy = est_centers[i][1] - gt_centers_metrics[i][1];
            double dz = est_centers[i][2] - gt_centers_metrics[i][2];
            ate_sum2 += dx * dx + dy * dy + dz * dz;
            y_max_dev = std::max(y_max_dev, std::abs((double)est_centers[i][1] - y_ref));
            ++ate_count;
        }

        if (ate_count > 0) {
            double ate_rmse = std::sqrt(ate_sum2 / ate_count);
            std::cout << "\n[Metrics] ATE RMSE: " << ate_rmse << " m  (over " << ate_count
                      << " frames)\n";
            std::cout << "[Metrics] Max Y deviation from start: " << y_max_dev << " m\n";
        }

        // RPE over 100-frame windows (KITTI convention)
        const int rpe_delta = 100;
        double rpe_t_sum2 = 0.0;
        double rpe_seg_dist_sum = 0.0;
        int rpe_count = 0;

        for (int i = start_idx;
             i + rpe_delta < end_idx && i + rpe_delta < (int)gt_centers_metrics.size(); ++i) {
            if (!est_valid[i] || !est_valid[i + rpe_delta]) continue;

            double gt_dx = gt_centers_metrics[i + rpe_delta][0] - gt_centers_metrics[i][0];
            double gt_dy = gt_centers_metrics[i + rpe_delta][1] - gt_centers_metrics[i][1];
            double gt_dz = gt_centers_metrics[i + rpe_delta][2] - gt_centers_metrics[i][2];
            double gt_len = std::sqrt(gt_dx * gt_dx + gt_dy * gt_dy + gt_dz * gt_dz);
            if (gt_len < 1.0) continue; // skip near-stationary

            double e_dx = est_centers[i + rpe_delta][0] - est_centers[i][0];
            double e_dy = est_centers[i + rpe_delta][1] - est_centers[i][1];
            double e_dz = est_centers[i + rpe_delta][2] - est_centers[i][2];

            double err_dx = e_dx - gt_dx;
            double err_dy = e_dy - gt_dy;
            double err_dz = e_dz - gt_dz;
            double err = std::sqrt(err_dx * err_dx + err_dy * err_dy + err_dz * err_dz);
            rpe_t_sum2 += (err / gt_len) * (err / gt_len);
            rpe_seg_dist_sum += gt_len;
            ++rpe_count;
        }

        if (rpe_count > 0) {
            double rpe_t_pct = 100.0 * std::sqrt(rpe_t_sum2 / rpe_count);
            std::cout << "[Metrics] RPE_t: " << rpe_t_pct << "%  (over " << rpe_count
                      << " segments of " << rpe_delta << " frames, "
                      << "avg segment " << rpe_seg_dist_sum / rpe_count << " m)\n";
        }

        if (ate_turn_n > 0)
            std::cout << "[Metrics] ATE turn RMSE: " << std::sqrt(ate_turn_sum2 / ate_turn_n)
                      << " m  (" << ate_turn_n << " frames)\n";
        if (ate_straight_n > 0)
            std::cout << "[Metrics] ATE straight RMSE: "
                      << std::sqrt(ate_straight_sum2 / ate_straight_n) << " m  (" << ate_straight_n
                      << " frames)\n";
        std::cout << "[Metrics] Max yaw error: " << max_yaw_err << " deg\n";
        std::cout << "[Metrics] Final yaw error: " << final_yaw_err << " deg\n";
        std::cout << "[Metrics] LOST transitions: " << lost_count << "\n";
    }

    return 0;
}
