# Stereo Visual SLAM

A ground-up stereo SLAM engine in C++17 and CUDA. Runs the full pipeline in real time on the KITTI odometry benchmark: GPU-accelerated ORB matching, metric stereo initialization, constant-velocity tracking with PnP-RANSAC, sliding-window bundle adjustment, pose graph optimization, and 3D visualization.

![SLAM trajectory vs ground truth on KITTI sequence 00](docs/trajectory.png)

*Green: SLAM estimate. Orange: KITTI ground truth. White: active map points.*

---

## How it works

### GPU Hamming Matching

All descriptor matching runs on a custom CUDA kernel. One thread block per query descriptor stripes over the train set in parallel, then warp-shuffles down to the best and second-best match. Three variants: nearest-neighbor, Lowe ratio test, and stereo epipolar (filters by row distance and disparity range before computing Hamming distance).

### Stereo Initialization

Metric depth from a single stereo frame using Z = fx * b / d. No temporal baseline, no scale normalization. The baseline b comes from the P1 projection matrix in calib.txt (b = -P1[3] / fx, about 0.537 m on KITTI).

### Tracking

Each frame predicts a pose with the constant-velocity model then runs two-phase matching: phase 1 projects the local map point pool onto the predicted pose and searches a spatial grid (radius scales with predicted rotation angle); phase 2 falls back to GPU Hamming with ratio test if phase 1 comes up short. PnP-RANSAC with SQPNP refines the pose, followed by a project-and-search pass that pulls in additional map points without a second RANSAC.

After bundle adjustment, the velocity is invalidated so the stale inter-KF delta (a BA correction, not physical motion) is never used as a prediction.

### Bundle Adjustment

Sliding 30-KF window, Ceres SPARSE_SCHUR. Three cost terms per keyframe:
- **Stereo reprojection** (3 residuals: u_L, v_L, u_R) with analytical Jacobians
- **Pitch/roll constraint** (2 residuals: R[3] and R[5], both zero when level) so the camera does not tilt implausibly at turns
- **Pose prior** (6 residuals) softly anchors each KF to its pre-BA PnP estimate to prevent divergence in low-feature regions

Huber loss threshold halves at sharp turns to downweight distant features with inflated reprojection error. Post-BA, any point with reprojection error over 6 px or stereo depth over 150 m is marked bad.

### Pose Graph Optimization

Every 5 keyframes, co-visibility edges are added between any KF pair outside the local BA window that shares at least 15 map points. A relative pose cost (SE3 log error, 6 residuals) is minimized with SPARSE_NORMAL_CHOLESKY.

### Relocalization and Recovery

On tracking loss the system coasts for up to 8 frames using the last velocity estimate. If coasting fails, it builds a descriptor pool from the entire map, runs GPU matching against the current frame, and attempts PnP with a 30-inlier threshold. If relocalization also fails, the map resets: keyframes are archived to a persistent trajectory history so the Rerun visualization never loses its path, then the system reinitializes from the last known pose.

---

## Performance

| Metric | Value |
|---|---|
| Frame rate | >60 FPS (RTX 3050 laptop, KITTI 1241x376) |
| Trajectory drift | <1.5% (KITTI seq 00, 3.7 km loop) |
| GPU matching | ~2 ms (2000x2000 descriptors) |
| Bundle adjustment | ~30 ms (30 KFs, SPARSE_SCHUR) |

---

## Dependencies

| Dependency | Version |
|---|---|
| MSVC | 19.x (VS 2022) |
| CUDA Toolkit | 12.x |
| CMake | 3.20+ |
| OpenCV | 4.x (core, features2d, calib3d, highgui) via vcpkg |
| Ceres Solver | 2.x (eigensparse + schur) via vcpkg |
| Eigen3 | 3.4+ via vcpkg |
| Rerun SDK | 0.22.1 via CMake FetchContent (auto) |

GPU target: Compute Capability 8.6 (RTX 30xx/40xx). Change `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt for other cards.

---

## Build and Run

```bat
:: install vcpkg dependencies
cd C:\Users\<you>\vcpkg
vcpkg install opencv4[core,features2d,calib3d,highgui] --triplet x64-windows
vcpkg install ceres[eigensparse,schur] --triplet x64-windows
vcpkg install eigen3 --triplet x64-windows
vcpkg integrate install

:: configure and build
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/Users/<you>/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Rerun SDK is downloaded automatically on first configure.

Place KITTI data under:

```
VSLAM/data/dataset/
  poses/00.txt
  sequences/00/
    calib.txt
    times.txt
    image_0/000000.png ...
    image_1/000000.png ...
```

```bat
cd C:\...\VSLAM
build\Release\vslam.exe --sequence data/dataset/sequences/00
```

Launch Rerun before or alongside SLAM. The viewer connects to `127.0.0.1:9876`.

| Flag | Default | Description |
|---|---|---|
| `--sequence <path>` | required | KITTI sequence directory |
| `--start <N>` | 0 | First frame index |
| `--end <N>` | last | Last frame index |
| `--no-viz` | off | Disable Rerun visualization |
