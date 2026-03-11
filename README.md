# Stereo Visual SLAM

A ground-up stereo Visual SLAM engine in C++17 and CUDA targeting real-time performance on the KITTI odometry benchmark. Implements the full loop: GPU-accelerated ORB feature matching → metric stereo initialization → constant-velocity tracking with PnP-RANSAC → sliding-window bundle adjustment → pose-graph optimization → real-time 3D visualization.

![SLAM trajectory vs ground truth — KITTI sequence 00](docs/trajectory.png)

*Green: SLAM estimate. Orange: KITTI ground truth. White: active map point cloud.*

---

## Features

- **Stereo metric scale** — single-frame triangulation gives absolute depth (no scale drift)
- **GPU-accelerated ORB matching** — custom CUDA kernel with warp-shuffle reduction; handles stereo epipolar constraints and ratio test natively on-device
- **Two-phase tracking** — projection-based spatial search (Phase 1) with GPU Hamming fallback (Phase 2); adaptive search radius scales with predicted rotation
- **Analytical-Jacobian bundle adjustment** — Ceres SPARSE_SCHUR, 30-KF sliding window, turn-adaptive Huber loss, pitch/roll soft constraint
- **Pose-graph optimization** — co-visibility loop edges, SPARSE_NORMAL_CHOLESKY, runs every 5 keyframes
- **Robust LOST recovery** — 8-frame coasting → global relocalization (≥30 PnP inliers) → map reset with pose propagation
- **Trajectory persistence** — map resets archive old keyframes so the visualization never loses history
- **Real-time visualization** — [Rerun 0.22.1](https://rerun.io/) with per-frame camera frustum, keypoints, trajectory, map cloud, and GT overlay

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        main loop                        │
│   KittiSequence → Frame  ──►  Tracker::track()          │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   Tracker          │
          │  ┌──────────────┐  │     GPU
          │  │ ORB extract  │──┼──► cuda_match_stereo_epipolar()
          │  │ stereo match │  │     (epipolar + disparity filter)
          │  └──────┬───────┘  │
          │         │          │
          │  ┌──────▼───────┐  │
          │  │ init / track │  │
          │  │ motion model │  │
          │  │ PnP-RANSAC   │  │
          │  └──────┬───────┘  │
          └─────────┼──────────┘
                    │ keyframe?
          ┌─────────▼──────────┐
          │   LocalBA          │   Ceres SPARSE_SCHUR
          │   30-KF window     │   StereoReprojCost (3-residual)
          │   analytical Jac.  │   PitchRollCost (2-residual)
          │   Huber loss       │   PosePriorCost  (6-residual)
          └─────────┬──────────┘
                    │ every 5 KFs
          ┌─────────▼──────────┐
          │   PoseGraph        │   Ceres SPARSE_NORMAL_CHOLESKY
          │   co-vis edges     │   RelPoseCost (6-residual)
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   Visualizer       │   Rerun TCP → localhost:9876
          └────────────────────┘
```

---

## Dependencies

| Dependency | Version | Source |
|---|---|---|
| MSVC | 19.x (VS 2022) | — |
| CUDA Toolkit | 12.x | — |
| CMake | 3.20+ | — |
| vcpkg | latest | — |
| OpenCV | 4.x (core, features2d, calib3d, highgui) | vcpkg |
| Ceres Solver | 2.x (eigensparse + schur) | vcpkg |
| Eigen3 | 3.4+ | vcpkg |
| Rerun SDK | 0.22.1 | CMake FetchContent (auto) |

> GPU target: Compute Capability 8.6 (RTX 30xx/40xx). Change `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt) for other cards (e.g. `75` for RTX 20xx, `89` for RTX 40xx).

---

## Building

```bat
:: 1. Install vcpkg dependencies
cd C:\Users\<you>\vcpkg
vcpkg install opencv4[core,features2d,calib3d,highgui] --triplet x64-windows
vcpkg install ceres[eigensparse,schur] --triplet x64-windows
vcpkg install eigen3 --triplet x64-windows
vcpkg integrate install

:: 2a. VS Code
:: Ctrl+Shift+P → "CMake: Configure" → "CMake: Build"

:: 2b. Developer Command Prompt
cmake -B build ^
  -DCMAKE_TOOLCHAIN_FILE=C:/Users/<you>/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Rerun SDK (~50 MB) is downloaded automatically on first configure.

---

## Dataset Setup

Download [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and place it under:

```
VSLAM/data/dataset/
  poses/00.txt              ← ground-truth (optional, for overlay)
  sequences/00/
    calib.txt               ← camera calibration (P0, P1 matrices)
    times.txt               ← per-frame timestamps
    image_0/000000.png …    ← left grayscale
    image_1/000000.png …    ← right grayscale (enables stereo mode)
```

Stereo mode activates automatically when `image_1/` is present. Without it, the system falls back to monocular initialization with median-depth scale normalization.

---

## Usage

```bat
cd C:\...\VSLAM
build\Release\vslam.exe --sequence data/dataset/sequences/00
```

| Flag | Default | Description |
|---|---|---|
| `--sequence <path>` | required | KITTI sequence directory |
| `--start <N>` | 0 | First frame index |
| `--end <N>` | last | Last frame index |
| `--no-viz` | off | Disable Rerun visualization |

Launch **Rerun** before or alongside SLAM. The viewer connects to `127.0.0.1:9876`. Per-frame console output:

```
[00150] track=95.0ms ba=0.0ms tracked=146 kf=13 pts=9052 pos=(15.92,-2.13,87.11) OK
```

---

## Project Structure

```
VSLAM/
├── CMakeLists.txt                 ← CUDA arch=86, vcpkg, FetchContent Rerun 0.22.1
├── vcpkg.json                     ← opencv4, ceres[eigensparse,schur], eigen3
├── include/
│   ├── slam/
│   │   ├── camera.hpp             ← pinhole model + KITTI calibration loader
│   │   ├── frame.hpp              ← image frame + keypoints + T_cw pose
│   │   ├── map_point.hpp          ← 3D landmark with observation tracking
│   │   ├── map.hpp                ← thread-safe KF + map point containers
│   │   ├── tracker.hpp            ← tracking state machine + Config
│   │   ├── local_ba.hpp           ← sliding-window BA + analytical cost structs
│   │   ├── pose_graph.hpp         ← PGO edges + Config
│   │   └── visualizer.hpp         ← Rerun logging interface
│   └── cuda/
│       └── hamming_matcher.cuh    ← GPU kernel declarations + CUDA_CHECK macro
├── src/
│   ├── main.cpp                   ← KITTI loader + main loop
│   ├── camera.cpp                 ← from_kitti_calib(), project(), unproject()
│   ├── tracker.cpp                ← full tracking pipeline (~1000 lines)
│   ├── local_ba.cpp               ← Ceres bundle adjustment (~570 lines)
│   ├── map.cpp                    ← map management + trajectory archive
│   ├── pose_graph.cpp             ← loop detection + PGO
│   └── visualizer.cpp             ← Rerun entity logging
└── cuda/
    └── hamming_matcher.cu         ← CUDA kernels: Hamming match, ratio test, stereo
```

---

## Algorithms & Math

### 1. Feature Extraction — ORB

Each frame extracts up to 2000 ORB keypoints on an 8-level pyramid (scale 1.2×). ORB produces binary 256-bit (32-byte) descriptors. The same detector runs on the right image for stereo matching.

```
Descriptor: 32 × uint8  =  256 bits  (used as 8 × uint32 on GPU)
```

---

### 2. GPU Hamming Distance Matching

All descriptor matching runs on a custom CUDA kernel.

#### Kernel Design (`cuda/hamming_matcher.cu`)

**Launch geometry:**
```
Grid:  (N_query, 1, 1)   — one block per query descriptor
Block: (256, 1, 1)        — 4 warps; fills a full SM on CC 8.6
Smem:  32 bytes query + 64 bytes warp reduction = 96 bytes/block
```

**Per-block algorithm:**
```
1. Thread 0–7: load query[0..7] (8 × uint32) into shared memory
2. __syncthreads()
3. Each thread i strides through train descriptors:
     for j = i, i+256, i+512, ...:
         dist = Σ __popc(q[k] XOR train[j*8+k])   k=0..7
         if dist < local_best: local_best = dist
4. Warp-level reduction via __shfl_down_sync():
     — butterfly reduction within 32-lane warp
     — pack (dist << 32 | idx) as uint64 → single min() call
5. Thread 0 in each warp writes to smem[warp_id]
6. Thread 0 reads 8 warp winners → final minimum → output
```

**Variants:**
- `hamming_match_kernel` — best match only (all correspondences)
- `hamming_match_ratio_kernel` — tracks best + second-best; applies Lowe's ratio test (0.75) on thread 0
- `hamming_stereo_kernel` — adds epipolar band filter (`|Δy| ≤ epi_tol`) and disparity range check (`d_min ≤ x_L − x_R ≤ d_max`) per train candidate

---

### 3. Stereo Initialization

Given a pair of rectified stereo images, metric depth is recovered from disparity **in a single frame** — no temporal baseline needed.

For a matched pair (left keypoint `(u_L, v_L)`, right x-coordinate `u_R`):

$$d = u_L - u_R \qquad \text{(disparity, pixels)}$$

$$Z = \frac{f_x \cdot b}{d}, \quad X = \frac{(u_L - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v_L - c_y) \cdot Z}{f_y}$$

where $b$ is the baseline (KITTI: $b \approx 0.537$ m derived from the P1 projection matrix as $b = -P_1[3] / f_x$).

Constraints applied:
- Epipolar row tolerance: $|v_L - v_R| \leq 2.0$ px
- Disparity range: $3 \leq d \leq 300$ px → depth range $\approx[0.35\text{ m},\ 128\text{ m}]$
- Absolute depth range: $[0.5\text{ m},\ 150\text{ m}]$

Each triangulated point starts with `observed_times = 2` (treated as a two-view verified landmark).

---

### 4. Pose Estimation — PnP-RANSAC

Given $N$ 3D–2D correspondences $\{(\mathbf{X}_i, \mathbf{x}_i)\}$, solve for camera pose $\mathbf{T}_{cw} \in SE(3)$:

$$\mathbf{x}_i = \pi(\mathbf{T}_{cw} \cdot \mathbf{X}_i), \qquad \pi(\mathbf{X}_c) = \begin{bmatrix} f_x X_c / Z_c + c_x \\ f_y Y_c / Z_c + c_y \end{bmatrix}$$

Implementation:
- Solver: `cv::SOLVEPNP_SQPNP` (Sum-of-Squares Polynomial; robust at low inlier counts)
- RANSAC threshold: 5.5 px
- Initial guess: velocity-predicted pose when $|\hat{\omega}| < 0.3$ rad
- Min inliers: 15

Sanity checks after RANSAC:
- $\|\Delta\mathbf{R}\|_{\text{angle-axis}} < 0.5$ rad (~29°)
- $\|\Delta\mathbf{t}\| < 5.0$ m per frame

After PnP, a **project-and-search** pass reprojects all remaining local map points onto the new pose and searches a 15 px radius for unmatched keypoints — typically 3–5× more matches without additional RANSAC.

---

### 5. Constant-Velocity Motion Model

The inter-frame velocity is maintained as a relative transform:

$$\mathbf{V} = \mathbf{T}_{cw}^{(k)} \cdot \left(\mathbf{T}_{cw}^{(k-1)}\right)^{-1}$$

Pose prediction for the next frame:

$$\hat{\mathbf{T}}_{cw}^{(k+1)} = \mathbf{V} \cdot \mathbf{T}_{cw}^{(k)}$$

After bundle adjustment, `velocity_valid_` is set to `false` so the stale inter-KF velocity (which encodes a BA correction, not physical motion) is never used as a prediction.

---

### 6. Sliding-Window Bundle Adjustment

#### Pose Parameterization

Each keyframe pose is stored as a 6-DOF vector:

$$\boldsymbol{\xi} = [\omega_0,\ \omega_1,\ \omega_2,\ t_0,\ t_1,\ t_2]^T$$

where $\boldsymbol{\omega} \in \mathbb{R}^3$ is the angle-axis rotation and $\mathbf{t}$ is the translation. The rotation matrix is recovered via Rodrigues' formula (handled by `ceres::AngleAxisToRotationMatrix`).

Camera projection:

$$\mathbf{X}_c = R(\boldsymbol{\omega})\,\mathbf{X}_w + \mathbf{t}$$

$$u = f_x \frac{X_c}{Z_c} + c_x, \qquad v = f_y \frac{Y_c}{Z_c} + c_y$$

#### Cost Functions

**Monocular Reprojection** (2 residuals, analytical Jacobians):

$$\mathbf{r}_\text{mono} = \begin{bmatrix} u - u^\text{obs} \\ v - v^\text{obs} \end{bmatrix}$$

Analytical Jacobians $\partial \mathbf{r}/\partial \boldsymbol{\xi}$ (2×6) and $\partial \mathbf{r}/\partial \mathbf{X}_w$ (2×3) are derived from the chain rule:

$$\frac{\partial \mathbf{r}}{\partial \boldsymbol{\omega}} = \frac{\partial \pi}{\partial \mathbf{X}_c} \cdot \frac{\partial \mathbf{X}_c}{\partial \boldsymbol{\omega}}, \qquad \frac{\partial \mathbf{X}_c}{\partial \boldsymbol{\omega}} = -[\mathbf{X}_c]_\times \quad \text{(skew-symmetric)}$$

**Stereo Reprojection** (3 residuals, analytical Jacobians):

$$\mathbf{r}_\text{stereo} = \begin{bmatrix} u_L - u_L^\text{obs} \\ v_L - v_L^\text{obs} \\ u_R - u_R^\text{obs} \end{bmatrix}, \qquad u_R = f_x \frac{X_c - b}{Z_c} + c_x$$

The right-camera Jacobian row differs only in the $X_c$ numerator ($X_c - b$ vs $X_c$); all other terms are identical to the left-camera rows.

**Pitch/Roll Soft Constraint** (2 residuals, auto-diff):

Prevents implausible camera tilts (e.g., 30° pitch at turns) without locking height:

$$\mathbf{r}_\text{pr} = w_\text{rp} \begin{bmatrix} R_{10} \\ R_{12} \end{bmatrix}, \qquad w_\text{rp} = 30.0 \ (\sigma \approx 1.9°)$$

$R_{10}$ and $R_{12}$ are entries of the $3\times3$ rotation matrix; both equal zero when the camera is level. Height ($Y$-translation) is left free and constrained naturally by the stereo $v_L$/$v_R$ residuals.

**Pose Prior** (6 residuals, auto-diff):

Soft anchor to the pre-BA PnP estimate $\boldsymbol{\xi}^\text{prior}$:

$$\mathbf{r}_\text{prior} = w \begin{bmatrix} \boldsymbol{\omega} - \boldsymbol{\omega}^\text{prior} \\ \mathbf{t} - \mathbf{t}^\text{prior} \end{bmatrix}, \qquad w = 0.5$$

Prevents BA from diverging when feature observations are sparse (dark turns, near-LOST states).

#### Loss Function

**Huber loss** (robust to outlier residuals):

$$\rho(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta\!\left(|r| - \frac{\delta}{2}\right) & |r| > \delta \end{cases}$$

Turn-adaptive: $\delta = 1.0$ px normally; $\delta = 0.5$ px when the most-recent KF-to-KF rotation exceeds 0.05 rad (~3°). Tighter loss downweights distant features with inflated reprojection error during sharp turns.

#### Solve Configuration

```
Window:          30 keyframes (sliding; oldest pose fixed for gauge freedom)
Linear solver:   SPARSE_SCHUR  (Schur complement eliminates point parameters)
Nonlinear:       LEVENBERG_MARQUARDT
Threads:         4
Max iterations:  60
Post-BA culling: reprojection > 6 px²  → mark map point bad
                 depth > 150 m (stereo) → mark map point bad
```

---

### 7. Pose Graph Optimization

#### Co-Visibility Loop Detection

For each new keyframe, count shared map point observations against all keyframes **outside** the local BA window:

$$|M_A \cap M_B| \geq 15 \implies \text{add edge}\ (A, B)$$

The edge stores the relative pose measurement:

$$\mathbf{T}_{AB}^\text{meas} = \mathbf{T}_{A,cw} \cdot \mathbf{T}_{B,cw}^{-1}$$

#### Relative Pose Cost (6 residuals, auto-diff)

$$\Delta = \mathbf{T}_{AB}^\text{meas} \cdot \mathbf{T}_{B,\text{est}} \cdot \mathbf{T}_{A,\text{est}}^{-1} \qquad (\text{should be } \mathbf{I})$$

$$\mathbf{r}_\text{rel} = \begin{bmatrix} w_r \cdot \boldsymbol{\omega}_\Delta \\ w_t \cdot \mathbf{t}_\Delta \end{bmatrix}, \quad w_r = 50.0,\ w_t = 10.0$$

Run every 5 new keyframes using `SPARSE_NORMAL_CHOLESKY`; oldest keyframe fixed as gauge anchor.

---

### 8. Keyframe Selection

A new keyframe is inserted when either condition holds:

1. $N_\text{tracked} < 80$ (absolute minimum feature count)
2. $N_\text{tracked} / N_\text{tracked,\,prev-KF} < 0.8$ (80% ratio — map point coverage degrading)

After insertion:
- **Multi-baseline triangulation**: match against last 3 KFs, add new map points from unmatched keypoints
- **Stereo enrichment**: re-run `triangulate_stereo()` for any keypoints without a map point

---

### 9. Relocalization

On tracking loss after 8 coasting frames:

1. **Build global pool**: all map points from all keyframes (full map, not sliding window)
2. **GPU match**: full descriptor matching with ratio test against current frame
3. **PnP RANSAC**: solve without initial guess, require ≥ 30 inliers
4. If successful → resume tracking on existing map
5. If failed → `map_->reset()` (archives KFs to trajectory history), propagate last-known pose, reinitialize

`Map::trajectory_archive_` accumulates keyframes from all prior map segments (never cleared), so the visualized trajectory persists through resets.

---

## Visualization

Visualized in real time via [Rerun](https://rerun.io/) over TCP to `localhost:9876`.

| Entity path | Type | Update |
|---|---|---|
| `world/camera/image` | Pinhole + Transform3D + Image | Every frame |
| `world/camera/image/keypoints` | Points2D (purple) | Every frame |
| `world/trajectory` | LineStrips3D (green) | Every frame |
| `world/map/points` | Points3D (white) | Every keyframe |
| `world/ground_truth/trajectory` | LineStrips3D (orange) | Once (static) |

The trajectory is rebuilt from the full archive each frame and split into segments at 50 m spatial gaps to handle reinit discontinuities. Camera poses are logged as `Transform3D` on the Pinhole entity so the frustum moves in the 3D view.

---

## Performance

| Metric | Target | Notes |
|---|---|---|
| Frame rate | >60 FPS | RTX 3050 laptop, KITTI 1241×376 |
| Trajectory drift | <1.5% | KITTI seq 00, 3.7 km loop |
| Map scale | Metric | From stereo baseline |
| Feature extraction | ~8 ms | ORB, 2000 features |
| GPU matching | ~2 ms | CUDA Hamming, 2000×2000 |
| PnP RANSAC | ~5 ms | SQPNP, 200 iterations |
| Bundle adjustment | ~30 ms | 30 KFs, SPARSE_SCHUR |
| Pose graph | ~10 ms | Per 5 KFs, co-visibility edges |

---

## Configuration Reference

All tunable parameters live in `Tracker::Config` ([include/slam/tracker.hpp](include/slam/tracker.hpp)) and `LocalBA::Config` / `PoseGraph::Config` in their respective headers.

| Parameter | Value | Effect |
|---|---|---|
| `orb_features` | 2000 | Features per frame |
| `hamming_threshold` | 60 | Max Hamming for valid match |
| `lowe_ratio` | 0.75 | Ratio test threshold |
| `pnp_reprojection` | 5.5 px | RANSAC inlier threshold |
| `pnp_min_inliers` | 15 | Minimum PnP inliers |
| `stereo_epi_tol` | 2.0 px | Epipolar band for stereo match |
| `stereo_d_min` | 3.0 px | Min disparity (~128 m max depth) |
| `stereo_d_max` | 300.0 px | Max disparity (~0.35 m min depth) |
| `kWindowSize` | 30 KFs | Local BA + tracking pool window |
| `huber_delta` | 1.0 px | BA Huber loss threshold |
| `pgo_interval` | 5 KFs | How often PGO runs |
| `min_shared_points` | 15 | Co-visibility edge threshold |

---

## Known Limitations

- **No visual loop closure**: Co-visibility PGO handles near-revisits; full loop closure requires a visual vocabulary (e.g., DBoW3). The hook exists in `detect_and_add_loops_visual()` but is disabled due to false positives on KITTI's repetitive urban scenes.
- **KITTI-only calibration loader**: The camera loader reads KITTI `calib.txt` format (`P0`/`P1` matrices). Other datasets require a custom loader.
- **Windows / MSVC only**: Build flags and path conventions target MSVC 14.x on Windows 11. Linux port requires minor CMake changes.
- **Monocular fallback**: Without `image_1/`, the system uses Essential Matrix + 20 m median-depth normalization (scale drift will accumulate).
