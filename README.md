# Stereo-SLAM: A Custom VSLAM Engine

This is my first iteration of implementing a Visual SLAM engine entirely from scratch. Built in C++17 and CUDA, the goal of this project was to dive deep into the math, parallel processing, and architecture required to build a real-time SLAM system.

It's rough around the edges and absolutely has limitations (see below), but it serves as a hands-on demonstration of stitching together modern computer vision, non-linear optimization, and custom GPU acceleration.

![SLAM trajectory vs ground truth on KITTI sequence 00](docs/trajectory.png)

Green: SLAM estimate. Orange: KITTI ground truth.

## Under the Hood

The engine is built around a standard front-end/back-end SLAM architecture, favoring explicit implementations over black-box libraries where possible so I could actually understand the core mechanics:

*   **GPU-Accelerated Feature Matching**: Instead of relying solely on OpenCV's CPU matchers, I wrote a custom CUDA kernel (`hamming_matcher.cu`) for extremely fast binary descriptor comparisons. It handles nearest-neighbor bounding, Lowe's ratio test, and stereo epipolar filtering directly on the GPU.
*   **Front-End Tracking**: Extracts ORB features and uses a constant-velocity motion model to guess the next pose. From there, it relies on a 3D-2D projection search and robust PnP-RANSAC (using SQPNP) to figure out exactly where the camera moved.
*   **Stereo Initialization**: Bootstraps the map using single-frame stereo triangulation based on the camera baseline. This grabs metric depth immediately, bypassing the notorious scale-drift nightmare of monocular initialization.
*   **Back-End Optimization**: Uses **Ceres Solver** to run a Local Bundle Adjustment (BA) over a sliding window of recent keyframes. It minimizes stereo reprojection errors while gently enforcing pitch/roll constraints for local consistency.
*   **Global Trajectory Correction**: A Pose Graph Optimizer periodically corrects accumulated drift using co-visibility edges between keyframes.
*   **Real-time Visualization**: Plumbed the entire geometric state (poses, point clouds, trajectory) into the **Rerun SDK**, streaming over TCP so I could watch the math do its work in real-time.

## The Tech Stack

*   **Core**: C++17, CMake (3.20+)
*   **GPU Compute**: CUDA Toolkit (Custom device code optimization)
*   **Math & Optimization**: Eigen3, Ceres Solver
*   **Vision Processing**: OpenCV 4 (Image I/O, ORB extraction)
*   **Telemetry & Viz**: Rerun SDK
*   **Dependencies**: Migrated to `vcpkg` for sane package management in C++.

## Limitations and Flaws

Because this is a first-pass VSLAM implementation, there are distinct trade-offs, flaws, and areas for improvement:

*   **Brittle Tracking in Edge Cases**: The constant-velocity model expects relatively smooth motion. If the camera jerks aggressively or frame rates drop unexpectedly, the 3D-2D projection search can fail to find enough matches to feed the PnP solver, resulting in dropped frames.
*   **ORB Feature Limitations**: ORB is extremely fast, but it struggles in environments with repetitive textures, motion blur, or drastic lighting changes. Modern learned descriptors (like SuperPoint) would be substantially more robust, but would heavily complicate this pure C++/CUDA pipeline.
*   **Loop Closure is Basic**: The Pose Graph Optimization relies heavily on simple co-visibility edges. True global loop closure (e.g., recognizing a place visited 10 minutes ago using a robust Bag-of-Words approach) isn't fully implemented, meaning long-term drift is inevitable on massive maps.
*   **Strict Hardware Requirements**: The custom CUDA matching kernels are tailored specifically to NVIDIA hardware (currently set to compile for Compute 8.6). If you don't have an NVIDIA GPU, a massive parallelized chunk of the pipeline drops out.

## Building and Running

### Prerequisites
*   MSVC 19.x (or modern C++17 compiler)
*   CUDA Toolkit 12.x
*   [vcpkg](https://github.com/microsoft/vcpkg)

### Installation

Install the required packages using vcpkg:
```bash
vcpkg install opencv4[core,features2d,calib3d,highgui] ceres[eigensparse,schur] eigen3 --triplet x64-windows
```

### Compilation

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Running on KITTI
The engine expects standard KITTI Odometry dataset directories:

```bash
build\Release\vslam.exe --sequence /path/to/kitti/dataset/sequences/00
```
*Launch the [Rerun viewer](https://rerun.io/) alongside it. The engine streams the 3D world actively to `127.0.0.1:9876`.*
