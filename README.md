# Stereo Visual SLAM — from scratch

<!-- VIDEO: drag-and-drop the mp4 into a GitHub issue comment to get a user-attachments URL, then paste it here -->

> KITTI sequence 00 — green is my estimate, orange is ground truth, white dots are the live map point cloud

---

## why i built this

i wanted to understand how SLAM actually works — not from a textbook, not from wrapping someone else's library, but by writing every piece myself. the tracker, the bundle adjustment, the GPU kernels, the map, the state machine, all of it. this is a stereo visual SLAM system written from scratch in C++17 and CUDA. it takes raw stereo images from the KITTI odometry benchmark and estimates the camera trajectory in real time while building a 3D map of the environment.

it isn't ORB-SLAM3. it doesn't have loop closure detection. it doesn't have an atlas or IMU fusion or relocalization via bag-of-words. what it does have is a clean, readable implementation of the core SLAM pipeline that i actually understand end to end.

---

## what it actually does

**stereo in, trajectory out.** left and right grayscale images come in, and the system:

1. extracts XFeat deep features (2000 keypoints/frame) via a TorchScript backbone running on GPU
2. matches stereo pairs on the GPU with epipolar constraints to get metric depth (`Z = fx * b / d`)
3. tracks frame-to-frame using a constant-velocity model + PnP-RANSAC
4. runs sliding-window bundle adjustment (30 keyframes, Ceres solver, analytical Jacobians)
5. checks for new co-visibility constraints every 5 keyframes and runs pose-graph optimization when found
6. streams everything to [Rerun](https://rerun.io/) for live 3D visualization

no scale ambiguity — stereo gives you real-world meters from frame one. the KITTI baseline is ~0.54m and it triangulates 1000+ map points on the very first frame.

there's also a classical ORB fallback mode for running without PyTorch.

---

## the gpu side

i wrote three CUDA kernels from scratch (Thrust used for the final sort in ANMS, everything else is plain CUDA):

- **FP16 L2 matcher** — one block per query descriptor, 256 threads, warp-shuffle butterfly reduction. half2 vectorization halves memory bandwidth, accumulates in float32 to avoid overflow. used for XFeat temporal matching and relocalization

- **stereo epipolar matcher** — same architecture but rejects candidate matches that violate row alignment or disparity range before computing distance. runs on every frame to get metric stereo depth

- **adaptive NMS** — shared-memory tile filter for XFeat heatmap peaks. pass 1 marks local maxima in a (2R+1)² neighbourhood, pass 2 stream-compacts via atomicAdd, thrust sorts by response to keep top-K

the L2 matchers run fully asynchronously on CUDA streams. the NMS syncs once between passes to read the candidate count back to the CPU.

---

## bundle adjustment

the BA is the part i'm most proud of and the part that took the longest to get right. it's a sliding-window optimizer over the last 30 keyframes using Ceres with SPARSE_SCHUR.

the analytical Jacobians are hand-derived — the stereo cost function has 3 residuals (left u, left v, right u) and i work out the full chain rule through the quaternion rotation, projection, and disparity. it's not pretty but it works and it's fast.

there's also a Schur complement marginalization prior that summarizes information from keyframes that slide out of the window, so the system doesn't just forget everything behind it.

what the BA does NOT have: any kind of pitch/roll constraint or pose prior. i tried both — they caused more problems than they solved. the BA runs unconstrained (except fixing the oldest keyframe for gauge freedom) and that turns out to be enough.

---

## the state machine

tracking isn't just "run PnP every frame." the system has five states:

- **NOT_INITIALIZED** — waiting for a good stereo frame to bootstrap
- **OK** — normal tracking, constant-velocity prediction + PnP refinement
- **COASTING** — lost the current frame but predicting forward with the last velocity for up to 8 frames, hoping to recover
- **LOST** — coasting failed, attempting synchronous relocalization against the full map then resetting

when relocalization fails, the map resets — but archived keyframes are preserved so the trajectory visualization never disappears. the system reinitializes from the last known pose and keeps going.

---

## what's honest about performance

i'm not going to quote drift percentages i haven't rigorously measured. here's what i can say:

- it runs. it tracks through KITTI sequence 00 (4541 frames, ~3.7km) without getting permanently lost
- stereo initialization gives metric scale from frame one — no scale drift from monocular bootstrapping
- it's ~6-7 FPS on my RTX 3050 laptop — the TorchScript XFeat inference is the bottleneck
- the trajectory visually follows ground truth closely on straight sections and drifts at sharp turns (no loop closure to correct this)
- it handles the KITTI 00 highway-to-residential transition without dying, which was genuinely hard

things that don't work well:
- no loop closure means drift accumulates and never gets corrected
- i haven't tested on anything other than KITTI

---

## building it

you need Windows, MSVC (VS 2022), CUDA 12.x, vcpkg, and PyTorch 2.1+ with CUDA.

```bat
:: pytorch (required for XFeat deep frontend)
pip install torch --index-url https://download.pytorch.org/whl/cu118

:: vcpkg dependencies
vcpkg install opencv4[core,features2d,calib3d,highgui] --triplet x64-windows
vcpkg install ceres[eigensparse,schur] --triplet x64-windows
vcpkg install eigen3 --triplet x64-windows

:: build
cmake -B build -DENABLE_DEEP_FRONTEND=ON -DCMAKE_TOOLCHAIN_FILE=<your-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Rerun SDK (0.22.1) downloads automatically via FetchContent — no separate install needed.

to build without PyTorch (classical ORB mode only), omit `-DENABLE_DEEP_FRONTEND=ON`.

---

## running it

```bat
:: download KITTI odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
:: place under data/dataset/sequences/00/ (with image_0/, image_1/, calib.txt, times.txt)

.\build\vslam.exe --sequence data/dataset/sequences/00 --hybrid --xfeat models/xfeat.pt
```

launch Rerun (`rerun`) before or alongside — the viewer connects on `127.0.0.1:9876`.

flags: `--start N`, `--end N`, `--no-viz`

---

## project structure

```
VSLAM/
├── src/                    C++ implementations (tracker, BA, map, PGO, visualizer, main)
├── include/slam/           headers for the SLAM pipeline
├── include/cuda/           CUDA kernel declarations
├── cuda/                   GPU kernels (L2 FP16, stereo epipolar, adaptive NMS)
├── include/deep/           deep frontend headers (XFeat, semi-dense)
├── models/                 exported TorchScript models
├── data/dataset/           KITTI sequences + ground truth poses
└── CMakeLists.txt          build config (CUDA arch=86, vcpkg, FetchContent Rerun)
```

---

## where this is going

this is a personal project. i built it to learn, and i learned a lot — about how fragile visual tracking really is, about how much of SLAM is just making the edge cases not explode, about how analytical Jacobians are worth the pain. i'm still working on it. the next things i want to tackle are loop closure, TensorRT for the deep frontend, and testing on more sequences.

if you read this far — thanks. feel free to look around the code. it's not perfect but it's mine.
