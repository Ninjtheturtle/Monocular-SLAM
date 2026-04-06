"""
Step 3: Export XFeat (from accelerated_features repo) to ONNX,
        then to a TensorRT FP16 engine.

Run:  python setup/03_export_models.py
Out:  models/xfeat_fp16.engine
"""

import subprocess, sys, os, tempfile
from pathlib import Path

ROOT     = Path(__file__).parent.parent
MODELS   = ROOT / "models"
ONNX_DIR = MODELS / "onnx"
MODELS.mkdir(exist_ok=True)
ONNX_DIR.mkdir(exist_ok=True)

TMP = Path(tempfile.gettempdir())
AF_DIR = TMP / "accelerated_features"   # XFeat repo


def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


# ── install deps ──────────────────────────────────────────────────────────────
print("Installing Python deps...")
pip_install("onnx>=1.16", "onnxruntime-gpu", "kornia", "einops")

if not AF_DIR.exists():
    print("Cloning accelerated_features (XFeat)...")
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/verlab/accelerated_features.git", str(AF_DIR)])
    # Add minimal setup.py so pip can install it
    (AF_DIR / "setup.py").write_text(
        "from setuptools import setup, find_packages\n"
        "setup(name='accelerated_features', version='1.0', packages=['modules'])\n"
    )

pip_install("-e", str(AF_DIR), "--quiet")

import torch, tensorrt as trt, torch.nn as nn
print(f"PyTorch  : {torch.__version__}")
print(f"TensorRT : {trt.__version__}")

from modules.xfeat import XFeat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 1242, 376
MAX_KPS      = 2000

# ── 1. Export XFeat ───────────────────────────────────────────────────────────
print("\n=== Exporting XFeat to ONNX ===")

class XFeatWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.xfeat = XFeat()
        self.xfeat.eval()

    def forward(self, x: torch.Tensor):
        # x: (1,1,H,W) float32 [0,1]
        out = self.xfeat.detectAndCompute(x, top_k=MAX_KPS)
        kps   = out[0]['keypoints'].unsqueeze(0)    # (1,N,2)
        scores = out[0]['scores'].unsqueeze(0)       # (1,N)
        descs  = out[0]['descriptors'].unsqueeze(0)  # (1,N,64)
        return kps, scores, descs

xfeat_wrapper = XFeatWrapper().to(DEVICE).eval()
dummy_img     = torch.zeros(1, 1, IMG_H, IMG_W, device=DEVICE)
xfeat_onnx    = str(ONNX_DIR / "xfeat.onnx")

print(f"  Tracing -> {xfeat_onnx}")
with torch.no_grad():
    torch.onnx.export(
        xfeat_wrapper, dummy_img, xfeat_onnx,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes={
            "image":       {0: "batch"},
            "keypoints":   {0: "batch", 1: "N"},
            "scores":      {0: "batch", 1: "N"},
            "descriptors": {0: "batch", 1: "N"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print("  XFeat ONNX OK")

# ── 2. ONNX → TRT FP16 ───────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_fp16_engine(onnx_path, engine_path, min_sh, opt_sh, max_sh):
    print(f"\n  Building TRT: {Path(engine_path).name}")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print("  ONNX error:", parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    profile = builder.create_optimization_profile()
    for name in min_sh:
        profile.set_shape(name, min_sh[name], opt_sh[name], max_sh[name])
    config.add_optimization_profile(profile)

    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TRT build failed")
    with open(engine_path, "wb") as f:
        f.write(data)
    sz = os.path.getsize(engine_path) / 1e6
    print(f"  Saved ({sz:.1f} MB): {engine_path}")


print("\n=== Building XFeat TRT FP16 engine ===")
build_fp16_engine(
    xfeat_onnx, str(MODELS / "xfeat_fp16.engine"),
    min_sh={"image": (1, 1, IMG_H, IMG_W)},
    opt_sh={"image": (1, 1, IMG_H, IMG_W)},
    max_sh={"image": (2, 1, IMG_H, IMG_W)},
)

print("\n=== DONE ===")
print(f"  {MODELS}/xfeat_fp16.engine")
print("\nBuild the project:")
print("  cmake -B build_hybrid -DENABLE_DEEP_FRONTEND=ON -DTRT_ROOT=C:/TensorRT")
print('           -DTorch_DIR="C:/libtorch/share/cmake/Torch" ..')
print("  cmake --build build_hybrid --config Release")
print("\nRun:")
print("  build_hybrid\\Release\\vslam.exe --sequence data/dataset/sequences/00")
print("    --hybrid --xfeat models/xfeat_fp16.engine")
