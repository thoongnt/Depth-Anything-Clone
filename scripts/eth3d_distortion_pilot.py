"""Pilot: does pairing UNDISTORTED ETH3D images with the official depth maps
(rendered for the DISTORTED images) create the encoder-independent error floor
seen in Table 2 (paper ETH3D ~0.127)? Scene `pipes` (14 images), ViT-S, CPU.

(a) distorted JPG  + official depth  -> correctly paired
(b) undistorted JPG + official depth -> mis-paired (hypothesis)
Both at stride 3 (depth centre pixel of each 3x3 cell, RGB INTER_AREA), OLS on
1/GT in disparity, metrics in disparity, mean over images.
"""
import os
import subprocess
import sys

sys.modules["xformers"] = None  # CPU inference: force the pure-PyTorch attention path
sys.modules["xformers.ops"] = None
ROOT = r"d:\Master\2025T2\Machinelearning\Depth-Anything-Clone"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from depth_anything.dpt import load_depth_anything  # noqa: E402
from evaluate_nyuv2 import make_transform  # noqa: E402

TAR = r"C:\Windows\System32\tar.exe"
DEPTH = r"data\eth3d_official\pipes_dslr_depth.7z"
DIST = r"data\eth3d_official\pipes_dslr_jpg.7z"
UNDIST = r"data\eth3d\multi_view_training_dslr_undistorted.7z"


def member(archive, name):
    return subprocess.run([TAR, "-xOf", archive, name], capture_output=True, check=True).stdout


def rgb(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.resize(img, (2016, 1344), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def score(pred, depth):
    m = np.isfinite(depth) & (depth > 0)
    gt = 1.0 / depth[m]
    x = pred[m].astype(np.float64)
    xc = x - x.mean()
    a = np.dot(xc, gt - gt.mean()) / np.dot(xc, xc) * xc + gt.mean()
    ok = a > 0
    r = np.where(ok, np.maximum(a / gt, gt / np.where(ok, a, 1)), np.inf)
    return float(np.mean(np.abs(a - gt) / gt)), float(np.mean(r < 1.25))


@torch.inference_mode()
def main():
    model = load_depth_anything("vits", "checkpoints", "cpu")
    tf = make_transform()
    names = [n for n in subprocess.run([TAR, "-tf", DEPTH], capture_output=True, text=True, check=True).stdout.split()
             if n.upper().endswith(".JPG")]
    res = {"a_distorted": [], "b_undistorted": []}
    for n in names:
        stem = n.split("/")[-1]
        depth = np.frombuffer(member(DEPTH, n), "<f4").reshape(4032, 6048)[1::3, 1::3]
        for key, arch, path in (("a_distorted", DIST, f"pipes/images/dslr_images/{stem}"),
                                ("b_undistorted", UNDIST, f"pipes/images/dslr_images_undistorted/{stem}")):
            image = rgb(member(arch, path))
            t = torch.from_numpy(tf({"image": image})["image"]).unsqueeze(0)
            p = F.interpolate(model(t).unsqueeze(1), size=depth.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            res[key].append(score(p, depth))
        print(stem, {k: tuple(round(x, 4) for x in v[-1]) for k, v in res.items()}, flush=True)
    for k, v in res.items():
        a = np.array(v)
        print(f"{k}: n={len(a)} abs_rel={a[:, 0].mean():.4f} delta1={a[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
