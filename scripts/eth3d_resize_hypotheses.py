"""ETH3D Table 2: does resizing the GT to the network output size create the
encoder-independent error floor of the paper (0.127 / 0.126 / 0.127)?

Uses the cached predictions of scripts/eval_relative.py (--cache), MoGe ETH3D bundle.
The cached prediction (bilinear upsampled to the GT grid) is brought back to the
network output size, and the GT is resized to that size in several ways:

  nearest          : GT and mask nearest (control)
  lin-depth/nn-mask: depth bilinear (holes = 0 are mixed in), valid mask nearest
  lin-depth/strict : depth bilinear, keep pixels whose bilinear mask == 1
  lin-depth/half   : depth bilinear, keep pixels whose bilinear mask >= 0.5
  area-depth/nn    : depth INTER_AREA (holes mixed in), valid mask nearest
  lin-disp/nn-mask : disparity bilinear (holes = 0 disparity), mask nearest

Scoring: per-image OLS on 1/GT in disparity, metrics in disparity, mean over images.
Usage: python scripts/eth3d_resize_hypotheses.py --encoder vits [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_relative import moge_depth  # noqa: E402

PAPER = {"vits": (0.127, 0.885), "vitb": (0.126, 0.884), "vitl": (0.127, 0.882)}


def net_size(h, w, short=518, mult=14):
    """Size produced by depth_anything Resize(lower_bound, keep_aspect, multiple 14)."""
    scale = short / min(h, w)
    nh, nw = h * scale, w * scale
    up = lambda x: int(np.ceil(x / mult) * mult)  # noqa: E731  lower_bound -> ceil
    return up(nh), up(nw)


def score(pred, gt_disp, mask):
    if mask.sum() < 10:
        return None
    x, y = pred[mask].astype(np.float64), gt_disp[mask].astype(np.float64)
    xc = x - x.mean()
    a = np.dot(xc, y - y.mean()) / np.dot(xc, xc) * xc + y.mean()
    ok = a > 0
    r = np.where(ok, np.maximum(a / y, y / np.where(ok, a, 1)), np.inf)
    return float(np.mean(np.abs(a - y) / y)), float(np.mean(r < 1.25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cache = ROOT / ".cache/table2/eth3d" / f"moge-{args.encoder}"
    res = {}
    with ZipFile(ROOT / "data/moge/ETH3D.zip") as z:
        keys = [k for k in z.read("ETH3D/.index.txt").decode().split() if k]
        keys = keys[:args.limit] if args.limit else keys
        for i, k in enumerate(keys):
            depth = moge_depth(z.read(f"ETH3D/{k}/depth.png"))
            pred_full = np.load(cache / f"{i:05d}.npy")
            h, w = net_size(*depth.shape)
            pred = cv2.resize(pred_full, (w, h), interpolation=cv2.INTER_LINEAR)
            valid = (depth > 0).astype(np.float32)
            d0 = depth.astype(np.float32)
            disp0 = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-6), 0).astype(np.float32)
            nn_mask = cv2.resize(valid, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            lin_mask = cv2.resize(valid, (w, h), interpolation=cv2.INTER_LINEAR)
            d_nn = cv2.resize(d0, (w, h), interpolation=cv2.INTER_NEAREST)
            d_lin = cv2.resize(d0, (w, h), interpolation=cv2.INTER_LINEAR)
            d_area = cv2.resize(d0, (w, h), interpolation=cv2.INTER_AREA)
            disp_lin = cv2.resize(disp0, (w, h), interpolation=cv2.INTER_LINEAR)
            inv = lambda d: np.where(d > 1e-6, 1.0 / np.maximum(d, 1e-6), 0)  # noqa: E731
            variants = {
                "nearest": (inv(d_nn), nn_mask & (d_nn > 0)),
                "lin-depth/nn-mask": (inv(d_lin), nn_mask & (d_lin > 0)),
                "lin-depth/strict": (inv(d_lin), (lin_mask > 0.999) & (d_lin > 0)),
                "lin-depth/half": (inv(d_lin), (lin_mask >= 0.5) & (d_lin > 0)),
                "area-depth/nn-mask": (inv(d_area), nn_mask & (d_area > 0)),
                "lin-disp/nn-mask": (disp_lin, nn_mask & (disp_lin > 0)),
            }
            for name, (gt, m) in variants.items():
                s = score(pred, gt, m)
                if s:
                    res.setdefault(name, []).append(s)
            if i % 100 == 0:
                print(f"{i}/{len(keys)}", flush=True)
    pa, pd = PAPER[args.encoder]
    out = {}
    for name, v in res.items():
        a = np.array(v)
        out[name] = dict(images=len(a), abs_rel=float(a[:, 0].mean()), delta1=float(a[:, 1].mean()))
        print(f"{name:20s} n={len(a)} abs_rel={a[:, 0].mean():.4f} d1={a[:, 1].mean():.4f}")
    print(f"paper {pa:.3f}/{pd:.3f}")
    tag = f"resize-hypotheses-{args.encoder}" + (f"-first{args.limit}" if args.limit else "")
    (ROOT / "results/table2/eth3d" / f"{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
