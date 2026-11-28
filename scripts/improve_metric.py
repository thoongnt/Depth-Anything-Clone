"""Inference-time improvement experiments for Table 3 (NYUv2) and Table 4 (KITTI).

Only the INFERENCE changes; data, GT, crops, depth range and metrics are exactly the
paper protocol that reproduced Tables 3/4 to 6/6:
  NYUv2: evaluate_nyuv2_metric.py (BTS in-memory export, rawDepths, Eigen crop, 1e-3..10 m)
  KITTI: metric_depth DepthDataLoader 'kitti' online_eval (KB crop, garg crop, 1e-3..80 m)
Both scored with zoedepth.utils.misc.compute_metrics, mean over images.

What is varied:
  * network input size. The released checkpoints resize every input to 392x518
    WITHOUT keeping the aspect ratio; for KITTI (352x1216 KB crop) that squeezes the
    image 2.3x horizontally. Sizes are given as HxW (multiples of 14).
  * test-time ensembles: averages of the flip-TTA predictions at several sizes
    (arithmetic mean of depth, or geometric mean = mean of log depth).
Every size runs once (with flip TTA, as in evaluate.py); ensembles reuse those
predictions. Each prediction is resized to the GT grid exactly like evaluate.py
(bilinear, align_corners=True) before averaging and scoring.

Usage:
  python scripts/improve_metric.py --dataset kitti --sizes 392x518,350x1218,392x1358 \
      --ensembles 392x518+392x1358 --limit 120 --tag screen
Writes results/table{3,4}/improve/<dataset>-<tag>.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metric_depth"))
os.chdir(ROOT / "metric_depth")

from zoedepth.models.builder import build_model  # noqa: E402
from zoedepth.utils.config import get_config  # noqa: E402
from zoedepth.utils.misc import compute_metrics  # noqa: E402

CKPT = {"nyu": "local::./checkpoints/depth_anything_metric_depth_indoor.pt",
        "kitti": "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"}
PAPER = {  # Ours rows of Table 3 / Table 4, in compute_metrics key names
    "nyu": dict(a1=0.984, a2=0.998, a3=1.000, abs_rel=0.056, rmse=0.206, log_10=0.024),
    "kitti": dict(a1=0.982, a2=0.998, a3=1.000, abs_rel=0.046, rmse=1.896, rmse_log=0.069),
}
TABLE = {"nyu": "table3", "kitti": "table4"}


class FixedResize(nn.Module):
    """Drop-in for DepthAnythingCore.prep.resizer: bilinear, align_corners=True (as upstream)."""

    def __init__(self, hw):
        super().__init__()
        self.hw = hw

    def forward(self, x):
        return F.interpolate(x, self.hw, mode="bilinear", align_corners=True)

    __call__ = forward


def parse_size(s):
    h, w = (int(v) for v in s.lower().split("x"))
    assert h % 14 == 0 and w % 14 == 0, f"{s}: sizes must be multiples of 14"
    return h, w


@torch.inference_mode()
def predict(model, image, size, dataset, gt_hw):
    """flip-TTA prediction at one input size, resized to the GT grid like evaluate.py."""
    prep = model.core.prep
    original = prep.resizer
    if size is not None:
        prep.resizer = FixedResize(size)
    try:
        p = model(image, dataset=dataset)["metric_depth"]
        p2 = model(torch.flip(image, [3]), dataset=dataset)["metric_depth"]
        p = 0.5 * (p + torch.flip(p2, [3]))
    finally:
        prep.resizer = original
    if p.dim() == 3:
        p = p.unsqueeze(1)
    return F.interpolate(p, gt_hw, mode="bilinear", align_corners=True)


def nyu_samples(limit):
    import h5py
    import scipy.io
    from evaluate_nyuv2_metric import prepare_bts_nyu_depth, prepare_bts_nyu_image
    idx = scipy.io.loadmat(ROOT / "data/nyuv2/splits.mat")["testNdxs"].ravel().astype(int) - 1
    idx = idx[:limit] if limit else idx
    with h5py.File(ROOT / "data/nyuv2/nyu_depth_v2_labeled.mat", "r") as data:
        for i in idx:
            img = prepare_bts_nyu_image(data["images"][i].transpose(2, 1, 0))
            tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
            depth = prepare_bts_nyu_depth(data["rawDepths"][i].T)
            yield tensor, torch.from_numpy(depth.astype(np.float32))[None, None]


def kitti_samples(config, limit):
    from zoedepth.data.data_mono import DepthDataLoader
    n = 0
    for sample in DepthDataLoader(config, "online_eval").data:
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue
        yield sample["image"], sample["depth"].squeeze().unsqueeze(0).unsqueeze(0)
        n += 1
        if limit and n >= limit:
            return


def score(dataset, gt, pred, config):
    if dataset == "nyu":  # identical to evaluate_nyuv2_metric.score_metric_depth
        return compute_metrics(gt.clone(), pred.clone(), dataset="nyu", eigen_crop=True,
                               min_depth_eval=1e-3, max_depth_eval=10.0)
    return compute_metrics(gt.clone(), pred.clone(), config=config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nyu", "kitti"])
    ap.add_argument("--sizes", default="", help="comma-separated HxW; 'default' = 392x518 resizer")
    ap.add_argument("--ensembles", default="", help="comma-separated A+B(+C) of sizes; prefix g: for geometric mean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    config = get_config("zoedepth", "eval", args.dataset, pretrained_resource=CKPT[args.dataset])
    model = build_model(config).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sizes = ["default"] + [s for s in args.sizes.split(",") if s and s != "default"]
    ensembles = [e for e in args.ensembles.split(",") if e]
    names = sizes + ensembles
    acc = {n: [] for n in names}
    samples = nyu_samples(args.limit) if args.dataset == "nyu" else kitti_samples(config, args.limit)
    t0 = time.time()
    for image, gt in tqdm(samples, total=args.limit or (654 if args.dataset == "nyu" else 652), mininterval=10):
        image, gt = image.cuda(), gt.cuda().float()
        preds = {s: predict(model, image, None if s == "default" else parse_size(s), args.dataset, gt.shape[-2:])
                 for s in sizes}
        for s in sizes:
            acc[s].append(score(args.dataset, gt, preds[s], config))
        for e in ensembles:
            geo = e.startswith("g:")
            parts = e[2:].split("+") if geo else e.split("+")
            stack = torch.stack([preds[p] for p in parts])
            pe = torch.exp(torch.log(stack.clamp_min(1e-6)).mean(0)) if geo else stack.mean(0)
            acc[e].append(score(args.dataset, gt, pe, config))
    paper = PAPER[args.dataset]
    out = dict(dataset=args.dataset, images=len(acc[sizes[0]]), seconds=round(time.time() - t0),
               paper=paper, results={})
    print(f"{'config':34s} " + " ".join(f"{k:>8s}" for k in paper))
    print(f"{'paper':34s} " + " ".join(f"{v:8.3f}" for v in paper.values()))
    for n in names:
        m = {k: float(np.mean([r[k] for r in acc[n]])) for k in paper}
        out["results"][n] = m
        print(f"{n:34s} " + " ".join(f"{m[k]:8.4f}" for k in paper))
    path = ROOT / "results" / TABLE[args.dataset] / "improve" / f"{args.dataset}-{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
