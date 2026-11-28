"""Evaluate the indoor metric model on SUN RGB-D against several ground-truth
variants with a single inference pass.

The prediction does not depend on the GT, so each RGB image is run through
the model once (same `infer` with flip-averaging as metric_depth/evaluate.py)
and then scored with the repo's `compute_metrics` against every variant.
This is numerically identical to running
    evaluate.py -m zoedepth -d sunrgbd sunrgbd_root=<root> sunrgbd_depth_scale=<scale>
once per variant, as long as the variants share the same RGB images.

Variants are given as name=root:scale, e.g.
    python scripts/eval_sunrgbd_variants.py \
        --variant raw=data/sunrgbd/test_raw:10000 \
        --variant mm=data/sunrgbd/test_mm:1000 \
        --out results/table5/sunrgbd_variants.json
All variant roots must use the rgb/img-XXXXXX.jpg + depth/N.png layout; RGB is
read from the first variant (the script checks the others have the same files).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "metric_depth"))
os.chdir(os.path.join(REPO, "metric_depth"))

from evaluate import infer  # noqa: E402
from zoedepth.models.builder import build_model  # noqa: E402
from zoedepth.utils.config import get_config  # noqa: E402
from zoedepth.utils.misc import RunningAverageDict, compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, help="name=root:scale")
    ap.add_argument("--ckpt", default="local::./checkpoints/depth_anything_metric_depth_indoor.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (debug)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    variants = []
    for v in args.variant:
        name, rest = v.split("=", 1)
        root, scale = rest.rsplit(":", 1)
        variants.append((name, os.path.join(REPO, root), float(scale)))

    rgb_root = os.path.join(variants[0][1], "rgb")
    names = sorted(os.listdir(rgb_root))
    if args.limit:
        names = names[:args.limit]
    for _, root, _ in variants[1:]:
        assert sorted(os.listdir(os.path.join(root, "rgb")))[:len(names)] == names, root

    config = get_config("zoedepth", "eval", "sunrgbd", pretrained_resource=args.ckpt)
    model = build_model(config).to(args.device).eval()

    metrics = {name: RunningAverageDict() for name, _, _ in variants}
    for fname in tqdm(names):
        idx = int(fname.replace(".jpg", "").replace("img-", ""))
        image = np.asarray(Image.open(os.path.join(rgb_root, fname)), dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = infer(model, image, dataset="sunrgbd",
                         focal=torch.Tensor([715.0873]).to(args.device))
        for name, root, scale in variants:
            # identical to zoedepth/data/sun_rgbd_loader.py
            depth = np.asarray(Image.open(os.path.join(root, "depth", f"{idx}.png")),
                               dtype="uint16") / scale
            depth[depth > 8] = -1
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(args.device)
            metrics[name].update(compute_metrics(depth, pred, config=config))

    out = {}
    for name, root, scale in variants:
        vals = {k: round(float(v), 3) for k, v in metrics[name].get_value().items()}
        out[name] = dict(root=os.path.relpath(root, REPO), depth_scale=scale, n=len(names), **vals)
        print(f"{name:12s} abs_rel={vals['abs_rel']:.3f}  a1={vals['a1']:.3f}  rmse={vals['rmse']:.3f}")
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
