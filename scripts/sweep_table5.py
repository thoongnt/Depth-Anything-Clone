"""Table 5 configuration sweep: one inference pass, many evaluation settings.

For every image the model is run exactly like metric_depth/evaluate.py
(`infer`: forward on the image and on its horizontal flip, averaged). The
un-flipped forward pass is kept as well, so "no flip TTA" costs nothing extra.
Each prediction is then scored under several one-factor-at-a-time variations
of the paper protocol (crop, depth range, GT encoding, masks) with the repo's
own `compute_metrics`.

Loader-level settings (DIODE antialias, vKITTI2 split seed, ...) need their own
pass: give them with --set key=value, exactly like evaluate.py overrides.

    python scripts/sweep_table5.py --dataset ibims --run ibims
    python scripts/sweep_table5.py --dataset diode_outdoor --run diode_outdoor

Writes results/table5/sweep/<run>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "metric_depth"))
os.chdir(os.path.join(REPO, "metric_depth"))

from zoedepth.data.data_mono import DepthDataLoader  # noqa: E402
from zoedepth.models.builder import build_model  # noqa: E402
from zoedepth.utils.arg_utils import infer_type  # noqa: E402
from zoedepth.utils.config import get_config  # noqa: E402
from zoedepth.utils.misc import RunningAverageDict, compute_errors, compute_metrics  # noqa: E402

INDOOR = "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
OUTDOOR = "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"
CKPT = {"sunrgbd": INDOOR, "ibims": INDOOR, "vkitti2": OUTDOOR, "diode_outdoor": OUTDOOR,
        "kitti": OUTDOOR}  # kitti = Table 4 (in-domain metric depth)


def to_depth(pred):  # same as get_depth_from_prediction in evaluate.infer
    if isinstance(pred, (list, tuple)):
        return pred[-1]
    if isinstance(pred, dict):
        return pred["metric_depth"] if "metric_depth" in pred else pred["out"]
    return pred


def score(gt, pred, cfg, crop="config", dataset="nyu"):
    """compute_metrics, plus a 'none' crop (compute_metrics has no such branch:
    with garg_crop=eigen_crop=False it raises NameError on eval_mask)."""
    if crop == "config":
        return compute_metrics(gt, pred, config=cfg, dataset=dataset)
    if crop == "eigen":
        c = Cfg(cfg, garg_crop=False, eigen_crop=True)
        return compute_metrics(gt, pred, config=c, dataset=dataset)
    if crop == "garg":
        c = Cfg(cfg, garg_crop=True, eigen_crop=False)
        return compute_metrics(gt, pred, config=c, dataset=dataset)
    assert crop == "none"
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=True)
    p = pred.squeeze().cpu().numpy()
    p = np.clip(p, cfg["min_depth_eval"], cfg["max_depth_eval"])
    g = gt.squeeze().cpu().numpy()
    m = np.logical_and(g > cfg["min_depth_eval"], g < cfg["max_depth_eval"])
    return compute_errors(g[m], p[m])


class Cfg(dict):  # compute_metrics reads config.<attr>
    __getattr__ = dict.__getitem__


def read_png(path, dtype=None, tries=5):
    """Image.open + np.asarray, retried: a long run once hit a transient
    'broken data stream' OSError on a PNG that decodes fine on re-read."""
    for k in range(tries):
        try:
            with Image.open(path) as im:
                return np.asarray(im, dtype=dtype)
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.0)


def variants(dataset, ds):
    """One-factor-at-a-time changes around the paper protocol ('base')."""
    gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))[None, None].cuda()  # noqa: E731
    V = [dict(name="base", desc="protocol gốc của evaluate.py"),
         dict(name="noflip", desc="không flip-TTA (chỉ 1 forward)", pred="noflip")]

    if dataset == "sunrgbd":
        def sun_gt(root, scale, cutoff=8.0):
            def f(i, _):
                d = read_png(os.path.join(REPO, root, "depth", os.path.basename(ds.depth_files[i])),
                             dtype="uint16") / scale
                if cutoff:
                    d[d > cutoff] = -1
                return gpu(d)
            return f
        V += [dict(name="no_eigen_crop", desc="bỏ eigen crop", crop="none"),
              dict(name="no_8m_cut", desc="không cắt GT > 8 m, max_depth_eval = 10",
                   gt=sun_gt("data/sunrgbd/test_ahanda", 10000, None), cfg=dict(max_depth_eval=10)),
              dict(name="gt_raw", desc="GT: depth/ chính thức (chưa lấp lỗ) /10000",
                   gt=sun_gt("data/sunrgbd/test_raw", 10000)),
              dict(name="gt_mm_bfx", desc="GT đúng thang: depth_bfx giải mã mm /1000",
                   gt=sun_gt("data/sunrgbd/test_mm_bfx", 1000)),
              dict(name="gt_mm", desc="GT đúng thang: depth/ giải mã mm /1000",
                   gt=sun_gt("data/sunrgbd/test_mm", 1000))]

    elif dataset == "ibims":
        def ibims_gt(use_invalid, use_transp):
            def f(i, _):
                _, dp, vp, tp = ds.samples[i]
                d = read_png(dp, dtype=np.uint16).astype("float") * 50.0 / 65535
                keep = np.ones(d.shape, bool)
                if use_invalid:
                    keep &= read_png(vp) > 0
                if use_transp:
                    keep &= read_png(tp) > 0
                return gpu(np.where(keep, d, -1))
            return f
        V += [dict(name="no_eigen_crop", desc="bỏ eigen crop", crop="none"),
              dict(name="max_eval_50", desc="max_depth_eval = 50 (toàn dải iBims)", cfg=dict(max_depth_eval=50)),
              dict(name="no_transp_mask", desc="giữ pixel trong suốt (chỉ mask invalid)", gt=ibims_gt(True, False)),
              dict(name="no_masks", desc="không dùng mask invalid/trong suốt", gt=ibims_gt(False, False))]

    elif dataset in ("vkitti2", "kitti"):
        V += [dict(name="eigen_crop", desc="eigen crop KITTI thay cho garg crop", crop="eigen", dataset="kitti"),
              dict(name="no_crop", desc="bỏ garg crop", crop="none")]
        if dataset == "kitti":  # Table 4 (in-domain): depth range of the evaluation
            V += [dict(name="max_eval_50", desc="max_depth_eval = 50", cfg=dict(max_depth_eval=50)),
                  dict(name="min_eval_1", desc="min_depth_eval = 1 m", cfg=dict(min_depth_eval=1.0))]

    elif dataset == "diode_outdoor":
        def diode_masked(i, gt):
            valid = torch.from_numpy(np.load(ds.depth_mask_files[i]).squeeze().astype(bool)).cuda()
            return torch.where(valid[None, None], gt, torch.full_like(gt, -1))
        V += [dict(name="no_crop", desc="bỏ garg crop", crop="none"),
              dict(name="eigen_crop", desc="eigen crop KITTI thay cho garg crop", crop="eigen", dataset="kitti"),
              dict(name="valid_mask", desc="áp depth_mask.npy (loader gốc bỏ qua)", gt=diode_masked),
              dict(name="max_eval_300", desc="max_depth_eval = 300 (giữ GT xa > 80 m)", cfg=dict(max_depth_eval=300))]
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CKPT))
    ap.add_argument("--run", required=True, help="name of the output json")
    ap.add_argument("--set", action="append", default=[], help="config override key=value (loader-level)")
    ap.add_argument("--only", default="", help="comma-separated variant names (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    args = ap.parse_args()

    overrides = {k: infer_type(v) for k, v in (s.split("=", 1) for s in args.set)}
    config = get_config("zoedepth", "eval", args.dataset, pretrained_resource=CKPT[args.dataset], **overrides)
    loader = DepthDataLoader(config, "online_eval").data
    ds = loader.dataset
    model = build_model(config).cuda().eval()

    V = variants(args.dataset, ds)
    if args.only:
        V = [v for v in V if v["name"] in args.only.split(",")]
    base_cfg = Cfg({k: config[k] for k in ("garg_crop", "eigen_crop", "min_depth_eval", "max_depth_eval")})
    metrics = {v["name"]: RunningAverageDict() for v in V}
    n = 0
    for i, sample in tqdm(enumerate(loader), total=len(loader)):
        if args.limit and i >= args.limit:
            break
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue
        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
        kw = dict(dataset=sample["dataset"][0], focal=focal)
        with torch.no_grad():  # == evaluate.infer, keeping the un-flipped pass
            p1 = to_depth(model(image, **kw))
            p2 = torch.flip(to_depth(model(torch.flip(image, [3]), **kw)), [3])
        preds = {"flip": 0.5 * (p1 + p2), "noflip": p1}
        for v in V:
            gt = v["gt"](i, depth) if "gt" in v else depth
            cfg = Cfg(base_cfg, **v.get("cfg", {}))
            metrics[v["name"]].update(score(gt, preds[v.get("pred", "flip")].clone(), cfg,
                                            crop=v.get("crop", "config"), dataset=v.get("dataset", "nyu")))
        n += 1

    out = dict(dataset=args.dataset, run=args.run, overrides=overrides, n=n, variants={})
    for v in V:
        vals = {k: round(float(x), 3) for k, x in metrics[v["name"]].get_value().items()}
        out["variants"][v["name"]] = dict(desc=v["desc"], **vals)
        print(f"{v['name']:16s} abs_rel={vals['abs_rel']:.3f}  a1={vals['a1']:.3f}  rmse={vals['rmse']:.3f}")
    table = "table4" if args.dataset == "kitti" else "table5"
    path = os.path.join(REPO, "results", table, "sweep", args.run + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
