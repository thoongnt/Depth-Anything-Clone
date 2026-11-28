"""Zero-shot relative depth (Table 2) for KITTI, Sintel and ETH3D, one inference
pass per image, scored under a grid of protocol variants.

Fixed part (same as evaluate_nyuv2.py, the NYUv2 evaluator that matches the paper):
  RGB float32 /255 -> short side 518, multiple of 14, cubic, ImageNet norm
  -> disparity prediction -> bilinear (align_corners=False) to the GT grid
  -> per-image float64 least squares of scale+shift onto 1/GT in disparity space
  -> AbsRel and delta1 in disparity space, unweighted mean over images.
Grid part (unknown details of the paper protocol, varied one at a time):
  depth cap, crop (KITTI), clamp of the aligned disparity at 1/cap, and a
  depth-space diagnostic. Every variant is reported; none is dropped.

Sources
  kitti   annotated : Eigen test 652 imgs, raw RGB PNG + data_depth_annotated (uint16/256)
          lidar     : Eigen test 697 imgs, raw RGB PNG + velodyne projected (monodepth2 style)
          moge      : MoGe evaluation bundle KITTI.zip (652, RGB jpg)
  sintel  final / clean : 1064 training frames, official RGB zip + depth .dpt zip
          moge      : MoGe bundle Sintel.zip
  eth3d   moge      : MoGe bundle ETH3D.zip (454)

Usage
  python scripts/eval_relative.py --dataset kitti --source annotated --encoder vitl
Writes results/table2/<dataset>/<source>-<encoder>[-flip]-grid.json (+ per-image npz).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from depth_anything.dpt import load_depth_anything  # noqa: E402
from evaluate_nyuv2 import make_transform  # noqa: E402

DATA = ROOT / "data"
PAPER = {  # Table 2 (AbsRel, delta1)
    "kitti": {"vits": (0.080, 0.936), "vitb": (0.080, 0.939), "vitl": (0.076, 0.947)},
    "sintel": {"vits": (0.464, 0.739), "vitb": (0.432, 0.756), "vitl": (0.458, 0.760)},
    "eth3d": {"vits": (0.127, 0.885), "vitb": (0.126, 0.884), "vitl": (0.127, 0.882)},
    "diode": {"vits": (0.076, 0.939), "vitb": (0.069, 0.946), "vitl": (0.066, 0.952)},
}
# MiDaS v3.1 DPT-BEiT-L-512 row of Table 2, used to calibrate the evaluation protocol
for _ds, _v in {"kitti": (0.127, 0.850), "sintel": (0.587, 0.699), "eth3d": (0.139, 0.867),
                "diode": (0.075, 0.942)}.items():
    PAPER[_ds]["midas"] = _v
MIDAS_REPO = ROOT / ".cache" / "midas_repo" / "MiDaS-master"   # github.com/isl-org/MiDaS (master)
MIDAS_CKPT = ROOT / "checkpoints" / "midas" / "dpt_beit_large_512.pt"


def load_midas(device):
    """MiDaS v3.1 BEiT-L-512 with its own preprocessing (512, keep aspect, mean/std 0.5)."""
    sys.path.insert(0, str(MIDAS_REPO))
    from midas.model_loader import load_model
    model, transform, _, _ = load_model(device, str(MIDAS_CKPT), "dpt_beit_large_512", optimize=False)
    return model, transform
EIGEN_LIST = ROOT / "metric_depth" / "train_test_inputs" / "kitti_eigen_test_files_with_gt.txt"


# ----------------------------------------------------------------------------- data
def read_png16(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def read_rgb(data):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def moge_depth(data):
    """MoGe log-quantised uint16 depth; 0 and 65535 are invalid (thuc_nghiem_1_eth3d)."""
    with Image.open(BytesIO(data)) as image:
        codes = np.array(image)
        near, far = float(image.info["near"]), float(image.info["far"])
        unit = float(image.info.get("unit", 1))
    valid = (codes != 0) & (codes != 65535)
    depth = np.zeros(codes.shape, np.float64)
    power = (codes[valid].astype(np.float64) - 1) / 65533
    depth[valid] = near ** (1 - power) * far ** power * unit
    return depth


def read_dpt(data):
    tag, width, height = struct.unpack("<fii", data[:12])
    if abs(tag - 202021.25) > 1e-3:
        raise ValueError("not a Sintel .dpt file")
    return np.frombuffer(data[12:], np.float32, count=width * height).reshape(height, width).astype(np.float64)


def kitti_calib(path):
    out = {}
    for line in open(path):
        key, _, value = line.partition(":")
        try:
            out[key] = np.array([float(x) for x in value.split()])
        except ValueError:
            pass
    return out


def kitti_lidar_depth(date_dir, velo_file):
    """Project velodyne points to image_02 (monodepth2 generate_depth_map, cam=2)."""
    c2c = kitti_calib(date_dir / "calib_cam_to_cam.txt")
    v2c = kitti_calib(date_dir / "calib_velo_to_cam.txt")
    velo2cam = np.vstack([np.hstack([v2c["R"].reshape(3, 3), v2c["T"][:, None]]), [0, 0, 0, 1.0]])
    h, w = c2c["S_rect_02"][::-1].astype(int)
    rect = np.eye(4)
    rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    proj = c2c["P_rect_02"].reshape(3, 4) @ rect @ velo2cam
    velo = np.fromfile(velo_file, np.float32).reshape(-1, 4).astype(np.float64)
    velo[:, 3] = 1.0
    velo = velo[velo[:, 0] >= 0]
    pts = (proj @ velo.T).T
    pts[:, :2] /= pts[:, 2:3]
    u, v = np.round(pts[:, 0]).astype(int) - 1, np.round(pts[:, 1]).astype(int) - 1
    ok = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    depth = np.full((h, w), np.inf)
    np.minimum.at(depth, (v[ok], u[ok]), pts[ok, 2])  # nearest point wins on duplicates
    depth[~np.isfinite(depth)] = 0
    depth[depth < 0] = 0
    return depth


def samples(dataset, source, limit):
    """Yield (key, rgb float32 HxWx3, depth float64 HxW with 0 = invalid)."""
    if dataset == "kitti" and source in ("annotated", "lidar"):
        rows = [l.split() for l in open(EIGEN_LIST) if l.strip()]
        if source == "annotated":
            rows = [r for r in rows if r[1] != "None"]
        rows = rows[:limit] if limit else rows
        for img, gt, _ in rows:
            rgb = read_rgb((DATA / "kitti/raw" / img).read_bytes())
            if source == "annotated":
                path = next(p for p in (DATA / "kitti/gt/train" / gt, DATA / "kitti/gt/val" / gt) if p.exists())
                depth = read_png16(path.read_bytes()).astype(np.float64) / 256.0
            else:
                date, drive, _, _, frame = img.split("/")
                depth = kitti_lidar_depth(DATA / "kitti/raw" / date,
                                          DATA / "kitti/raw" / date / drive / "velodyne_points/data" / frame.replace(".png", ".bin"))
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"shape mismatch {img}: {depth.shape} vs {rgb.shape}")
            yield img, rgb, depth
    elif dataset == "diode" and source in ("native", "moge-mask"):
        root = DATA / "diode"
        keys = sorted(p.relative_to(root).with_suffix("").as_posix() for p in (root / "val").glob("*/*/*/*.png"))
        keys = keys[:limit] if limit else keys
        moge = ZipFile(DATA / "moge" / "DIODE.zip") if source == "moge-mask" else None
        for k in keys:
            rgb = read_rgb((root / f"{k}.png").read_bytes())
            depth = np.load(root / f"{k}_depth.npy").squeeze().astype(np.float64)
            valid = np.load(root / f"{k}_depth_mask.npy").squeeze() > 0
            if moge is not None:  # intersect with the MoGe-filtered mask (teammate's best config)
                valid &= moge_depth(moge.read(f"DIODE/{k}/depth.png")) > 0
            depth[~valid] = 0
            yield k, rgb, depth
    elif source == "moge":
        name = {"kitti": "KITTI", "sintel": "Sintel", "eth3d": "ETH3D", "diode": "DIODE"}[dataset]
        with ZipFile(DATA / "moge" / f"{name}.zip") as z:
            keys = [k for k in z.read(f"{name}/.index.txt").decode().split() if k]
            keys = keys[:limit] if limit else keys
            for k in keys:
                rgb = read_rgb(z.read(f"{name}/{k}/image.jpg"))
                depth = moge_depth(z.read(f"{name}/{k}/depth.png"))
                if depth.shape != rgb.shape[:2]:
                    raise ValueError(f"shape mismatch {k}")
                yield k, rgb, depth
    elif dataset == "sintel" and source in ("final", "clean"):
        with ZipFile(DATA / "Sintel/MPI-Sintel-training_images.zip") as zi, \
                ZipFile(DATA / "sintel_depth/MPI-Sintel-depth-training-20150305.zip") as zd:
            depth_names = sorted(n for n in zd.namelist() if n.endswith(".dpt") and "/depth/" in n)
            depth_names = depth_names[:limit] if limit else depth_names
            image_names = set(zi.namelist())
            for dn in depth_names:
                scene, frame = dn.split("/")[-2], dn.split("/")[-1].replace(".dpt", ".png")
                iname = f"training/{source}/{scene}/{frame}"
                if iname not in image_names:
                    raise FileNotFoundError(iname)
                rgb = read_rgb(zi.read(iname))
                depth = read_dpt(zd.read(dn))
                yield f"{scene}/{frame[:-4]}", rgb, depth
    else:
        raise ValueError(f"unknown dataset/source {dataset}/{source}")


# ----------------------------------------------------------------------------- scoring
def crop_mask(shape, crop):
    h, w = shape
    m = np.zeros(shape, bool)
    if crop == "none":
        m[:] = True
    elif crop == "garg":
        m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    elif crop == "eigen":
        m[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)] = True
    return m


def variants(dataset, grid="default"):
    if grid == "extended":
        # Hypotheses for the encoder-independent error floor of Sintel / ETH3D:
        # far pixels clipped to the cap instead of dropped (mode=gtclip), larger caps,
        # and evaluation on a downsampled grid where the GT is resized with
        # nearest vs bilinear interpolation (bilinear mixes holes / far values).
        caps = {"sintel": (72, 80, 100, 200, 500, 1000), "eth3d": (None, 20, 50, 72),
                "kitti": (80,), "diode": (None, 80)}[dataset]
        out = [dict(cap=c, crop="none", clamp=cl, mode=m)
               for c in caps for cl in ("cap", "none") for m in ("exclude", "gtclip") if c or m == "exclude"]
        for f in (0.5, 0.25):
            for gi in ("nearest", "linear"):
                out.append(dict(cap=caps[0] if dataset == "sintel" else None, crop="none", clamp="cap",
                                mode="exclude", res=f, gtinterp=gi))
        return out
    caps = {"kitti": (80, None), "sintel": (None, 72, 80), "eth3d": (None, 72, 80), "diode": (None, 80)}[dataset]
    crops = ("none", "garg", "eigen") if dataset == "kitti" else ("none",)
    return [dict(cap=cap, crop=crop, clamp=clamp)
            for crop in crops for cap in caps for clamp in ("cap", "none")]


def vkey(v):
    k = f"crop={v['crop']},cap={v['cap']},clamp={v['clamp']}"
    if v.get("mode", "exclude") != "exclude":
        k += f",mode={v['mode']}"
    if v.get("res", 1.0) != 1.0:
        k += f",res={v['res']},gt={v['gtinterp']}"
    return k


def score_one(pred, depth, v):
    if v.get("res", 1.0) != 1.0:  # evaluate on a downsampled grid
        h, w = depth.shape
        size = (max(1, int(round(w * v["res"]))), max(1, int(round(h * v["res"]))))
        gi = cv2.INTER_NEAREST if v["gtinterp"] == "nearest" else cv2.INTER_LINEAR
        depth = cv2.resize(np.where(np.isfinite(depth), depth, 0), size, interpolation=gi)
        pred = cv2.resize(pred.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    mask = np.isfinite(depth) & (depth > 1e-3) & crop_mask(depth.shape, v["crop"])
    if v.get("mode") == "gtclip" and v["cap"]:
        depth = np.minimum(depth, v["cap"])  # far pixels kept, GT clipped at the cap
    elif v["cap"]:
        mask &= depth < v["cap"]
    if mask.sum() < 10:
        return None
    gt = 1.0 / depth[mask]
    x = pred[mask].astype(np.float64)
    xc = x - x.mean()
    scale = np.dot(xc, gt - gt.mean()) / np.dot(xc, xc)
    aligned = scale * xc + gt.mean()
    if v["clamp"] == "cap":  # MiDaS style: depth cap -> minimum disparity 1/cap
        floor = 1.0 / (v["cap"] or depth[mask].max())
        aligned = np.maximum(aligned, floor)
    ok = aligned > 0
    ratio = np.where(ok, np.maximum(aligned / gt, gt / np.where(ok, aligned, 1)), np.inf)
    d_pred = 1.0 / np.maximum(aligned, 1.0 / (v["cap"] or depth[mask].max()))
    return (float(np.mean(np.abs(aligned - gt) / gt)), float(np.mean(ratio < 1.25)),
            float(np.mean(np.abs(d_pred - depth[mask]) / depth[mask])),
            float(np.mean(np.maximum(d_pred / depth[mask], depth[mask] / d_pred) < 1.25)), int(mask.sum()))


# ----------------------------------------------------------------------------- main
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PAPER))
    ap.add_argument("--source", required=True)
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "midas"])
    ap.add_argument("--flip", action="store_true", help="average with the flipped prediction")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid", default="default", choices=["default", "extended"])
    ap.add_argument("--cache", action="store_true",
                    help="store/reuse float32 predictions in .cache/table2/ (re-score without inference)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    model = None
    transform = make_transform()
    V = variants(args.dataset, args.grid)
    keys, per = [], {vkey(v): [] for v in V}
    cache_dir = ROOT / ".cache" / "table2" / args.dataset / (f"{args.source}-{args.encoder}" + ("-flip" if args.flip else ""))
    if args.cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (key, rgb, depth) in enumerate(tqdm(samples(args.dataset, args.source, args.limit), mininterval=10,
                                               desc=f"{args.dataset}/{args.source}/{args.encoder}")):
        cfile = cache_dir / f"{i:05d}.npy"
        if args.cache and cfile.exists():
            pred = np.load(cfile)
        else:
            if model is None:
                if args.encoder == "midas":
                    model, transform = load_midas(device)
                else:
                    model = load_depth_anything(args.encoder, str(ROOT / "checkpoints"), device)
            tensor = torch.from_numpy(transform({"image": rgb})["image"]).unsqueeze(0).to(device)
            pred = model(tensor)
            if args.flip:
                pred = 0.5 * (pred + model(tensor.flip(-1)).flip(-1))
            pred = F.interpolate(pred.unsqueeze(1), size=depth.shape, mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
            if args.cache:
                np.save(cfile, pred.astype(np.float32))
        if pred.shape != depth.shape:
            raise ValueError(f"cached prediction shape mismatch for {key}")
        keys.append(key)
        for v in V:
            per[vkey(v)].append(score_one(pred, depth, v) or (np.nan,) * 4 + (0,))

    tag = (f"{args.source}-{args.encoder}" + ("-flip" if args.flip else "") + (f"-first{args.limit}" if args.limit else "")
           + ("-ext" if args.grid == "extended" else ""))
    out_dir = ROOT / "results" / "table2" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = {}
    for v in V:
        a = np.array(per[vkey(v)], dtype=np.float64)
        n_ok = int(np.isfinite(a[:, 0]).sum())
        grid[vkey(v)] = dict(**v, images=n_ok, skipped=len(keys) - n_ok,
                             abs_rel=float(np.nanmean(a[:, 0])), delta1=float(np.nanmean(a[:, 1])),
                             depth_abs_rel=float(np.nanmean(a[:, 2])), depth_delta1=float(np.nanmean(a[:, 3])),
                             mean_valid_pixels=float(a[:, 4].mean()))
    pa, pd = PAPER[args.dataset][args.encoder]
    summary = dict(dataset=args.dataset, source=args.source, encoder=args.encoder, flip=args.flip,
                   images=len(keys), paper=dict(abs_rel=pa, delta1=pd), seconds=round(time.time() - t0),
                   checkpoint_sha256=hashlib.sha256((MIDAS_CKPT if args.encoder == "midas" else ROOT / f"checkpoints/depth_anything_{args.encoder}14.pth").read_bytes()).hexdigest(),
                   protocol_fixed="short side 518 /14 cubic, ImageNet norm, bilinear to GT, OLS on 1/GT in disparity, "
                                  "metrics in disparity, mean over images",
                   grid=grid)
    (out_dir / f"{tag}-grid.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / f"{tag}-per-image.npz", keys=np.array(keys),
                        **{k: np.array(v, dtype=np.float64) for k, v in per.items()})
    best = min(grid.values(), key=lambda g: max(abs(round(g["abs_rel"], 3) - pa), abs(round(g["delta1"], 3) - pd)))
    for k, g in grid.items():
        mark = " *" if g is best else ""
        print(f"{k:34s} n={g['images']:4d} abs_rel={g['abs_rel']:.4f} d1={g['delta1']:.4f} "
              f"| depth {g['depth_abs_rel']:.4f}/{g['depth_delta1']:.4f}{mark}")
    print(f"paper {pa:.3f}/{pd:.3f}")


if __name__ == "__main__":
    main()
