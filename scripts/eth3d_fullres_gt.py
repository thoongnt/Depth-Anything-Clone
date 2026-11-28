"""ETH3D Table 2 hypothesis: the paper's encoder-independent error floor (~0.127)
comes from resizing the ORIGINAL 6048x4032 ETH3D depth to the network output size.

GT: official per-scene archives <scene>_dslr_depth.7z (float32 6048x4032, inf = no
data), streamed member by member through Windows bsdtar (never written to disk).
Predictions: cached by scripts/eval_relative.py --cache (MoGe ETH3D bundle RGB, same
image IDs), brought back to the network output size. The pilot
scripts/eth3d_distortion_pilot.py showed that pairing undistorted images with the
(distorted-geometry) official depth changes AbsRel by < 0.001.

GT -> network size (518 x 784):
  nearest      depth and mask nearest
  lin/half     depth bilinear with holes as 0, keep bilinear mask >= 0.5
  lin/strict   depth bilinear, keep bilinear mask == 1
  lin/nnmask   depth bilinear, mask nearest
  area/half    depth INTER_AREA with holes as 0, keep area mask >= 0.5
  areanorm/half  hole-aware area average (sum(d*m)/sum(m)), keep area mask >= 0.5
Scoring: per-image OLS on 1/GT in disparity, metrics in disparity, mean over images.

Usage: python scripts/eth3d_fullres_gt.py   (all scenes whose archive is present)
"""
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eth3d_resize_hypotheses import PAPER, net_size, score  # noqa: E402

TAR = r"C:\Windows\System32\tar.exe"
ARCH = ROOT / "data" / "eth3d_official"
H, W = 4032, 6048
ENCODERS = ("vits", "vitb", "vitl")


def members(archive):
    out = subprocess.run([TAR, "-tvf", str(archive)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split()
        name, size = parts[-1], int(parts[4])
        if size > 0:
            rows.append((name, size))
    return rows


def variants(depth, h, w):
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    d0 = np.where(valid > 0, depth, 0).astype(np.float32)
    size = (w, h)
    nn_m = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    lin_m = cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR)
    area_m = cv2.resize(valid, size, interpolation=cv2.INTER_AREA)
    d_nn = cv2.resize(d0, size, interpolation=cv2.INTER_NEAREST)
    d_lin = cv2.resize(d0, size, interpolation=cv2.INTER_LINEAR)
    d_area = cv2.resize(d0, size, interpolation=cv2.INTER_AREA)
    d_areanorm = np.where(area_m > 0, d_area / np.maximum(area_m, 1e-6), 0)
    inv = lambda d: np.where(d > 1e-6, 1.0 / np.maximum(d, 1e-6), 0)  # noqa: E731
    return {
        "nearest": (inv(d_nn), (nn_m > 0.5) & (d_nn > 0)),
        "lin/half": (inv(d_lin), (lin_m >= 0.5) & (d_lin > 0)),
        "lin/strict": (inv(d_lin), (lin_m > 0.999) & (d_lin > 0)),
        "lin/nnmask": (inv(d_lin), (nn_m > 0.5) & (d_lin > 0)),
        "area/half": (inv(d_area), (area_m >= 0.5) & (d_area > 0)),
        "areanorm/half": (inv(d_areanorm), (area_m >= 0.5) & (d_areanorm > 0)),
    }


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def sweep_variants(depth, h, w):
    """--sweep: interpolation (bilinear / area, holes mixed in) x interpolated-mask threshold."""
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    d0 = np.where(valid > 0, depth, 0).astype(np.float32)
    out = {}
    for interp, flag in (("lin", cv2.INTER_LINEAR), ("area", cv2.INTER_AREA)):
        m = cv2.resize(valid, (w, h), interpolation=flag)
        d = cv2.resize(d0, (w, h), interpolation=flag)
        gt = np.where(d > 1e-6, 1.0 / np.maximum(d, 1e-6), 0)
        for t in THRESHOLDS:
            out[f"{interp}/m>={t}"] = (gt, (m >= t) & (d > 0))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="interpolation x mask-threshold sweep")
    ap.add_argument("--thresholds", default="", help="comma-separated mask thresholds for --sweep")
    ap.add_argument("--tag", default="", help="suffix of the output json")
    args = ap.parse_args()
    global THRESHOLDS
    if args.thresholds:
        THRESHOLDS = tuple(float(t) for t in args.thresholds.split(","))
    make = sweep_variants if args.sweep else variants
    with ZipFile(ROOT / "data/moge/ETH3D.zip") as z:
        keys = [k for k in z.read("ETH3D/.index.txt").decode().split() if k]
    index = {k: i for i, k in enumerate(keys)}
    res = {e: {} for e in ENCODERS}
    done = 0
    for archive in sorted(ARCH.glob("*_dslr_depth.7z")):
        scene = archive.name.replace("_dslr_depth.7z", "")
        rows = members(archive)
        proc = subprocess.Popen([TAR, "-xOf", str(archive)], stdout=subprocess.PIPE, bufsize=1 << 24)
        for name, size in rows:
            data = proc.stdout.read(size)
            if len(data) != size:
                raise IOError(f"short read {name}")
            if size != H * W * 4:
                continue
            key = f"{scene}/{Path(name).stem}"
            if key not in index:
                continue
            depth = np.frombuffer(data, "<f4").reshape(H, W)
            h, w = net_size(1365, 2048)  # network size of the cached MoGe-resolution predictions
            gts = make(depth, h, w)
            for e in ENCODERS:
                pred_full = np.load(ROOT / ".cache/table2/eth3d" / f"moge-{e}" / f"{index[key]:05d}.npy")
                pred = cv2.resize(pred_full, (w, h), interpolation=cv2.INTER_LINEAR)
                for vname, (gt, m) in gts.items():
                    s = score(pred, gt, m)
                    if s:
                        res[e].setdefault(vname, []).append(s)
            done += 1
        proc.stdout.close()
        proc.wait()
        print(f"{scene}: images so far {done}", flush=True)
    out = {}
    for e in ENCODERS:
        pa, pd = PAPER[e]
        out[e] = {}
        for vname, v in res[e].items():
            a = np.array(v)
            out[e][vname] = dict(images=len(a), abs_rel=float(a[:, 0].mean()), delta1=float(a[:, 1].mean()))
            print(f"{e} {vname:14s} n={len(a)} abs_rel={a[:, 0].mean():.4f} d1={a[:, 1].mean():.4f}  (paper {pa}/{pd})")
    name = (f"fullres-gt-sweep{args.tag}.json" if args.sweep else "fullres-gt-hypotheses.json")
    (ROOT / "results/table2/eth3d" / name).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
