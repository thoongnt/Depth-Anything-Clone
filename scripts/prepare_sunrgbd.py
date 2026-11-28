"""Build the SUN RGB-D zero-shot test set in the layout read by
metric_depth/zoedepth/data/sun_rgbd_loader.py:

    <out>/rgb/img-000001.jpg ... img-005050.jpg
    <out>/depth/1.png        ... 5050.png

The 5,050 test frames come from the official split
(SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat, `alltest`).

Depth Anything's loader divides the depth PNG by 10000, but does not say
which PNG encoding it was fed. Several GT variants are written so they can
be compared against the paper:

    raw      official depth/*.png copied as-is (bit-rotated, ~8 x millimetres)
    raw_bfx  official depth_bfx/*.png (hole-filled) copied as-is
    mm       depth/*.png decoded like the toolbox (readframeSUNRGBD) -> uint16 mm
    mm_bfx   depth_bfx/*.png decoded the same way -> uint16 mm

Evaluate `raw*` with the default sunrgbd_depth_scale=10000 and `mm*` with
sunrgbd_depth_scale=1000.

Usage:
    python scripts/prepare_sunrgbd.py --sun-root data/sunrgbd \
        --split data/sunrgbd/toolbox_dl/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat \
        --out data/sunrgbd/test --variants raw raw_bfx mm mm_bfx
"""
import argparse
import glob
import os
import shutil

import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

PREFIX = "/n/fs/sun3d/data/"


def decode_sunrgbd_depth(png_path):
    """Official SUN RGB-D decoding: bitor(bitshift(D,-3), bitshift(D,16-3)) -> mm."""
    raw = np.asarray(Image.open(png_path)).astype(np.uint32)
    mm = ((raw >> 3) | (raw << 13)) & 0xFFFF
    return mm.astype(np.uint16)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-root", required=True, help="folder that contains SUNRGBD/")
    ap.add_argument("--split", required=True, help="allsplit.mat")
    ap.add_argument("--out", required=True, help="output prefix; variants go to <out>_<variant>")
    ap.add_argument("--variants", nargs="+", default=["raw", "raw_bfx", "mm", "mm_bfx"])
    args = ap.parse_args()

    split = sio.loadmat(args.split)
    seqs = [str(x[0]).rstrip("/") for x in split["alltest"].squeeze()]
    assert len(seqs) == 5050, len(seqs)

    for variant in args.variants:
        for sub in ("rgb", "depth"):
            os.makedirs(os.path.join(f"{args.out}_{variant}", sub), exist_ok=True)

    missing = 0
    for i, seq in enumerate(tqdm(seqs), start=1):
        assert seq.startswith(PREFIX), seq
        seq_dir = os.path.join(args.sun_root, seq[len(PREFIX):])
        rgbs = sorted(glob.glob(os.path.join(seq_dir, "image", "*.jpg")))
        depths = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
        bfxs = sorted(glob.glob(os.path.join(seq_dir, "depth_bfx", "*.png")))
        if not (rgbs and depths and bfxs):
            missing += 1
            print("missing files in", seq_dir)
            continue
        for variant in args.variants:
            root = f"{args.out}_{variant}"
            link_or_copy(rgbs[0], os.path.join(root, "rgb", f"img-{i:06d}.jpg"))
            dst = os.path.join(root, "depth", f"{i}.png")
            if os.path.exists(dst):
                continue
            src = bfxs[0] if variant.endswith("_bfx") else depths[0]
            if variant.startswith("raw"):
                link_or_copy(src, dst)
            else:
                Image.fromarray(decode_sunrgbd_depth(src)).save(dst)
    print(f"done, {len(seqs) - missing} frames, {missing} missing")


if __name__ == "__main__":
    main()
