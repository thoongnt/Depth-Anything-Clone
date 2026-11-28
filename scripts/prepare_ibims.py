"""Convert iBims-1 (ibims1_core_mat/*.mat + rgb/*.png) into the
`ibims1_core_raw` layout read by metric_depth/zoedepth/data/ibims.py:

    <out>/imagelist.txt
    <out>/rgb/<name>.png
    <out>/depth/<name>.png          uint16, metres = value * 50 / 65535
    <out>/mask_invalid/<name>.png   uint8, 1 = valid
    <out>/mask_transp/<name>.png    uint8, 1 = not transparent

Usage:
    python scripts/prepare_ibims.py --src data/ibims/iBims-1 --out data/ibims/ibims1_core_raw
"""
import argparse
import glob
import os
import shutil

import numpy as np
import scipy.io as sio
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for sub in ("rgb", "depth", "mask_invalid", "mask_transp"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    names = []
    max_err = 0.0
    for mat_path in sorted(glob.glob(os.path.join(args.src, "ibims1_core_mat", "*.mat"))):
        name = os.path.splitext(os.path.basename(mat_path))[0]
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]

        depth = np.asarray(data.depth, dtype=np.float64)
        depth_png = np.clip(np.round(depth * 65535.0 / 50.0), 0, 65535).astype(np.uint16)
        max_err = max(max_err, float(np.abs(depth_png * 50.0 / 65535 - depth).max()))
        Image.fromarray(depth_png).save(os.path.join(args.out, "depth", name + ".png"))

        for key in ("mask_invalid", "mask_transp"):
            mask = np.asarray(getattr(data, key)).astype(np.uint8)
            assert set(np.unique(mask)) <= {0, 1}, (name, key, np.unique(mask))
            Image.fromarray(mask).save(os.path.join(args.out, key, name + ".png"))

        rgb_src = os.path.join(args.src, "rgb", name + ".png")
        if os.path.exists(rgb_src):
            shutil.copyfile(rgb_src, os.path.join(args.out, "rgb", name + ".png"))
        else:
            Image.fromarray(np.asarray(data.rgb, dtype=np.uint8)).save(
                os.path.join(args.out, "rgb", name + ".png"))
        names.append(name)

    with open(os.path.join(args.out, "imagelist.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"{len(names)} images written, max depth quantisation error {max_err * 1000:.3f} mm")


if __name__ == "__main__":
    main()
