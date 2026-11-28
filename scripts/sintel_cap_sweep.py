"""Sintel Table 2: finer sweep of the far-pixel handling, from cached predictions.

Uses the float32 predictions cached by scripts/eval_relative.py --cache
(.cache/table2/sintel/final-<encoder>/) and the exact scoring function of that
script (score_one). Varies: depth cap, far pixels dropped (exclude) or GT clipped to
the cap (gtclip), and clamping the aligned disparity at 1/cap. This is a search for
the configuration closest to the paper; the result is a hypothesis, not a confirmed
protocol.

Usage: python scripts/sintel_cap_sweep.py
Writes results/table2/sintel/cap-sweep.json
"""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_relative import DATA, PAPER, read_dpt, score_one  # noqa: E402

CAPS = (60, 64, 68, 72, 76, 80, 90, 100, 120, 150, 175, 200, 250, 300)
ENCODERS = ("vits", "vitb", "vitl")


def main():
    variants = [dict(cap=c, crop="none", clamp=cl, mode=m) for c in CAPS
                for m in ("exclude", "gtclip") for cl in ("cap", "none")]
    key = lambda v: f"cap={v['cap']},mode={v['mode']},clamp={v['clamp']}"  # noqa: E731
    acc = {e: {key(v): [] for v in variants} for e in ENCODERS}
    with ZipFile(DATA / "sintel_depth/MPI-Sintel-depth-training-20150305.zip") as zd:
        names = sorted(n for n in zd.namelist() if n.endswith(".dpt") and "/depth/" in n)
        for i, n in enumerate(names):
            depth = read_dpt(zd.read(n))
            for e in ENCODERS:
                pred = np.load(ROOT / ".cache/table2/sintel" / f"final-{e}" / f"{i:05d}.npy")
                for v in variants:
                    s = score_one(pred, depth, v)
                    if s:
                        acc[e][key(v)].append(s[:2])
            if i % 100 == 0:
                print(f"{i}/{len(names)}", flush=True)
    out = {}
    for e in ENCODERS:
        pa, pd = PAPER["sintel"][e]
        out[e] = {}
        for k, v in acc[e].items():
            a = np.array(v)
            out[e][k] = dict(images=len(a), abs_rel=float(a[:, 0].mean()), delta1=float(a[:, 1].mean()),
                             dist=float(max(abs(round(a[:, 0].mean(), 3) - pa), abs(round(a[:, 1].mean(), 3) - pd))))
    # one configuration for all three encoders: minimise the worst encoder distance
    worst = {k: max(out[e][k]["dist"] for e in ENCODERS) for k in out["vits"]}
    best = sorted(worst, key=worst.get)[:8]
    for k in best:
        cells = " | ".join(f"{out[e][k]['abs_rel']:.3f}/{out[e][k]['delta1']:.3f} (n={out[e][k]['images']})"
                           for e in ENCODERS)
        print(f"{k:34s} worst={worst[k]:.3f} | {cells}")
    print("paper", " | ".join(f"{PAPER['sintel'][e][0]:.3f}/{PAPER['sintel'][e][1]:.3f}" for e in ENCODERS))
    (ROOT / "results/table2/sintel/cap-sweep.json").write_text(
        json.dumps(dict(results=out, common_ranking=[(k, worst[k]) for k in sorted(worst, key=worst.get)]), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
