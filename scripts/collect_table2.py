"""Collect Table 2 (zero-shot relative depth) results and judge them against the paper.

Reads the summaries already produced by the relative-depth evaluators under
results/table2/{nyuv2,diode,eth3d}/ and writes results/table2/summary.{md,json}.
DDAD is out of scope (like HyperSim in Table 5).

Verdict on the values rounded to 3 decimals, d = max(|dAbsRel|, |d_delta1|):
  KHỚP d == 0 · ĐẠT d <= --tol (0.005) · KHÔNG ĐẠT otherwise · CHƯA CHẠY no result.

Usage: python scripts/collect_table2.py [results/table2] [--tol 0.005]
"""
import argparse
import json
import os
import sys

ENCODERS = ("vits", "vitb", "vitl")
DATASETS = ("kitti", "nyuv2", "sintel", "eth3d", "diode")
NAMES = {"kitti": "KITTI", "nyuv2": "NYUv2", "sintel": "Sintel", "eth3d": "ETH3D", "diode": "DIODE"}
PAPER = {  # Table 2, Depth Anything rows (AbsRel, delta1); DDAD omitted
    "vits": {"kitti": (0.080, 0.936), "nyuv2": (0.053, 0.972), "sintel": (0.464, 0.739),
             "eth3d": (0.127, 0.885), "diode": (0.076, 0.939)},
    "vitb": {"kitti": (0.080, 0.939), "nyuv2": (0.046, 0.979), "sintel": (0.432, 0.756),
             "eth3d": (0.126, 0.884), "diode": (0.069, 0.946)},
    "vitl": {"kitti": (0.076, 0.947), "nyuv2": (0.043, 0.981), "sintel": (0.458, 0.760),
             "eth3d": (0.127, 0.882), "diode": (0.066, 0.952)},
}


def load(root, rel):
    p = os.path.join(root, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def from_metrics(root, rel):
    d = load(root, rel)
    return (d["metrics"]["abs_rel"], d["metrics"]["delta1"]) if d else None


def from_results(root, rel, configuration, space="disparity", subset="all"):
    d = load(root, rel)
    for r in (d or {}).get("results", []):
        if r["configuration"] == configuration and r["metric_space"] == space and r["subset"] == subset:
            return r["metrics"]["abs_rel"], r["metrics"]["delta1"]
    return None


def from_comparisons(root, rel, variant="no-flip", space="disparity"):
    d = load(root, rel)
    for r in (d or {}).get("comparisons", []):
        if (r["variant"], r["cap"], r["subset"], r["space"], r["weighting"]) == (variant, None, "all", space, "image"):
            return r["abs_rel"], r["delta1"]
    return None


# (dataset, encoder, config, description, protocol_confirmed, getter)
CONFIGS = [
    ("nyuv2", "vitl", "midas", "rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip", True,
     lambda r: from_metrics(r, "nyuv2/vitl-midas-summary.json")),
    ("nyuv2", "vits", "midas", "rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip", True,
     lambda r: from_metrics(r, "nyuv2/vits-midas-summary.json")),
    ("nyuv2", "vitb", "midas", "rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip", True,
     lambda r: from_metrics(r, "nyuv2/vitb-midas-summary.json")),
    ("nyuv2", "vitl", "filled-eigen", "đối chứng: depths đã inpaint + Eigen crop, chấm disparity", False,
     lambda r: from_metrics(r, "nyuv2/vitl-filled-eigen-disparity-diagnostic-summary.json")),
    ("diode", "vitl", "native", "mask cảm biến gốc, toàn ảnh, fit + chấm disparity, không flip", False,
     lambda r: from_metrics(r, "diode/vitl-native-all-summary.json")),
    ("diode", "vits", "native", "mask cảm biến gốc, toàn ảnh, fit + chấm disparity, không flip", False,
     lambda r: from_metrics(r, "diode/followup/vits-native-all-summary.json")),
    ("diode", "vitb", "native", "mask cảm biến gốc, toàn ảnh, fit + chấm disparity, không flip", False,
     lambda r: from_metrics(r, "diode/followup/vitb-native-all-summary.json")),
    ("diode", "vitl", "moge-mask", "giao mask gốc với mask MoGe (lọc biên/điểm xa), disparity, không flip", False,
     lambda r: from_results(r, "diode/followup/mask-summary.json", "native-pred-native-depth-moge-mask")),
    ("diode", "vits", "moge-mask", "giao mask gốc với mask MoGe, disparity, không flip", False,
     lambda r: from_results(r, "diode/followup/mask-vits-summary.json", "native-pred-native-depth-moge-mask")),
    ("diode", "vitb", "moge-mask", "giao mask gốc với mask MoGe, disparity, không flip", False,
     lambda r: from_results(r, "diode/followup/mask-vitb-summary.json", "native-pred-native-depth-moge-mask")),
    ("diode", "vitl", "moge-mask-flip", "mask MoGe + flip-TTA", False,
     lambda r: from_results(r, "diode/followup/mask-flip-summary.json", "native-pred-native-depth-moge-mask")),
    ("diode", "vitl", "moge-mask-392", "mask MoGe, cạnh ngắn 392", False,
     lambda r: from_results(r, "diode/followup/resolution-summary.json", "moge-mask-short-side-392")),
    ("eth3d", "vitl", "official-ols", "depth ETH3D chính thức stride 3, 518 px, OLS, disparity, không flip", False,
     lambda r: from_comparisons(r, "eth3d/official-stride3-vitl-518-summary.json")),
    ("eth3d", "vitl", "official-ols-depth", "như trên, chấm depth (clamp)", False,
     lambda r: from_comparisons(r, "eth3d/official-stride3-vitl-518-summary.json", space="clamped-depth")),
    ("eth3d", "vitl", "moge-ols", "gói ETH3D của MoGe, 518 px, OLS, disparity, không flip", False,
     lambda r: from_comparisons(r, "eth3d/vitl-518-flip-summary.json")),
    ("eth3d", "vitl", "official-median-mad", "depth chính thức, fit median–MAD", False,
     lambda r: from_comparisons(r, "eth3d/official-stride3-vitl-518-median-mad-summary.json")),
]


SOURCE_DESC = {
    "annotated": "Eigen test 652, depth annotated",
    "lidar": "Eigen test 697, LiDAR thô chiếu lên ảnh",
    "moge": "gói đánh giá MoGe",
    "final": "pass final, depth .dpt chính thức",
    "clean": "pass clean, depth .dpt chính thức",
}


def eth3d_resize_rows(root):
    """scripts/eth3d_resize_hypotheses.py: GT resized to the network output size."""
    out = []
    desc = {"nearest": "GT nearest về kích thước mạng", "lin-depth/strict": "GT bilinear, giữ mask nội suy = 1",
            "lin-depth/half": "GT bilinear, giữ mask nội suy ≥ 0.5"}
    for enc in ENCODERS:
        d = load(root, f"eth3d/resize-hypotheses-{enc}.json")
        for name, label in desc.items():
            if d and name in d:
                out.append(("eth3d", enc, f"moge-netres|{name}", f"gói MoGe; {label}", False,
                            (d[name]["abs_rel"], d[name]["delta1"])))
    # scripts/eth3d_fullres_gt.py: original 6048x4032 official depth resized to the network size
    full = load(root, "eth3d/fullres-gt-hypotheses.json") or {}
    fdesc = {"nearest": "nearest", "lin/half": "bilinear, mask ≥ 0.5", "area/half": "area, mask ≥ 0.5",
             "areanorm/half": "area có trọng số mask, mask ≥ 0.5"}
    for enc in ENCODERS:
        for name, label in fdesc.items():
            v = full.get(enc, {}).get(name)
            if v:
                skipped = f" (chỉ {v['images']}/454 ảnh còn pixel hợp lệ)" if v["images"] < 454 else ""
                out.append(("eth3d", enc, f"official-fullres|{name}",
                            f"depth chính thức 6048×4032 thu nhỏ về kích thước mạng; {label}{skipped}", False,
                            (v["abs_rel"], v["delta1"])))
    # --sweep: bilinear (holes mixed in) x interpolated-mask threshold, TUNED to approach the paper
    for tag in ("", "-fine", "-fine2"):
        sw = load(root, f"eth3d/fullres-gt-sweep{tag}.json") or {}
        for enc in ENCODERS:
            for name, v in sw.get(enc, {}).items():
                if name.startswith("lin/") and v["images"] == 454:
                    out.append(("eth3d", enc, f"official-fullres|{name}",
                                f"depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask {name[6:]} "
                                f"(ngưỡng được CHỈNH để khớp paper)", False, (v["abs_rel"], v["delta1"])))
    # scripts/sintel_cap_sweep.py: finer depth caps, TUNED to approach the paper
    cs = (load(root, "sintel/cap-sweep.json") or {}).get("results", {})
    for enc in ENCODERS:
        for name, v in cs.get(enc, {}).items():
            skipped = f" (chỉ {v['images']}/1064 ảnh)" if v["images"] < 1064 else ""
            out.append(("sintel", enc, f"final-capsweep|{name}",
                        f"pass final; {name} (cap được CHỈNH để khớp paper){skipped}", False,
                        (v["abs_rel"], v["delta1"])))
    return out


def grid_rows(root):
    """Rows from scripts/eval_relative.py outputs: <dataset>/<source>-<encoder>[-flip]-grid.json."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*-grid.json"))):
        if "-first" in os.path.basename(p):
            continue  # --limit debug runs
        d = load(root, os.path.relpath(p, root))
        flip = "-flip" if d.get("flip") else ""
        for key, g in d["grid"].items():
            # A variant can leave some images without valid pixels (e.g. Sintel frames
            # entirely beyond the depth cap); it is kept but flagged.
            skipped = f" (bỏ {g['skipped']} ảnh không còn pixel hợp lệ)" if g["skipped"] else ""
            out.append((d["dataset"], d["encoder"], f"{d['source']}{flip}|{key}",
                        f"{SOURCE_DESC.get(d['source'], d['source'])}{', flip-TTA' if flip else ''}; {key}{skipped}",
                        False, (g["abs_rel"], g["delta1"])))
    return out


def judge(enc, ds, abs_rel, a1, tol):
    pa, pd = PAPER[enc][ds]
    da, dd = round(round(abs_rel, 3) - pa, 3), round(round(a1, 3) - pd, 3)
    d = max(abs(da), abs(dd))
    verdict = "KHỚP" if d == 0 else ("ĐẠT" if d <= tol + 1e-9 else "KHÔNG ĐẠT")
    return dict(paper_abs_rel=pa, paper_a1=pd, d_abs_rel=da, d_a1=dd, dist=round(d, 3), verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/table2")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()

    rows, seen = [], set()
    entries = [(ds, enc, name, desc, confirmed, get(args.root)) for ds, enc, name, desc, confirmed, get in CONFIGS]
    for ds, enc, name, desc, confirmed, v in entries + grid_rows(args.root) + eth3d_resize_rows(args.root):
        if v is None or (ds, enc, name) in seen:
            continue  # default and extended grids share some variants (same predictions)
        seen.add((ds, enc, name))
        rows.append(dict(dataset=ds, encoder=enc, config=name, desc=desc, protocol_confirmed=confirmed,
                         abs_rel=round(v[0], 3), a1=round(v[1], 3), **judge(enc, ds, v[0], v[1], args.tol)))

    # One protocol for all three encoders: configuration whose worst encoder is closest.
    common = {}
    for ds in DATASETS:
        by_cfg = {}
        for r in rows:
            if r["dataset"] == ds:
                by_cfg.setdefault(r["config"], {})[r["encoder"]] = r
        full = {c: e for c, e in by_cfg.items() if len(e) == len(ENCODERS)}
        if full:
            c = min(full, key=lambda c: (max(r["dist"] for r in full[c].values()),
                                         sum(r["dist"] for r in full[c].values())))
            common[ds] = (c, full[c])

    best = {}  # closest configuration per (dataset, encoder)
    for r in rows:
        k = (r["dataset"], r["encoder"])
        if k not in best or r["dist"] < best[k]["dist"]:
            best[k] = r

    out = [f"Ngưỡng: KHỚP = trùng 3 chữ số; ĐẠT = |ΔAbsRel| và |Δδ1| ≤ {args.tol}; còn lại KHÔNG ĐẠT. "
           "DDAD nằm ngoài phạm vi.", "",
           "## Trạng thái (cấu hình gần paper nhất mỗi ô)", "",
           "| Encoder | " + " | ".join(NAMES[d] for d in DATASETS) + " |",
           "|---|" + "---|" * len(DATASETS)]
    for enc in ENCODERS:
        cells = []
        for ds in DATASETS:
            b = best.get((ds, enc))
            pa, pd = PAPER[enc][ds]
            if b is None:
                cells.append(f"CHƯA CHẠY (paper {pa:.3f} / {pd:.3f})")
            else:
                note = "" if b["protocol_confirmed"] else "*"
                cells.append(f"{b['abs_rel']:.3f} / {b['a1']:.3f} — {b['verdict']}{note} ({b['config']})")
        out.append(f"| {enc} | " + " | ".join(cells) + " |")
    out += ["", "\\* protocol chưa được tác giả xác nhận (giả thuyết khớp số).", "",
            "## Protocol chung cho cả 3 encoder", "",
            "Mỗi dataset: cấu hình có độ lệch lớn nhất qua ViT-S/B/L nhỏ nhất (một protocol duy nhất, "
            "khó \"ăn may\" hơn chọn riêng từng ô).", "",
            "| Dataset | Cấu hình | ViT-S | ViT-B | ViT-L | Kết luận chung |",
            "|---|---|---|---|---|---|"]
    order = {"KHỚP": 0, "ĐẠT": 1, "KHÔNG ĐẠT": 2}
    for ds, (c, e) in common.items():
        worst = max((e[k]["verdict"] for k in ENCODERS), key=order.get)
        cells = [f"{e[k]['abs_rel']:.3f} / {e[k]['a1']:.3f} ({e[k]['verdict']})" for k in ENCODERS]
        out.append(f"| {NAMES[ds]} | {c} | " + " | ".join(cells) + f" | {worst} |")
    out += ["", "## Các cấu hình gần paper nhất (tối đa 8 dòng mỗi ô; đủ trong summary.json)", "",
            "| Dataset | Encoder | Cấu hình | Mô tả | AbsRel | δ1 | Paper | ΔAbsRel | Δδ1 | Kết luận |",
            "|---|---|---|---|---:|---:|---|---:|---:|---|"]
    shown = {}
    for r in sorted(rows, key=lambda r: (DATASETS.index(r["dataset"]), ENCODERS.index(r["encoder"]), r["dist"])):
        k = (r["dataset"], r["encoder"])
        shown[k] = shown.get(k, 0) + 1
        if shown[k] > 8:
            continue
        mark = " ★" if best[k] is r else ""
        out.append(f"| {NAMES[r['dataset']]} | {r['encoder']} | {r['config']}{mark} | {r['desc']} | "
                   f"{r['abs_rel']:.3f} | {r['a1']:.3f} | {r['paper_abs_rel']:.3f} / {r['paper_a1']:.3f} | "
                   f"{r['d_abs_rel']:+.3f} | {r['d_a1']:+.3f} | {r['verdict']} |")
    out += ["", "★ = cấu hình gần paper nhất của ô (dataset, encoder)."]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(args.root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(tol=args.tol, paper=PAPER, rows=rows,
                       common={ds: dict(config=c, rows=e) for ds, (c, e) in common.items()}),
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
