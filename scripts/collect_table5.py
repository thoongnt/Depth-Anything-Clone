"""Collect Table 5 results and judge every configuration against the paper.

Sources (all under results/table5/):
  *.log                   metric_depth/evaluate.py runs (dict printed last)
  sunrgbd_variants.json   SUN RGB-D GT variants (scripts/eval_sunrgbd_variants.py)
  sweep/<dataset>.json    one-factor configuration sweep (scripts/sweep_table5.py)

Verdict, with d = max(|dAbsRel|, |d_delta1|) on the 3-decimal values:
  KHỚP       d == 0
  ĐẠT        d <= --tol (default 0.005)
  KHÔNG ĐẠT  otherwise

Usage: python scripts/collect_table5.py [results/table5] [--tol 0.005]
"""
import argparse
import ast
import glob
import json
import os
import re
import statistics
import sys

PAPER = {  # Table 5, Depth Anything row: (AbsRel, delta1)
    "sunrgbd": (0.500, 0.660),
    "ibims": (0.150, 0.714),
    "vkitti2": (0.085, 0.913),
    "diode_outdoor": (0.794, 0.288),
}
NAMES = {"sunrgbd": "SUN RGB-D", "ibims": "iBims-1", "vkitti2": "Virtual KITTI 2",
         "diode_outdoor": "DIODE Outdoor"}
MAIN = {  # the paper-protocol run of each dataset: (source, key)
    "sunrgbd": ("sun", "sunrgbd_paper"),
    "ibims": ("log", "ibims"),
    "vkitti2": ("log", "vkitti2_seed0"),
    "diode_outdoor": ("log", "diode_outdoor"),
}
LOG_CONFIGS = {  # extra evaluate.py runs that belong to the sweep
    "diode_outdoor_antialias": ("diode_outdoor", "Resize antialias=True (mặc định torchvision ≥ 0.17)"),
    **{f"vkitti2_seed{s}": ("vkitti2", f"split ngẫu nhiên seed {s}") for s in range(1, 10)},
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path):
    text = ANSI.sub("", open(path, encoding="utf-8", errors="replace").read())
    found = re.findall(r"\{'a1': [^}]*\}", text)
    return ast.literal_eval(found[-1]) if found else None


def judge(dataset, abs_rel, a1, tol):
    pa, pd = PAPER[dataset]
    da, dd = round(abs_rel - pa, 3), round(a1 - pd, 3)
    d = max(abs(da), abs(dd))
    verdict = "KHỚP" if d == 0 else ("ĐẠT" if d <= tol + 1e-9 else "KHÔNG ĐẠT")
    return dict(paper_abs_rel=pa, paper_a1=pd, d_abs_rel=da, d_a1=dd, dist=round(d, 3), verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/table5")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()
    root, tol = args.root, args.tol

    logs = {os.path.splitext(os.path.basename(p))[0]: parse_log(p)
            for p in sorted(glob.glob(os.path.join(root, "*.log")))}
    logs = {k: v for k, v in logs.items() if v}
    sun_path = os.path.join(root, "sunrgbd_variants.json")
    sun = json.load(open(sun_path)) if os.path.exists(sun_path) else {}
    sweeps = {}
    for p in glob.glob(os.path.join(root, "sweep", "*.json")):
        s = json.load(open(p, encoding="utf-8"))
        if not s.get("overrides"):  # loader-level overrides are separate runs
            sweeps[s["dataset"]] = s

    # ---- main results: the paper protocol of each dataset
    main_rows = []
    for ds, (src, key) in MAIN.items():
        m = logs.get(key) if src == "log" else sun.get(key)
        if m is None and ds in sweeps:
            m = sweeps[ds]["variants"]["base"]
        if m is None:
            continue
        main_rows.append(dict(dataset=ds, run=key, abs_rel=m["abs_rel"], a1=m["a1"], rmse=m["rmse"],
                              **judge(ds, m["abs_rel"], m["a1"], tol)))

    # ---- configuration sweep per dataset
    sweep_rows, checks = [], []
    for ds in PAPER:
        rows = []
        if ds in sweeps:
            for name, v in sweeps[ds]["variants"].items():
                rows.append(dict(config=name, desc=v["desc"], abs_rel=v["abs_rel"], a1=v["a1"], rmse=v["rmse"]))
            main = next((r for r in main_rows if r["dataset"] == ds), None)
            b = sweeps[ds]["variants"]["base"]
            if main:
                same = (b["abs_rel"], b["a1"]) == (main["abs_rel"], main["a1"])
                checks.append(f"{NAMES[ds]}: sweep base {b['abs_rel']:.3f} / {b['a1']:.3f}, "
                              f"evaluate.py {main['abs_rel']:.3f} / {main['a1']:.3f} → {'trùng' if same else 'KHÁC'}")
        elif ds == "sunrgbd" and sun:
            for name, v in sun.items():
                rows.append(dict(config=name, desc=f"GT {v['root']} / {v['depth_scale']:g}",
                                 abs_rel=v["abs_rel"], a1=v["a1"], rmse=v["rmse"]))
        else:  # no sweep yet: the paper-protocol run is the baseline
            main = next((r for r in main_rows if r["dataset"] == ds), None)
            if main:
                rows.append(dict(config="base", desc=f"protocol gốc ({main['run']})", abs_rel=main["abs_rel"],
                                 a1=main["a1"], rmse=main["rmse"]))
        for log, (lds, desc) in LOG_CONFIGS.items():
            if lds == ds and log in logs:
                rows.append(dict(config=log, desc=desc, abs_rel=logs[log]["abs_rel"],
                                 a1=logs[log]["a1"], rmse=logs[log]["rmse"]))
        if ds == "vkitti2":
            seeds = [logs[k] for k in sorted(logs) if re.fullmatch(r"vkitti2_seed\d+", k)]
            if len(seeds) > 1:
                rows.append(dict(config=f"seed_mean_{len(seeds)}", desc=f"trung bình {len(seeds)} split seed",
                                 **{k: round(statistics.mean(s[k] for s in seeds), 3)
                                    for k in ("abs_rel", "a1", "rmse")}))
        for r in rows:
            r.update(dataset=ds, **judge(ds, r["abs_rel"], r["a1"], tol))
        rows.sort(key=lambda r: (r["dist"], r["config"] != "base"))
        if rows:
            rows[0]["closest"] = True
        sweep_rows += rows

    # ---- write
    out = [f"Ngưỡng: KHỚP = trùng 3 chữ số; ĐẠT = |ΔAbsRel| và |Δδ1| ≤ {tol}; còn lại KHÔNG ĐẠT.", "",
           "## Kết quả chính (protocol của paper)", "",
           "| Dataset | Run | AbsRel | δ1 | Paper AbsRel | Paper δ1 | ΔAbsRel | Δδ1 | Kết luận |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in main_rows:
        out.append(f"| {NAMES[r['dataset']]} | {r['run']} | {r['abs_rel']:.3f} | {r['a1']:.3f} | "
                   f"{r['paper_abs_rel']:.3f} | {r['paper_a1']:.3f} | {r['d_abs_rel']:+.3f} | {r['d_a1']:+.3f} | "
                   f"{r['verdict']} |")
    for ds in PAPER:
        rows = [r for r in sweep_rows if r["dataset"] == ds]
        if not rows:
            continue
        pa, pd = PAPER[ds]
        out += ["", f"## Quét cấu hình — {NAMES[ds]} (paper {pa:.3f} / {pd:.3f})", "",
                "| Cấu hình | Mô tả | AbsRel | δ1 | ΔAbsRel | Δδ1 | Kết luận |",
                "|---|---|---:|---:|---:|---:|---|"]
        for r in rows:
            mark = " ★" if r.get("closest") else ""
            out.append(f"| {r['config']}{mark} | {r['desc']} | {r['abs_rel']:.3f} | {r['a1']:.3f} | "
                       f"{r['d_abs_rel']:+.3f} | {r['d_a1']:+.3f} | {r['verdict']} |")
    if checks:
        out += ["", "Kiểm tra sweep `base` so với `evaluate.py`:", ""] + [f"- {c}" for c in checks]
    out += ["", "★ = cấu hình gần paper nhất (khoảng cách nhỏ nhất; hoà thì ưu tiên `base`)."]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(tol=tol, main=main_rows, sweep=sweep_rows, checks=checks), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
