"""Collect Table 4 (KITTI in-domain metric depth) and judge against the paper.

Sources: results/table4/kitti.log (metric_depth/evaluate.py -d kitti, last dict)
         results/table4/sweep/kitti.json (scripts/sweep_table5.py --dataset kitti)
Verdict on AbsRel and delta1 (3-decimal values), like Tables 2, 3, 5:
  KHỚP d == 0 · ĐẠT d <= --tol · KHÔNG ĐẠT otherwise; plus "x/6" metrics equal to the paper.

Usage: python scripts/collect_table4.py [results/table4] [--tol 0.005]
"""
import argparse
import ast
import json
import os
import re
import sys

PAPER = dict(a1=0.982, a2=0.998, a3=1.000, abs_rel=0.046, rmse=1.896, rmse_log=0.069)
NAMES = dict(a1="δ1", a2="δ2", a3="δ3", abs_rel="AbsRel", rmse="RMSE", rmse_log="RMSE log")


def parse_log(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, encoding="utf-8", errors="replace").read())
    found = re.findall(r"\{'a1': [^}]*\}", text)
    return ast.literal_eval(found[-1]) if found else None


def judge(m, tol):
    da, dd = round(m["abs_rel"] - PAPER["abs_rel"], 3), round(m["a1"] - PAPER["a1"], 3)
    d = max(abs(da), abs(dd))
    same = sum(round(m[k], 3) == v for k, v in PAPER.items() if k in m)
    return dict(d_abs_rel=da, d_a1=dd, dist=round(d, 3), same=same,
                verdict="KHỚP" if d == 0 else ("ĐẠT" if d <= tol + 1e-9 else "KHÔNG ĐẠT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/table4")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()
    rows = []
    log = os.path.join(args.root, "kitti.log")
    if os.path.exists(log) and parse_log(log):
        rows.append(dict(run="evaluate.py", desc="lệnh chính thức `evaluate.py -d kitti`", **parse_log(log)))
    sweep = os.path.join(args.root, "sweep", "kitti.json")
    if os.path.exists(sweep):
        for name, v in json.load(open(sweep, encoding="utf-8"))["variants"].items():
            rows.append(dict(run=f"sweep:{name}", **v))
    for r in rows:
        r.update(judge(r, args.tol))

    head = "| Lượt | Mô tả | " + " | ".join(NAMES[k] for k in PAPER) + " | Khớp | Kết luận |"
    out = [f"Ngưỡng (AbsRel, δ1): KHỚP = trùng 3 chữ số; ĐẠT = |Δ| ≤ {args.tol}; còn lại KHÔNG ĐẠT.", "",
           head, "|---|---|" + "---:|" * len(PAPER) + "---|---|",
           "| **Paper** | Ours, Table 4 | " + " | ".join(f"{v:.3f}" for v in PAPER.values()) + " | | |"]
    for r in rows:
        out.append(f"| {r['run']} | {r['desc']} | " + " | ".join(f"{r[k]:.3f}" for k in PAPER)
                   + f" | {r['same']}/6 | {r['verdict']} |")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(args.root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(tol=args.tol, paper=PAPER, rows=rows), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
