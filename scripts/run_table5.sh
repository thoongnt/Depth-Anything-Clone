#!/usr/bin/env bash
# Reproduce Table 5 (zero-shot metric depth) of Depth Anything.
# Run from the repo root:  bash scripts/run_table5.sh [diode|ibims|vkitti2|sunrgbd|vkitti2_seeds|diode_antialias ...]
# Logs go to results/table5/<run>.log, summary via scripts/collect_table5.py.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$REPO/.venv/bin/python"
OUT="$REPO/results/table5"; mkdir -p "$OUT"
INDOOR="local::./checkpoints/depth_anything_metric_depth_indoor.pt"
OUTDOOR="local::./checkpoints/depth_anything_metric_depth_outdoor.pt"

ev() {  # ev <run-name> <ckpt> <dataset> [key=value ...]
  local name="$1" ckpt="$2" ds="$3"; shift 3
  echo "[table5] $name"
  (cd "$REPO/metric_depth" && "$PY" evaluate.py -m zoedepth --pretrained_resource="$ckpt" -d "$ds" "$@") \
    > "$OUT/$name.log" 2>&1
}

targets=("${@:-diode ibims vkitti2 sunrgbd}")
for t in ${targets[@]}; do
  case "$t" in
    diode)   ev diode_outdoor "$OUTDOOR" diode_outdoor ;;
    ibims)   ev ibims "$INDOOR" ibims ;;
    vkitti2) ev vkitti2_seed0 "$OUTDOOR" vkitti2 --vkitti2_split_seed=0 ;;
    # Extra runs: split variance of the (unreleased, random) vKITTI2 test split,
    # and DIODE with torchvision>=0.17's default antialiased resize.
    vkitti2_seeds) for s in 1 2 3 4; do ev "vkitti2_seed$s" "$OUTDOOR" vkitti2 --vkitti2_split_seed=$s; done ;;
    diode_antialias) ev diode_outdoor_antialias "$OUTDOOR" diode_outdoor --diode_resize_antialias=True ;;
    sunrgbd)
      # Paper protocol: ankurhanda's SUN RGB-D test set (= official depth_bfx PNG read /10000,
      # i.e. GT = 0.8 x true depth). The other variants are scored from the same predictions.
      "$PY" "$REPO/scripts/eval_sunrgbd_variants.py" \
        --variant sunrgbd_paper=data/sunrgbd/test_ahanda:10000 \
        --variant mm_bfx=data/sunrgbd/test_mm_bfx:1000 \
        --variant mm=data/sunrgbd/test_mm:1000 \
        --variant raw=data/sunrgbd/test_raw:10000 \
        --out results/table5/sunrgbd_variants.json > "$OUT/sunrgbd_variants.txt" 2>&1 ;;
    *) echo "unknown target $t" >&2; exit 1 ;;
  esac
done
"$PY" "$REPO/scripts/collect_table5.py" "$OUT"
