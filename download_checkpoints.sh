#!/usr/bin/env bash
# download_checkpoints.sh — Tai checkpoint Depth Anything V1 tu Hugging Face
# Trang chinh thuc:
#   Relative: https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints
#   Metric:   https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth
# Dung luong ky vong: ViT-S ~99 MB, ViT-B ~390 MB, ViT-L ~1.34 GB; metric moi file ~1.34 GB.
#
# Cach dung: bash download_checkpoints.sh [CKPT_DIR]   (mac dinh ./checkpoints)
set -euo pipefail

CKPT_DIR="${1:-./checkpoints}"
BASE="https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main"
mkdir -p "$CKPT_DIR"

# Kiem tra dia trong >= 6 GB
avail_kb=$(df -Pk "$CKPT_DIR" | awk 'NR==2{print $4}')
avail_gb=$(( avail_kb / 1024 / 1024 ))
echo "[disk] Trong: ${avail_gb} GB tai $CKPT_DIR (can ~6 GB)"
[ "$avail_gb" -ge 6 ] || { echo "CANH BAO: dia trong < 6 GB" >&2; exit 1; }

dl() { # dl <url> <file> <min_bytes> <label>
  local url="$1" file="$2" minb="$3" label="$4"
  echo "=== [$label] $url"
  wget -c --tries=3 --timeout=60 -O "$CKPT_DIR/$file" "$url"
  local sz
  sz=$(stat -c%s "$CKPT_DIR/$file")
  echo "  -> $file : $(( sz / 1024 / 1024 )) MB (toi thieu ky vong $(( minb / 1024 / 1024 )) MB)"
  if [ "$sz" -lt "$minb" ]; then
    echo "  LOI: file nho hon ky vong — co the tai do dang/HTML loi. Kiem tra lai link tren trang HF." >&2
    exit 1
  fi
}

# Relative (Table 2) — min bytes dat o ~80% dung luong ky vong de bat file loi
dl "$BASE/checkpoints/depth_anything_vits14.pth" "depth_anything_vits14.pth" 80000000  "ViT-S relative"
dl "$BASE/checkpoints/depth_anything_vitb14.pth" "depth_anything_vitb14.pth" 320000000 "ViT-B relative"
dl "$BASE/checkpoints/depth_anything_vitl14.pth" "depth_anything_vitl14.pth" 1100000000 "ViT-L relative"

# Metric (Table 3-5, dung voi metric_depth/evaluate.py)
dl "$BASE/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt"  "depth_anything_metric_depth_indoor.pt"  1100000000 "Metric indoor (NYUv2)"
dl "$BASE/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt" "depth_anything_metric_depth_outdoor.pt" 1100000000 "Metric outdoor (KITTI)"

echo "Xong. Danh sach checkpoint:"
ls -lh "$CKPT_DIR"
echo "Luu y: dung V1 (Apache-2.0). Dung nham checkpoint V2 (CC-BY-NC, ten file khac)."
