#!/usr/bin/env bash
#
# download_data.sh — fetch the small evaluation/benchmark datasets used in
# docs/data.md (zero-shot relative depth, metric depth, and semantic
# segmentation transfer). This intentionally does NOT try to download the
# 1.5M-image labeled training set or the 62M+ image unlabeled training set
# from the original Depth Anything paper — those are terabyte-scale, and
# several require registration/paid access. See docs/data.md, section 1.
#
# Usage:
#   ./download_data.sh <dataset> [<dataset> ...]
#   ./download_data.sh all        # every dataset with a direct, no-login URL
#   ./download_data.sh list       # show status of every known dataset
#
# Datasets with a stable, official, no-login URL (downloaded automatically):
#   nyuv2       NYUv2 official labeled .mat + train/test split indices
#   kitti       KITTI Eigen split raw images + annotated depth ground truth
#   diode       DIODE validation split (indoor + outdoor, i.e. also covers
#               "DIODE Outdoor" used in the zero-shot metric depth table)
#   sunrgbd     SUN RGB-D full RGB-D dataset (includes the 5,050 test images)
#   vkitti2     Virtual KITTI 2 (rgb + depth)
#   ade20k      ADE20K (ADEChallengeData2016), validation split
#
# Datasets that require your own account credentials (scripted login, but
# YOU must have registered first):
#   cityscapes  Requires CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars
#
# Datasets with no stable single-command official URL (login wall, request
# form, or dataset too large to fit "small experiment" scope). Running these
# just prints where to go and where to put the files:
#   sintel      MPI-Sintel depth benchmark (account required)
#   eth3d       ETH3D high-res multi-view benchmark (pick subset yourself)
#   ibims1      iBims-1 indoor benchmark (request form on project page)
#   hypersim    Apple HyperSim (huge; use their official downloader tool)
#   ddad        TRI-ML DDAD (257GB train+val tar, no validation-only split)
#
# Env vars:
#   DATA_ROOT   Where to put everything. Default: ./data
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KITTI_EIGEN_TEST_FILE="${SCRIPT_DIR}/metric_depth/train_test_inputs/kitti_eigen_test_files_with_gt.txt"

log()  { printf '\n\033[1;34m[download_data]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[download_data][WARN]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[download_data][ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# fetch <url> <output_path>  — resumable download, skips if already present.
fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    log "Already downloaded, skipping: $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  log "Downloading: $url"
  curl -L --fail --retry 3 -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}

extract() {
  local archive="$1" dest="$2"
  shift 2
  mkdir -p "$dest"
  case "$archive" in
    *.zip)
      unzip -q -o "$archive" -d "$dest" "$@"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$dest" "$@"
      ;;
    *.tar)
      tar -xf "$archive" -C "$dest" "$@"
      ;;
    *)
      die "Don't know how to extract: $archive"
      ;;
  esac
}

manual_instructions() {
  local name="$1" url="$2" note="$3" dest="$4"
  warn "$name has no single-command official download (login wall / request form / too large)."
  printf '  - Visit:        %s\n' "$url"
  printf '  - Note:         %s\n' "$note"
  printf '  - Put files in: %s\n' "$dest"
}

# ---------------------------------------------------------------------------
# NYUv2 — official Silberman lab release: full labeled .mat (1449 images)
# plus the official train/test split indices. This repo's split files under
# metric_depth/train_test_inputs/nyudepthv2_test_files_with_gt.txt reference
# 654 of these frames as the test set.
# ---------------------------------------------------------------------------
download_nyuv2() {
  local dest="$DATA_ROOT/nyuv2"
  log "NYUv2 -> $dest"
  fetch "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" \
        "$dest/nyu_depth_v2_labeled.mat"
  fetch "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat" \
        "$dest/splits.mat"
  log "NYUv2 done. nyu_depth_v2_labeled.mat (all 1449 frames) + splits.mat (official train/test indices)."
  log "See metric_depth/train_test_inputs/nyudepthv2_test_files_with_gt.txt for the 654-image test list."
}

# ---------------------------------------------------------------------------
# KITTI — Eigen test split. Images come from the official raw data zips,
# ground truth comes from the official annotated depth completion zip.
# The exact (date, drive) pairs needed are parsed from this repo's own
# kitti_eigen_test_files_with_gt.txt, so we only fetch what's required.
# ---------------------------------------------------------------------------
download_kitti() {
  [ -f "$KITTI_EIGEN_TEST_FILE" ] || die "Cannot find $KITTI_EIGEN_TEST_FILE"
  local dest="$DATA_ROOT/kitti"
  log "KITTI (Eigen test split) -> $dest"

  local drives dates
  drives=$(awk '{print $1}' "$KITTI_EIGEN_TEST_FILE" | cut -d/ -f2 | sort -u)
  dates=$(printf '%s\n' "$drives" | cut -c1-10 | sort -u)

  log "Raw drives needed: $(printf '%s\n' "$drives" | wc -l | tr -d ' ') sequences across $(printf '%s\n' "$dates" | wc -l | tr -d ' ') dates."

  local date drive
  for date in $dates; do
    fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}/${date}_calib.zip" \
          "$dest/raw/${date}_calib.zip"
    extract "$dest/raw/${date}_calib.zip" "$dest/raw"
  done
  for drive in $drives; do
    date="${drive:0:10}"
    fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}/${drive}.zip" \
          "$dest/raw/${drive}.zip"
    extract "$dest/raw/${drive}.zip" "$dest/raw"
  done

  warn "Ground-truth depth (data_depth_annotated.zip) is ~14GB for train+val combined; there is no official val-only download."
  fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip" \
        "$dest/data_depth_annotated.zip"
  log "Extracting only the val/ scenes needed for the Eigen test split..."
  for drive in $drives; do
    extract "$dest/data_depth_annotated.zip" "$dest/gt" "val/${drive}/*" || true
  done
  log "KITTI done. Raw images: $dest/raw, depth ground truth: $dest/gt/val."
}

# ---------------------------------------------------------------------------
# DIODE — official S3 bucket. The validation tarball already contains both
# indoor and outdoor scenes, i.e. it covers both "DIODE" (zero-shot relative
# depth, 771 images) and "DIODE Outdoor" (zero-shot metric depth, 325 images)
# from docs/data.md.
# ---------------------------------------------------------------------------
download_diode() {
  local dest="$DATA_ROOT/diode"
  log "DIODE (validation, indoor+outdoor) -> $dest"
  fetch "http://diode-dataset.s3.amazonaws.com/val.tar.gz" "$dest/val.tar.gz"
  extract "$dest/val.tar.gz" "$dest"
  log "DIODE done. Outdoor subset is under $dest/val/outdoor, indoor under $dest/val/indoor."
}

# ---------------------------------------------------------------------------
# SUN RGB-D — official Princeton release (full dataset; the standard 5,050
# image test split is a subset of this, selected via the official toolbox's
# allsplit.mat).
# ---------------------------------------------------------------------------
download_sunrgbd() {
  local dest="$DATA_ROOT/sunrgbd"
  log "SUN RGB-D -> $dest (this is ~6.8GB and includes all 10,335 images, not just the test split)"
  fetch "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip" "$dest/SUNRGBD.zip"
  extract "$dest/SUNRGBD.zip" "$dest"
  log "SUN RGB-D done. Use the official SUNRGBDtoolbox allsplit.mat to select the 5,050 test images."
}

# ---------------------------------------------------------------------------
# Virtual KITTI 2 — official NAVER LABS Europe release.
# ---------------------------------------------------------------------------
download_vkitti2() {
  local dest="$DATA_ROOT/vkitti2"
  log "Virtual KITTI 2 -> $dest"
  fetch "http://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar" "$dest/vkitti_2.0.3_rgb.tar"
  extract "$dest/vkitti_2.0.3_rgb.tar" "$dest"
  fetch "http://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar" "$dest/vkitti_2.0.3_depth.tar"
  extract "$dest/vkitti_2.0.3_depth.tar" "$dest"
  log "Virtual KITTI 2 done."
}

# ---------------------------------------------------------------------------
# ADE20K — official MIT CSAIL release (SceneParsing benchmark).
# ---------------------------------------------------------------------------
download_ade20k() {
  local dest="$DATA_ROOT/ade20k"
  log "ADE20K -> $dest"
  fetch "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" "$dest/ADEChallengeData2016.zip"
  extract "$dest/ADEChallengeData2016.zip" "$dest"
  log "ADE20K done. Validation split (2,000 images): $dest/ADEChallengeData2016/images/validation."
}

# ---------------------------------------------------------------------------
# Cityscapes — requires a free account at cityscapes-dataset.com. Provide
# CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD as env vars to let this script
# log in and download for you (same flow as the official csDownload tool).
# ---------------------------------------------------------------------------
download_cityscapes() {
  local dest="$DATA_ROOT/cityscapes"
  if [ -z "${CITYSCAPES_USERNAME:-}" ] || [ -z "${CITYSCAPES_PASSWORD:-}" ]; then
    manual_instructions "Cityscapes" "https://www.cityscapes-dataset.com/downloads/" \
      "Register a free account, then re-run with CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD set (we need gtFine_trainvaltest.zip [packageID=1] and leftImg8bit_trainvaltest.zip [packageID=3])." \
      "$dest"
    return 0
  fi
  log "Cityscapes -> $dest"
  local cookies="$dest/.cookies"
  mkdir -p "$dest"
  curl -sc "$cookies" --data "username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" \
       https://www.cityscapes-dataset.com/login/ -o /dev/null

  fetch_authed() {
    local package_id="$1" out="$2"
    [ -f "$out" ] && { log "Already downloaded, skipping: $out"; return 0; }
    curl -Lb "$cookies" --fail --retry 3 -o "$out.part" \
         "https://www.cityscapes-dataset.com/file-handling/?packageID=${package_id}"
    mv "$out.part" "$out"
  }

  fetch_authed 1 "$dest/gtFine_trainvaltest.zip"
  fetch_authed 3 "$dest/leftImg8bit_trainvaltest.zip"
  rm -f "$cookies"
  extract "$dest/gtFine_trainvaltest.zip" "$dest"
  extract "$dest/leftImg8bit_trainvaltest.zip" "$dest"
  log "Cityscapes done. Validation split (500 images, 1024x2048): $dest/leftImg8bit/val and $dest/gtFine/val."
}

# ---------------------------------------------------------------------------
# Datasets with no reliable single-command official URL.
# ---------------------------------------------------------------------------
download_sintel() {
  manual_instructions "Sintel depth benchmark" "https://sintel-depth.csail.mit.edu/landing" \
    "Free account required (separate from the optical-flow site) before you can download the depth training/test data." \
    "$DATA_ROOT/sintel"
}

download_eth3d() {
  manual_instructions "ETH3D" "https://www.eth3d.net/datasets" \
    "Pick the 'low-res many-view / two-view' training+test data (several download options, up to ~12GB); no single canonical zip." \
    "$DATA_ROOT/eth3d"
}

download_ibims1() {
  manual_instructions "iBims-1" "https://www.asg.ed.tum.de/lmf/ibims1/" \
    "The dataset (100 images) is distributed on request from the project page; no stable public zip URL." \
    "$DATA_ROOT/ibims1"
}

download_hypersim() {
  manual_instructions "HyperSim" "https://github.com/apple/ml-hypersim" \
    "Full dataset is ~1.9TB. Use their official 'dataset_download_images.py' tool and select only the scenes you need." \
    "$DATA_ROOT/hypersim"
}

download_ddad() {
  manual_instructions "DDAD" "https://github.com/TRI-ML/DDAD" \
    "Only a single combined train+val tarball is published officially (~257GB, curl https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar); there is no validation-only download." \
    "$DATA_ROOT/ddad"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
AUTO_DATASETS="nyuv2 kitti diode sunrgbd vkitti2 ade20k"
ALL_DATASETS="$AUTO_DATASETS cityscapes sintel eth3d ibims1 hypersim ddad"

usage() {
  sed -n '2,33p' "${BASH_SOURCE[0]}"
}

list_status() {
  local name
  for name in $ALL_DATASETS; do
    if [ -d "$DATA_ROOT/$name" ] && [ -n "$(ls -A "$DATA_ROOT/$name" 2>/dev/null || true)" ]; then
      printf '  [x] %s (found under %s)\n' "$name" "$DATA_ROOT/$name"
    else
      printf '  [ ] %s\n' "$name"
    fi
  done
}

main() {
  [ "$#" -eq 0 ] && { usage; exit 1; }
  need_cmd curl
  need_cmd unzip
  need_cmd tar
  need_cmd awk

  local targets=()
  for arg in "$@"; do
    case "$arg" in
      all)  targets+=($AUTO_DATASETS) ;;
      list) list_status; exit 0 ;;
      *)    targets+=("$arg") ;;
    esac
  done

  local t
  for t in "${targets[@]}"; do
    case "$t" in
      nyuv2)      download_nyuv2 ;;
      kitti)      download_kitti ;;
      diode)      download_diode ;;
      sunrgbd)    download_sunrgbd ;;
      vkitti2)    download_vkitti2 ;;
      ade20k)     download_ade20k ;;
      cityscapes) download_cityscapes ;;
      sintel)     download_sintel ;;
      eth3d)      download_eth3d ;;
      ibims1)     download_ibims1 ;;
      hypersim)   download_hypersim ;;
      ddad)       download_ddad ;;
      *) die "Unknown dataset: $t (known: $ALL_DATASETS)" ;;
    esac
  done
  log "All requested datasets processed. Data root: $DATA_ROOT"
}

main "$@"
