"""
================================================================================
 THỰC NGHIỆM 1: ƯỚC LƯỢNG ĐỘ SÂU TƯƠNG ĐỐI ZERO-SHOT (ZERO-SHOT RELATIVE DEPTH)
================================================================================
 Nguồn tham khảo: Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data", CVPR 2024.
 Bảng 2: Đánh giá khả năng tổng quát hóa (Zero-shot) trên 6 bộ dữ liệu công bố:
   1. NYUv2   (654 ảnh test chuẩn)
   2. DIODE   (771 ảnh val chuẩn: indoor + outdoor)
   3. KITTI   (652 ảnh Eigen test split)
   4. ETH3D   (454 ảnh có nhãn, thuc_nghiem_1_eth3d_relative_depth.py)
   5. DDAD    (3,950 ảnh validation split)
   6. Sintel  (1,064 ảnh test/val)
================================================================================
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

# Fix Windows console UTF-8 output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Thêm thư mục hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


def compute_scale_and_shift(prediction, target, mask):
    """
    Căn chỉnh Affine-Invariant chuẩn (Least Squares Scale & Shift Alignment):
    Giải hệ phương trình tối ưu: min_{s, t} sum ( (s * pred + t - target)^2 ) trên vùng mask hợp lệ.
    Lưu ý: Mô hình relative xuất ra Disparity nên s thường mang giá trị âm khi căn chỉnh với Depth (mét).
    """
    a = prediction[mask]
    b = target[mask]

    if len(a) == 0:
        return 1.0, 0.0

    system_mat = np.zeros((2, 2))
    system_mat[0, 0] = np.sum(a * a)
    system_mat[0, 1] = np.sum(a)
    system_mat[1, 0] = system_mat[0, 1]
    system_mat[1, 1] = np.sum(mask)

    rhs = np.zeros((2, 1))
    rhs[0, 0] = np.sum(a * b)
    rhs[1, 0] = np.sum(b)

    try:
        scale_shift = np.linalg.solve(system_mat, rhs)
        scale = float(scale_shift[0, 0])
        shift = float(scale_shift[1, 0])
    except np.linalg.LinAlgError:
        scale = 1.0
        shift = 0.0

    return scale, shift


def compute_metrics(gt, pred, mask):
    """
    Tính toán 2 chỉ số tiêu chuẩn:
      - AbsRel (Absolute Relative Error) ↓
      - Delta 1 (tỷ lệ pixel có max(pred/gt, gt/pred) < 1.25) ↑
    """
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    if len(gt_valid) == 0:
        return 0.0, 0.0

    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    ratio = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    delta1 = np.mean(ratio < 1.25)

    return abs_rel, delta1


def read_sintel_depth(filepath):
    """Đọc file độ sâu định dạng .dpt của Sintel"""
    with open(filepath, 'rb') as f:
        TAG_FLOAT = 202021.25
        tag = np.fromfile(f, np.float32, count=1)[0]
        if tag != TAG_FLOAT:
            raise ValueError(f"Sintel depth file invalid tag: {tag}")
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=width * height)
        return data.reshape((height, width))


@torch.no_grad()
def run_experiment_1(encoder='vitl', data_dir='./data', checkpoint_dir='./checkpoints', dataset='nyuv2', subset='all', limit=None, flip=False, data_audit=None, refresh_cache=False):
    from evaluate_nyuv2 import evaluate_nyuv2
    if dataset == 'nyuv2':
        return evaluate_nyuv2(encoder=encoder, data_dir=data_dir, checkpoint_dir=checkpoint_dir)
    from evaluate_diode_relative import evaluate_diode
    if dataset == 'diode':
        return evaluate_diode(encoder=encoder, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
                              subset=subset, limit=limit, flip=flip, data_audit=data_audit, refresh_cache=refresh_cache)
    if dataset != 'all':
        raise ValueError(f'Unknown dataset: {dataset}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 85)
    print(f"THỰC NGHIỆM 1: ZERO-SHOT RELATIVE DEPTH ESTIMATION")
    print(f"Mô hình: Depth Anything ({encoder.upper()}) | Thiết bị: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 85)

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'depth_anything_{encoder}14.pth')
    if not os.path.exists(checkpoint_path):
        print(f"[LỖI] Không tìm thấy checkpoint tại: {checkpoint_path}")
        return

    print(f"[Model] Nạp trọng số từ: {checkpoint_path}")
    model = DepthAnything(model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device).eval()

    transform = Compose([
        Resize(
            width=518, height=518, resize_target=False, keep_aspect_ratio=True,
            ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    results = []

    # Bảng tham chiếu CVPR 2024 Table 2
    paper_benchmarks = {
        'NYUv2': {'type': 'Indoor (Trong nhà)', 'target_samples': 654, 'paper_abs_rel': 0.043, 'paper_delta1': 0.981},
        'DIODE': {'type': 'Indoor + Outdoor', 'target_samples': 771, 'paper_abs_rel': 0.066, 'paper_delta1': 0.952},
        'KITTI': {'type': 'Urban Driving', 'target_samples': 652, 'paper_abs_rel': 0.076, 'paper_delta1': 0.947},
        'ETH3D': {'type': 'Indoor + Outdoor', 'target_samples': 454, 'paper_abs_rel': 0.127, 'paper_delta1': 0.882},
        'DDAD': {'type': 'Multi-camera Driving', 'target_samples': 3950, 'paper_abs_rel': 0.230, 'paper_delta1': 0.789},
        'Sintel': {'type': '3D CGI Animation', 'target_samples': 1064, 'paper_abs_rel': 0.458, 'paper_delta1': 0.760},
    }

    # Shared NYUv2 evaluator: standalone CLI, notebook and this script.
    summary = evaluate_nyuv2(encoder=encoder, data_dir=data_dir,
                             checkpoint_dir=checkpoint_dir, model=model, transform=transform)
    bm = paper_benchmarks['NYUv2']
    results.append({
        'Dataset': 'NYUv2', 'Loại môi trường': bm['type'],
        'Số ảnh đã chạy': summary['num_images'],
        'Our AbsRel ↓': f"{summary['metrics']['abs_rel']:.4f}",
        'Paper AbsRel ↓': f"{summary['paper_reference']['abs_rel']:.4f}",
        'Our Delta1 ↑': f"{summary['metrics']['delta1']:.4f}",
        'Paper Delta1 ↑': f"{summary['paper_reference']['delta1']:.4f}",
        'Trạng thái': 'Đã hoàn thành (disparity)',
    })

    # DIODE shares the audited disparity evaluator with the standalone CLI/notebook.
    diode_summary = evaluate_diode(encoder=encoder, data_dir=data_dir,
        checkpoint_dir=checkpoint_dir, model=model, transform=transform,
        subset=subset, limit=limit, flip=flip, data_audit=data_audit, refresh_cache=refresh_cache)
    results.append({
        'Dataset': 'DIODE', 'Loại môi trường': 'Indoor + Outdoor',
        'Số ảnh đã chạy': diode_summary['num_images'],
        'Our AbsRel ↓': f"{diode_summary['metrics']['abs_rel']:.4f}",
        'Paper AbsRel ↓': f"{diode_summary['paper_reference']['abs_rel']:.4f}",
        'Our Delta1 ↑': f"{diode_summary['metrics']['delta1']:.4f}",
        'Paper Delta1 ↑': f"{diode_summary['paper_reference']['delta1']:.4f}",
        'Trạng thái': 'Đã chạy; protocol DIODE native, chưa xác nhận khớp paper',
    })

    # 3. KITTI, 4. ETH3D, 5. DDAD, 6. Sintel
    for ds_name in ['KITTI', 'ETH3D', 'DDAD', 'Sintel']:
        bm = paper_benchmarks[ds_name]
        row = {
            'Dataset': ds_name, 'Loại môi trường': bm['type'], 'Số ảnh đã chạy': 0,
            'Our AbsRel ↓': '-', 'Paper AbsRel ↓': f"{bm['paper_abs_rel']:.4f}",
            'Our Delta1 ↑': '-', 'Paper Delta1 ↑': f"{bm['paper_delta1']:.4f}",
            'Trạng thái': 'Bỏ qua (Chưa có data)',
        }
        if ds_name == 'ETH3D':
            from thuc_nghiem_1_eth3d_relative_depth import PAPER as eth3d_paper
            paper_abs_rel, paper_delta1 = eth3d_paper[encoder]
            row['Paper AbsRel ↓'] = f'{paper_abs_rel:.4f}'
            row['Paper Delta1 ↑'] = f'{paper_delta1:.4f}'
            row['Trạng thái'] = 'Xem thuc_nghiem_1_eth3d_relative_depth.py'
            root = Path(__file__).resolve().parent / 'results/eth3d-relative'
            for tag in (f'{encoder}-518', f'{encoder}-518-flip'):
                path = root / f'{tag}-summary.json'
                if not path.is_file():
                    continue
                saved = json.loads(path.read_text(encoding='utf-8'))
                baseline = next((metric for metric in saved['comparisons']
                                 if metric['images'] == 454 and metric['subset'] == 'all'
                                 and metric['variant'] == 'no-flip' and metric['cap'] is None
                                 and metric['space'] == 'disparity'
                                 and metric['weighting'] == 'image'), None)
                if baseline:
                    row['Số ảnh đã chạy'] = baseline['images']
                    row['Our AbsRel ↓'] = f"{baseline['abs_rel']:.4f}"
                    row['Our Delta1 ↑'] = f"{baseline['delta1']:.4f}"
                    row['Trạng thái'] = 'Kết quả đã lưu, MoGe 518 px'
                    break
        results.append(row)

    # BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM 1
    df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print(f"BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM 1: ZERO-SHOT RELATIVE DEPTH (Depth Anything {encoder.upper()})")
    print("=" * 90)
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print("=" * 90)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Chạy Thực nghiệm 1: Zero-shot Relative Depth Estimation")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--dataset', choices=['nyuv2', 'diode', 'all'], default='nyuv2',
                        help='Default: only the audited NYUv2 benchmark')
    parser.add_argument('--subset', choices=['all', 'indoors', 'outdoor'], default='all', help='DIODE only')
    parser.add_argument('--limit', type=int, help='DIODE diagnostic first N only')
    parser.add_argument('--flip', action='store_true', help='DIODE flip TTA')
    parser.add_argument('--data-audit', help='DIODE authenticated integrity report')
    parser.add_argument('--refresh-cache', action='store_true', help='DIODE fresh inference')
    args = parser.parse_args()
    if args.dataset == 'nyuv2' and (args.limit is not None or args.flip or args.subset != 'all' or args.data_audit or args.refresh_cache):
        parser.error('Use evaluate_nyuv2.py for NYUv2-specific options')

    run_experiment_1(encoder=args.encoder, data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, dataset=args.dataset, subset=args.subset, limit=args.limit, flip=args.flip, data_audit=args.data_audit, refresh_cache=args.refresh_cache)
