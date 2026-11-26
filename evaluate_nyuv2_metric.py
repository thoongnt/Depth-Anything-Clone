"""NYUv2 in-domain metric depth evaluation: CVPR 2024, Table 3.

Uses the released NYUv2-finetuned indoor checkpoint, ZoeDepth configuration,
flip inference and metric computation. Reads the already downloaded labeled MAT
with BTS-equivalent RGB JPEG and raw-depth millimeter conversion in memory. No GT-based prediction alignment.
"""
import argparse
import csv
from io import BytesIO
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import cv2
from PIL import Image
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'metric_depth'))
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import compute_metrics
from evaluate_nyuv2 import sha256

PAPER = {'delta1': 0.984, 'delta2': 0.998, 'delta3': 1.000,
         'abs_rel': 0.056, 'rmse': 0.206, 'log10': 0.024}


def prepare_bts_nyu_image(image):
    """Reproduce BTS test export + PIL loader in memory, including JPEG quality 95."""
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError('Expected NYUv2 uint8 RGB image, shape 480x640x3')
    bgr = image[:, :, ::-1]
    boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    boundary[7:474, 7:632] = bgr[7:474, 7:632]
    ok, encoded = cv2.imencode('.jpg', boundary, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('BTS JPEG encoding failed')
    with Image.open(BytesIO(encoded.tobytes())) as decoded:
        return np.array(decoded, dtype=np.float32) / 255.0


def prepare_bts_nyu_depth(depth):
    """BTS export truncates rawDepths*1000 to uint16 PNG; loader divides by 1000."""
    if depth.shape != (480, 640):
        raise ValueError('Expected NYUv2 depth 480x640')
    with np.errstate(invalid='ignore'):
        return (depth * 1000.0).astype(np.uint16).astype(np.float32) / 1000.0


def metric_mask(depth):
    if depth.shape != (480, 640):
        raise ValueError(f'Expected depth shape (480, 640), got {depth.shape}')
    mask = np.zeros_like(depth, dtype=bool)
    mask[45:471, 41:601] = True
    return mask & np.isfinite(depth) & (depth > 1e-3) & (depth < 10.0)


def score_metric_depth(prediction, depth):
    """Official metric implementation, Eigen mask, direct meters, no affine fit."""
    mask = metric_mask(depth)
    if not mask.any():
        raise ValueError('No valid NYUv2 metric evaluation pixels')
    if prediction.shape != depth.shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction must be finite and have the GT resolution')
    # The official function clamps predictions in place: protect cached arrays.
    pred = torch.from_numpy(np.array(prediction, dtype=np.float32, copy=True))[None, None]
    gt = torch.from_numpy(np.array(depth, dtype=np.float32, copy=True))[None, None]
    metrics = compute_metrics(gt, pred, dataset='nyu', eigen_crop=True,
                              min_depth_eval=1e-3, max_depth_eval=10.0)
    names = {'delta1': 'a1', 'delta2': 'a2', 'delta3': 'a3',
             'abs_rel': 'abs_rel', 'rmse': 'rmse', 'log10': 'log_10'}
    result = {name: float(metrics[key]) for name, key in names.items()}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Non-finite metric result')
    return {**result, 'valid_pixels': int(mask.sum())}


@torch.inference_mode()
def infer_metric(model, image, flip=True):
    """Matches metric_depth/evaluate.py: flip average before output resize."""
    pred = model(image, dataset='nyu')['metric_depth']
    if flip:
        pred_flip = model(image.flip(-1), dataset='nyu')['metric_depth'].flip(-1)
        pred = (pred + pred_flip) * 0.5
    if pred.shape[-2:] != (480, 640):
        pred = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=True)
    return pred[0, 0].float().cpu().numpy()


def evaluate_nyuv2_metric(data_dir=ROOT / 'data', checkpoint_dir=ROOT / 'checkpoints',
                          output_dir=ROOT / 'results' / 'nyuv2-metric',
                          cache_dir=ROOT / '.cache' / 'nyuv2-metric',
                          limit=None, flip=True, device=None):
    data_dir, checkpoint_dir = Path(data_dir), Path(checkpoint_dir)
    mat = data_dir / 'nyuv2' / 'nyu_depth_v2_labeled.mat'
    split = data_dir / 'nyuv2' / 'splits.mat'
    checkpoint = checkpoint_dir / 'depth_anything_metric_depth_indoor.pt'
    encoder = checkpoint_dir / 'depth_anything_vitl14.pth'
    for path in (mat, split, checkpoint, encoder):
        if not path.is_file():
            raise FileNotFoundError(path)
    indices = scipy.io.loadmat(split)['testNdxs'].ravel().astype(int) - 1
    if len(indices) != 654 or len(set(indices)) != 654 or min(indices) < 0 or max(indices) >= 1449:
        raise ValueError('Expected the 654 unique official NYUv2 test indices')
    if limit is not None:
        if not 1 <= limit <= 654:
            raise ValueError('--limit must be in [1, 654]')
        indices = indices[:limit]
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    config = get_config('zoedepth', 'eval', 'nyu',
                        pretrained_resource=f'local::{checkpoint.resolve()}',
                        encoder_checkpoint=str(encoder.resolve()))
    source = hashlib.sha256()
    for path in sorted((ROOT / 'metric_depth' / 'zoedepth').rglob('*.py')):
        source.update(str(path.relative_to(ROOT)).replace('\\', '/').encode())
        source.update(path.read_bytes())
    identity = {
        'checkpoint_sha256': sha256(checkpoint), 'encoder_checkpoint_sha256': sha256(encoder),
        'model_source_sha256': source.hexdigest(), 'split_sha256': sha256(split),
        'dataset': str(mat.resolve()), 'dataset_size': mat.stat().st_size,
        'dataset_mtime_ns': mat.stat().st_mtime_ns, 'flip': flip,
        'input': 'BTS export: black boundary [7:474,7:632], OpenCV JPEG quality 95, PIL decode RGB float32 [0,1]; internal model norm/resize',
        'opencv': cv2.__version__,
        'img_size': config.img_size, 'keep_aspect_ratio': config.get('force_keep_ar', False),
        'output_resize': 'bilinear align_corners=True, after flip averaging',
        'precision': 'float32', 'torch': torch.__version__,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f'indoor-{key}' if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / 'manifest.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    model = None
    rows = []
    start = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with h5py.File(mat, 'r') as data:
        if data['images'].shape != (1449, 3, 640, 480) or data['rawDepths'].shape != (1449, 640, 480):
            raise ValueError('Unexpected NYUv2 labeled MAT layout')
        for idx in tqdm(indices, desc='NYUv2 metric indoor', mininterval=10):
            cp = cache / f'{idx:04d}.npy' if cache else None
            if cp and cp.exists():
                prediction = np.load(cp, allow_pickle=False)
            else:
                if model is None:
                    model = build_model(config).to(device).eval()
                image = prepare_bts_nyu_image(data['images'][idx].transpose(2, 1, 0))
                tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(device)
                prediction = infer_metric(model, tensor, flip=flip)
                if cp:
                    with cp.with_suffix('.tmp').open('wb') as stream:
                        np.save(stream, prediction, allow_pickle=False)
                    cp.with_suffix('.tmp').replace(cp)
            # Follow the BTS export used by the official ZoeDepth/metric loader.
            depth = prepare_bts_nyu_depth(data['rawDepths'][idx].T)
            rows.append({'sample_id': int(idx + 1), **score_metric_depth(prediction, depth)})
    metrics = {name: float(np.mean([r[name] for r in rows])) for name in PAPER}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment': 'in-domain metric depth, evaluation of released NYUv2-finetuned checkpoint',
        'paper_table': 3, 'num_images': len(rows), 'full_test_split': len(rows) == 654,
        'metrics': metrics, 'paper_reference': PAPER,
        'difference_from_paper': {k: metrics[k] - v for k, v in PAPER.items()},
        'protocol': {**identity, 'ground_truth_field': 'rawDepths; BTS uint16 millimeter truncation then /1000',
                     'mask': 'Eigen [45:471,41:601], finite GT, 0.001 < depth < 10 m',
                     'alignment': 'none; no median scaling or affine fitting',
                     'prediction_clip_meters': [0.001, 10.0],
                     'aggregation': 'unweighted mean of per-image metrics; no skips',
                     'sources': ['https://github.com/LiheYoung/Depth-Anything/tree/main/metric_depth',
                                 'https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/evaluate.py',
                                 'https://github.com/cleinc/bts/blob/master/utils/extract_official_train_test_set_from_mat.py']},
        'prediction_cache': str(cache) if cache else None,
        'elapsed_seconds': time.monotonic() - start,
        'environment': {'torch': torch.__version__, 'numpy': np.__version__,
                        'device': str(device), 'gpu': torch.cuda.get_device_name(device)
                        if device.type == 'cuda' else None},
        'training_performed': False,
    }
    result['matches_paper_at_3_decimals'] = {k: round(metrics[k], 3) == v for k, v in PAPER.items()}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = 'indoor' + ('-flip' if flip else '-no-flip') + (f'-first-{len(rows)}' if len(rows) != 654 else '')
    with (out / f'{stem}-per-image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f'{stem}-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'nyuv2-metric')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache' / 'nyuv2-metric')
    parser.add_argument('--limit', type=int, help='Diagnostic subset only, separate output filenames')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--no-flip', action='store_false', dest='flip', default=True)
    evaluate_nyuv2_metric(**vars(parser.parse_args()))
