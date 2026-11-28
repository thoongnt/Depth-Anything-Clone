"""NYUv2 zero-shot relative depth: CVPR 2024 Table 2.

Author's clarification: https://github.com/LiheYoung/Depth-Anything/issues/174
Alignment AND scoring are in disparity space, without min/max normalization.
NYUv2 rawDepths and full-image valid mask follow the MiDaS reference evaluator
linked by DPT. All choices are recorded; this is a reproduction, not author code.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from depth_anything.dpt import load_depth_anything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

ROOT = Path(__file__).resolve().parent
PAPER = {'vits': (0.053, 0.972), 'vitb': (0.046, 0.979), 'vitl': (0.043, 0.981)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def nyu_mask(depth, eigen_crop=False):
    if depth.shape != (480, 640):
        raise ValueError(f'Expected NYUv2 depth 480x640, got {depth.shape}')
    mask = np.ones(depth.shape, dtype=bool)
    if eigen_crop:
        mask[:] = False
        mask[45:471, 41:601] = True
    return mask & np.isfinite(depth) & (depth > 0) & (depth < 10.0)


def align_disparity(prediction, target, mask):
    """Centered float64 least squares; fail on invalid/constant predictions."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target and mask must have identical shapes')
    if not mask.any() or not np.isfinite(prediction[mask]).all():
        raise ValueError('Empty evaluation mask or non-finite prediction')
    x, y = prediction[mask].astype(np.float64), target[mask].astype(np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.dot(xc, xc)
    if denominator <= np.finfo(np.float64).eps * max(1.0, np.dot(x, x)):
        raise ValueError('Constant prediction: affine alignment is undefined')
    scale = np.dot(xc, yc) / denominator
    shift = y.mean() - scale * x.mean()
    return prediction.astype(np.float64) * scale + shift, float(scale), float(shift)


def score_disparity(prediction, depth, eigen_crop=False):
    mask = nyu_mask(depth, eigen_crop=eigen_crop)
    target = np.zeros(depth.shape, dtype=np.float64)
    np.divide(1.0, depth, out=target, where=mask)
    aligned, scale, shift = align_disparity(prediction, target, mask)
    gt, raw = target[mask], aligned[mask]
    # MiDaS caps depth at 10 m, i.e. aligned disparity at a minimum of 0.1.
    # The legacy diagnostic uses only a numerical positivity floor.
    pred = np.maximum(raw, 0.1 if not eigen_crop else 1e-8)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        'abs_rel': float(np.mean(np.abs(gt - pred) / gt)),
        'delta1': float(np.mean(ratio < 1.25)),
        'valid_pixels': int(mask.sum()),
        'nonpositive_predictions': int(np.count_nonzero(raw <= 0)),
        'scale': scale, 'shift': shift,
        # Old depth-space AbsRel is diagnostic only, not the benchmark metric.
        'depth_abs_rel_diagnostic': float(np.mean(
            np.abs(depth[mask] - 1.0 / np.maximum(raw, 1e-4)) / depth[mask])),
    }


def make_transform():
    return Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method='lower_bound',
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


@torch.inference_mode()
def evaluate_nyuv2(encoder='vitl', data_dir=ROOT / 'data',
                   checkpoint_dir=ROOT / 'checkpoints',
                   output_dir=ROOT / 'results' / 'table2' / 'nyuv2',
                   cache_dir=ROOT / '.cache' / 'nyuv2-relative',
                   limit=None, flip=False, model=None, transform=None, protocol='midas'):
    data_dir, checkpoint_dir = Path(data_dir), Path(checkpoint_dir)
    mat_path = data_dir / 'nyuv2' / 'nyu_depth_v2_labeled.mat'
    split_path = data_dir / 'nyuv2' / 'splits.mat'
    checkpoint = checkpoint_dir / f'depth_anything_{encoder}14.pth'
    for path in (mat_path, split_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    indices = scipy.io.loadmat(split_path)['testNdxs'].ravel().astype(int) - 1
    if len(indices) != 654 or len(np.unique(indices)) != 654 or np.any(indices < 0) or np.any(indices >= 1449):
        raise ValueError('Expected 654 unique official NYUv2 test indices')
    if limit is not None:
        if not 1 <= limit <= 654:
            raise ValueError('--limit must be between 1 and 654')
        indices = indices[:limit]
    if protocol not in ('midas', 'legacy-filled-eigen'):
        raise ValueError(f'Unknown protocol: {protocol}')
    eigen_crop = protocol == 'legacy-filled-eigen'
    gt_field = 'depths' if eigen_crop else 'rawDepths'
    protocol = {
        'name': protocol, 'ground_truth_field': gt_field,
        'dataset': 'NYUv2', 'encoder': encoder, 'split_sha256': sha256(split_path),
        'checkpoint_sha256': sha256(checkpoint),
        'dataset_file': str(mat_path.resolve()), 'dataset_size': mat_path.stat().st_size,
        'dataset_mtime_ns': mat_path.stat().st_mtime_ns,
        'input': 'RGB float32, ImageNet norm, short side 518, multiple 14, full image',
        'interpolation': 'OpenCV cubic input; bilinear align_corners=False prediction',
        'precision': 'float32', 'flip_augmentation': flip,
        'evaluation_mask': ('Eigen [45:471,41:601], finite GT, 0 < depth < 10 m'
                            if eigen_crop else 'Full image, finite raw GT, 0 < depth < 10 m'),
        'alignment': 'per-image float64 affine least squares on evaluation mask in disparity space',
        'metrics_space': 'disparity', 'min_prediction': 1e-8 if eigen_crop else 0.1,
        'aggregation': 'unweighted mean of per-image metrics, no skipped images',
        'source': 'https://github.com/LiheYoung/Depth-Anything/issues/174',
        'dataset_protocol_source': 'https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022',
    }
    identity = {k: protocol[k] for k in (
        'checkpoint_sha256', 'split_sha256', 'dataset_file', 'dataset_size',
        'dataset_mtime_ns', 'input', 'interpolation', 'precision', 'flip_augmentation')}
    identity['model_source_sha256'] = sha256(ROOT / 'depth_anything' / 'dpt.py')
    identity['transform_source_sha256'] = sha256(ROOT / 'depth_anything' / 'util' / 'transform.py')
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f'{encoder}-{cache_key}' if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / 'manifest.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    device = next(model.parameters()).device if model is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    transform = transform or make_transform()
    rows = []
    with h5py.File(mat_path, 'r') as dataset:
        if dataset['images'].shape != (1449, 3, 640, 480) or dataset['depths'].shape != (1449, 640, 480):
            raise ValueError('Unexpected NYUv2 labeled MAT layout')
        for index in tqdm(indices, desc=f'NYUv2 {encoder}', mininterval=10):
            cache_file = cache / f'{index:04d}.npy' if cache else None
            if cache_file and cache_file.exists():
                prediction = np.load(cache_file, allow_pickle=False)
            else:
                if model is None:
                    model = load_depth_anything(encoder, str(checkpoint_dir), str(device))
                image = dataset['images'][index].transpose(2, 1, 0).astype(np.float32) / 255.0
                tensor = torch.from_numpy(transform({'image': image})['image']).unsqueeze(0).to(device)
                prediction = model(tensor)
                if flip:
                    prediction = (prediction + model(tensor.flip(-1)).flip(-1)) * 0.5
                prediction = F.interpolate(prediction.unsqueeze(1), size=(480, 640),
                    mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                if cache_file:
                    temporary = cache_file.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream, prediction, allow_pickle=False)
                    temporary.replace(cache_file)
            rows.append({'sample_id': int(index + 1),
                         **score_disparity(prediction, dataset[gt_field][index].T, eigen_crop=eigen_crop)})
    reference = dict(zip(('abs_rel', 'delta1'), PAPER[encoder]))
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'full_test_split': len(rows) == 654, 'num_images': len(rows),
        'metrics': {k: float(np.mean([r[k] for r in rows])) for k in (
            'abs_rel', 'delta1', 'depth_abs_rel_diagnostic')},
        'paper_table': 2, 'paper_reference': reference,
        'difference_from_paper': {k: float(np.mean([r[k] for r in rows])) - v
                                  for k, v in reference.items()},
        'nonpositive_predictions': sum(r['nonpositive_predictions'] for r in rows),
        'protocol': protocol,
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'numpy': np.__version__, 'opencv': cv2.__version__, 'h5py': h5py.__version__,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None},
        'prediction_cache': str(cache) if cache else None,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = encoder + '-' + protocol['name'] + ('-flip' if flip else '') + (f'-first-{len(rows)}' if len(rows) != 654 else '')
    with (out / f'{suffix}-per-image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f'{suffix}-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=list(PAPER), default='vitl')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'table2' / 'nyuv2')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache' / 'nyuv2-relative')
    parser.add_argument('--limit', type=int, help='Diagnostic subset only, not full benchmark')
    parser.add_argument('--protocol', choices=['midas', 'legacy-filled-eigen'], default='midas')
    parser.add_argument('--flip', action='store_true', help='Optional flip augmentation, recorded separately')
    evaluate_nyuv2(**vars(parser.parse_args()))
