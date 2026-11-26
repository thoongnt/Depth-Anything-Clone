"""DIODE native validation: zero-shot relative depth (CVPR 2024 Table 2).

Reproduction baseline, not the unpublished author evaluator. Uses supplied mask,
affine disparity alignment and disparity metrics, no crop or depth cap.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from depth_anything.dpt import load_depth_anything
from evaluate_nyuv2 import make_transform, align_disparity, sha256

ROOT = Path(__file__).resolve().parent
COUNTS = {'indoors': 325, 'outdoor': 446}
PAPER = {'vits': (0.076, 0.939), 'vitb': (0.069, 0.946), 'vitl': (0.066, 0.952)}


def discover_samples(val_dir, subset='all'):
    if subset not in ('all', *COUNTS):
        raise ValueError(f'Unknown subset: {subset}')
    val_dir = Path(val_dir)
    samples = []
    for domain in COUNTS if subset == 'all' else (subset,):
        files = sorted((val_dir / domain).rglob('*.png'))
        if len(files) != COUNTS[domain]:
            raise ValueError(f'{domain}: expected {COUNTS[domain]} validation images, found {len(files)}')
        for image in files:
            paths = [image, image.with_name(image.stem + '_depth.npy'),
                     image.with_name(image.stem + '_depth_mask.npy')]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            samples.append({'image': image.relative_to(val_dir).as_posix(), 'domain': domain,
                            'files': [{'path': p.relative_to(val_dir).as_posix(),
                                       'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                      for p in paths]})
    return samples


def score_diode(prediction, depth, provided_mask):
    if depth.ndim != 2 or prediction.shape != depth.shape or provided_mask.shape != depth.shape:
        raise ValueError('Prediction, depth and provided mask must be identical 2D shapes')
    mask = (provided_mask > 0) & np.isfinite(provided_mask) & np.isfinite(depth) & (depth > 0)
    target = np.zeros(depth.shape, dtype=np.float64)
    np.divide(1.0, depth, out=target, where=mask)
    aligned, scale, shift = align_disparity(prediction, target, mask)
    a, q = aligned[mask], target[mask]
    ratio = np.full(a.shape, np.inf)
    positive = a > 0
    ratio[positive] = np.maximum(a[positive] / q[positive], q[positive] / a[positive])
    return {'abs_rel': float(np.mean(np.abs(a - q) / q)),
            'delta1': float(np.mean(ratio < 1.25)), 'valid_pixels': int(mask.sum()),
            'nonpositive_predictions': int(np.count_nonzero(~positive)),
            'scale': scale, 'shift': shift}


def aggregate(rows):
    if not rows:
        raise ValueError('No evaluated images')
    return {'num_images': len(rows),
            'metrics': {k: float(np.mean([r[k] for r in rows])) for k in ('abs_rel', 'delta1')},
            'valid_pixels': sum(r['valid_pixels'] for r in rows),
            'nonpositive_predictions': sum(r['nonpositive_predictions'] for r in rows)}


@torch.inference_mode()
def evaluate_diode(encoder='vitl', data_dir=ROOT/'data', checkpoint_dir=ROOT/'checkpoints',
                   output_dir=ROOT/'results/diode-relative', cache_dir=ROOT/'.cache/diode-relative',
                   subset='all', limit=None, flip=False, model=None, transform=None,
                   data_audit=None, refresh_cache=False):
    if encoder not in PAPER:
        raise ValueError(f'Unknown encoder: {encoder}')
    val_dir = Path(data_dir).resolve() / 'diode/val'
    verification = None
    if data_audit is not None:
        from audit_diode import verify_reference_report
        verification = verify_reference_report(val_dir, data_audit)
    samples = discover_samples(val_dir, subset)
    total = len(samples)
    if limit is not None:
        if not 1 <= limit <= total:
            raise ValueError(f'--limit must be between 1 and {total}')
        samples = samples[:limit]
    checkpoint = Path(checkpoint_dir) / f'depth_anything_{encoder}14.pth'
    checkpoint_hash = sha256(checkpoint)
    protocol = {
        'name': 'native-mask-disparity', 'dataset': 'DIODE', 'split': 'native validation',
        'encoder': encoder, 'checkpoint_sha256': checkpoint_hash,
        'input': 'RGB float32/255, short side >=518, aspect preserved, multiple14, ImageNet norm',
        'interpolation': 'OpenCV cubic input; bilinear align_corners=False prediction',
        'precision': 'float32 inference; float64 affine fit and scoring',
        'flip_augmentation': flip,
        'evaluation_mask': 'provided mask > 0, finite mask and depth, depth > 0; full frame',
        'depth_cap': None, 'prediction_clipping': None,
        'alignment': 'per-image least squares scale and shift to reciprocal GT on evaluation mask',
        'metrics_space': 'disparity', 'nonpositive_delta1': 'count as failure; keep GT mask',
        'aggregation': 'unweighted mean of per-image metrics; no skipped images',
        'source': 'https://github.com/LiheYoung/Depth-Anything/issues/174',
        'paper_protocol_fully_confirmed': False,
        'unconfirmed_details': ['exact paper split/manifest', 'depth caps and clipping', 'TTA'],
    }
    identity = {'checkpoint_sha256': checkpoint_hash, 'flip': flip, 'input': protocol['input'],
                'interpolation': protocol['interpolation'], 'precision': 'float32',
                'model_source': sha256(ROOT/'depth_anything/dpt.py'),
                'transform_source': sha256(ROOT/'depth_anything/util/transform.py'),
                'preprocess_source': sha256(ROOT/'evaluate_nyuv2.py'),
                'evaluator_source': sha256(Path(__file__)), 'dataset_root': str(val_dir)}
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f'{encoder}-{cache_key}' if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache/'manifest.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    device = next(model.parameters()).device if model is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    transform, rows, hits = transform or make_transform(), [], 0
    if model is not None:
        model.eval()
    for sample in tqdm(samples, desc=f'DIODE {encoder} {subset}', mininterval=10):
        image_file, depth_file, mask_file = [val_dir/f['path'] for f in sample['files']]
        depth = np.load(depth_file, allow_pickle=False).squeeze()
        mask = np.load(mask_file, allow_pickle=False).squeeze()
        if depth.shape != (768, 1024) or mask.shape != depth.shape:
            raise ValueError(f'Invalid DIODE shape: {image_file}: {depth.shape}, {mask.shape}')
        # File metadata is part of each cache key so edited images invalidate predictions.
        image_key = hashlib.sha256(json.dumps(sample['files'][0], sort_keys=True).encode()).hexdigest()[:16]
        cached = cache/f'{image_file.stem}-{image_key}.npy' if cache else None
        if cached and cached.exists() and not refresh_cache:
            prediction = np.load(cached, allow_pickle=False)
            hits += 1
        else:
            bgr = cv2.imread(str(image_file))
            if bgr is None or bgr.shape != (768, 1024, 3):
                raise ValueError(f'Unreadable or invalid RGB image: {image_file}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(transform({'image': rgb})['image'])[None].to(device)
            if model is None:
                model = load_depth_anything(encoder, str(checkpoint_dir), str(device)).eval()
            prediction = model(tensor)
            if flip:
                prediction = (prediction + model(tensor.flip(-1)).flip(-1)) * 0.5
            prediction = F.interpolate(prediction[:, None], size=depth.shape, mode='bilinear',
                                       align_corners=False)[0, 0].cpu().numpy()
            if cached:
                temporary = cached.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, prediction, allow_pickle=False)
                temporary.replace(cached)
        try:
            metrics = score_diode(prediction, depth, mask)
        except ValueError as error:
            raise ValueError(f'{sample["image"]}: {error}') from error
        rows.append({'image': sample['image'], 'domain': sample['domain'], **metrics})
    reference = dict(zip(('abs_rel', 'delta1'), PAPER[encoder]))
    manifest_hash = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), **aggregate(rows),
               'subset': subset, 'full_requested_split': len(rows) == total,
               'full_validation_split': subset == 'all' and len(rows) == 771,
               'by_domain': {d: aggregate([r for r in rows if r['domain'] == d])
                             for d in COUNTS if any(r['domain'] == d for r in rows)},
               'paper_table': 2, 'paper_reference': reference,
               'paper_comparison_scope': 'Published DIODE aggregate; domain/subset runs are diagnostic',
               'difference_from_paper': {k: aggregate(rows)['metrics'][k]-v for k,v in reference.items()}
                    if subset == 'all' and len(rows) == 771 else None,
               'protocol': protocol, 'manifest_sha256': manifest_hash,
               'prediction_cache': str(cache) if cache else None, 'cache_hits': hits,
               'data_integrity_verification': verification, 'refresh_cache': refresh_cache,
               'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'numpy': np.__version__, 'opencv': cv2.__version__,
                               'device': str(device), 'gpu': torch.cuda.get_device_name(device)
                               if device.type == 'cuda' else None}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f'{encoder}-native-{subset}' + ('-flip' if flip else '')
    if len(rows) != total:
        suffix += f'-first-{len(rows)}'
    with (out/f'{suffix}-per-image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in [('summary', summary), ('manifest', samples)]:
        (out/f'{suffix}-{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=list(PAPER), default='vitl')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/diode-relative')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'.cache/diode-relative')
    parser.add_argument('--subset', choices=['all', *COUNTS], default='all')
    parser.add_argument('--limit', type=int, help='Diagnostic first N only, not a paper reproduction')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--data-audit', type=Path, help='Recheck all dataset bytes against authenticated audit report')
    parser.add_argument('--refresh-cache', action='store_true', help='Force fresh inference, ignoring cached predictions')
    evaluate_diode(**vars(parser.parse_args()))
