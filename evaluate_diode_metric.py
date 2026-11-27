"""DIODE metric depth using the released V1 indoor/outdoor checkpoints.

Primary scores follow the released ZoeDepth evaluator. The second score applies
DIODE's sensor validity mask while retaining the primary crop and depth bounds.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

from evaluate_diode_relative import COUNTS, discover_samples
from evaluate_nyuv2 import sha256

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'metric_depth'))
from zoedepth.data.diode import DIODE
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import compute_metrics


@torch.inference_mode()
def evaluate(domain, output_dir, limit=None):
    val_dir = ROOT / 'data/diode/val'
    folder = 'indoors' if domain == 'indoor' else 'outdoor'
    samples = discover_samples(val_dir, folder)
    if limit is not None:
        if not 1 <= limit <= len(samples):
            raise ValueError(f'limit must be between 1 and {len(samples)}')
        samples = samples[:limit]
    dataset = DIODE(str(val_dir / folder))
    dataset.image_files, dataset.depth_files, dataset.depth_mask_files = (
        [str(val_dir / sample['files'][i]['path']) for sample in samples] for i in range(3)
    )
    checkpoint = ROOT / 'checkpoints' / f'depth_anything_metric_depth_{domain}.pt'
    encoder = ROOT / 'checkpoints/depth_anything_vitl14.pth'
    config = get_config('zoedepth', 'eval', f'diode_{domain}',
                        pretrained_resource=f'local::{checkpoint}', encoder_checkpoint=str(encoder))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(config).to(device).eval()
    rows = []
    # Iterate the released dataset in process to avoid Windows worker startup.
    for index, sample_record in enumerate(tqdm(samples, desc=f'DIODE metric {domain}', mininterval=10)):
        sample = dataset[index]
        rgb, gt = sample['image'][None].to(device), sample['depth'][None]
        pred = model(rgb, dataset=sample['dataset'])['metric_depth']
        flipped = model(rgb.flip(-1), dataset=sample['dataset'])['metric_depth'].flip(-1)
        pred = interpolate((pred + flipped) * .5, size=gt.shape[-2:],
                           mode='bilinear', align_corners=True).cpu()
        if pred.shape != gt.shape or not pred.isfinite().all():
            raise ValueError(f'Invalid prediction for {sample_record["image"]}')
        scores = compute_metrics(gt, pred.clone(), config=config)
        sensor_mask = torch.from_numpy(np.load(dataset.depth_mask_files[index], allow_pickle=False).squeeze() > 0)
        if sensor_mask.shape != gt.shape[-2:]:
            raise ValueError(f'Invalid sensor mask for {sample_record["image"]}')
        sensor_scores = compute_metrics(gt.masked_fill(~sensor_mask, 0), pred.clone(), config=config)
        row = {'image': sample_record['image']}
        for label, score in [('released', scores), ('sensor_mask', sensor_scores)]:
            for name, key in [('abs_rel', 'abs_rel'), ('delta1', 'a1')]:
                value = float(score[key])
                if not np.isfinite(value):
                    raise ValueError(f'Invalid {label} score on {sample_record["image"]}')
                row[f'{label}_{name}'] = value
        rows.append(row)
    primary = {k: float(np.mean([r['released_'+k] for r in rows])) for k in ('abs_rel', 'delta1')}
    sensor = {k: float(np.mean([r['sensor_mask_'+k] for r in rows])) for k in ('abs_rel', 'delta1')}
    paper = {'abs_rel': .794, 'delta1': .288} if domain == 'outdoor' else None
    result = {'domain': domain, 'images': len(rows), 'expected_images': COUNTS[folder],
              'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint),
              'encoder_sha256': sha256(encoder), 'device': str(device),
              'protocol': {'units': 'meters', 'flip': True, 'affine_alignment': False,
                           'loader_rgb_short_side': 480, 'model_input_shape': config.img_size,
                           'output_resize': 'bilinear align_corners=True',
                           'crop': 'Garg' if config.garg_crop else 'Eigen',
                           'depth_range_m': [config.min_depth_eval, config.max_depth_eval],
                           'primary_mask': 'Released evaluator crop and GT depth bounds',
                           'secondary_mask': 'Primary mask intersected with DIODE sensor validity',
                           'aggregation': 'mean of per-image metrics'},
              'released': primary, 'sensor_mask': sensor, 'paper_table5': paper,
              'full_domain': len(rows) == COUNTS[folder]}
    output_dir.mkdir(parents=True, exist_ok=True)
    name = domain if result['full_domain'] else f'{domain}-first-{len(rows)}'
    with (output_dir / f'{name}-per-image.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f'{name}-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['outdoor', 'indoor', 'both'], default='both')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/diode-metric')
    parser.add_argument('--limit', type=int, help='Optional smoke test limit for each domain')
    args = parser.parse_args()
    for domain in ('outdoor', 'indoor') if args.domain == 'both' else (args.domain,):
        evaluate(domain, args.output_dir, args.limit)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
