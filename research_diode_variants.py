"""Exploratory DIODE protocol comparisons; never overrides the baseline evaluator.

Full configured subsets and mask-retention counts are reported. Numbers selected
by proximity to the paper are post-hoc exploration, not an official reproduction.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_nyuv2 import align_disparity, sha256

ROOT = Path(__file__).resolve().parent
REFERENCE = {'abs_rel': .066, 'delta1': .952}
CONFIGS = [
    {'name': 'native', 'alignment': 'ols'},
    {'name': 'native-clamp-gt', 'alignment': 'ols', 'floor': 'gt-max'},
    {'name': 'native-depth-metric', 'alignment': 'ols', 'floor': 'gt-max', 'space': 'depth'},
    {'name': 'range-80m', 'alignment': 'ols', 'max_depth': 80},
    {'name': 'range-0.5-80m', 'alignment': 'ols', 'min_depth': .5, 'max_depth': 80},
    {'name': 'range-0.6-80m', 'alignment': 'ols', 'min_depth': .6, 'max_depth': 80},
    {'name': 'range-10m', 'alignment': 'ols', 'max_depth': 10},
    {'name': 'range-10m-clamp10', 'alignment': 'ols', 'max_depth': 10, 'floor': .1},
    {'name': 'range-0.5-10m', 'alignment': 'ols', 'min_depth': .5, 'max_depth': 10},
    {'name': 'nyu-eigen-fraction-crop', 'alignment': 'ols', 'crop': True},
    {'name': 'median-mad', 'alignment': 'median-mad'},
    {'name': 'relative-ls', 'alignment': 'relative-ls'},
    {'name': 'huber-8', 'alignment': 'huber'},
]


def weighted_fit(x, y, weight):
    w = weight / weight.sum()
    mx, my = np.dot(w, x), np.dot(w, y)
    xc = x - mx
    denominator = np.dot(w, xc * xc)
    if denominator <= np.finfo(float).eps * max(1., np.dot(w, x*x)):
        raise ValueError('Constant prediction under weighted fit')
    scale = np.dot(w, xc*(y-my)) / denominator
    return float(scale), float(my-scale*mx)


def alternative_fit(x, y, method):
    x, y = x.astype(np.float64), y.astype(np.float64)
    if method == 'median-mad':
        mx, my = np.median(x), np.median(y)
        deviation = np.mean(np.abs(x-mx))
        if deviation <= np.finfo(float).eps:
            raise ValueError('Constant prediction')
        scale = np.mean(np.abs(y-my))/deviation
        return float(scale), float(my-scale*mx)
    if method == 'relative-ls':
        # Minimize sum(((s*x+t-y)/y)**2), different from ordinary LS.
        return weighted_fit(x, y, 1/y**2)
    if method == 'huber':
        scale, shift = weighted_fit(x, y, np.ones_like(x))
        for _ in range(8):
            residual = scale*x+shift-y
            sigma = 1.4826 * np.median(np.abs(residual-np.median(residual)))
            if np.max(np.abs(residual)) < 1e-10:
                break
            # MAD can be zero when most residuals are identical but outliers remain.
            sigma = max(sigma, 1e-9)
            weights = np.minimum(1., 1.345*sigma/np.maximum(np.abs(residual), 1e-12))
            scale, shift = weighted_fit(x, y, weights)
        return scale, shift
    raise ValueError(method)


def evaluation_mask(depth, provided, config):
    valid = (provided > 0) & np.isfinite(provided) & np.isfinite(depth) & (depth > 0)
    valid &= depth > config.get('min_depth', 0.)
    if 'max_depth' in config:
        valid &= depth < config['max_depth']
    if config.get('crop'):
        h, w = depth.shape
        crop = np.zeros(depth.shape, dtype=bool)
        crop[int(45/480*h):int(471/480*h), int(41/640*w):int(601/640*w)] = True
        valid &= crop
    return valid


def score_vectors(pred, gt):
    positive = pred > 0
    ratios = np.full(pred.shape, np.inf)
    ratios[positive] = np.maximum(pred[positive]/gt[positive], gt[positive]/pred[positive])
    return {'abs_rel': float(np.mean(np.abs(pred-gt)/gt)),
            'delta1': float(np.mean(ratios < 1.25)),
            'nonpositive_predictions': int(np.count_nonzero(~positive))}


def summarize(rows, expected, aggregation='image-mean'):
    good = [r for r in rows if r['status'] == 'ok']
    weights = None if aggregation == 'image-mean' else [r['valid_pixels'] for r in good]
    metrics = {k: float(np.average([r[k] for r in good], weights=weights))
               if good else None for k in REFERENCE}
    baseline = sum(r['baseline_valid_pixels'] for r in rows)
    valid = sum(r['valid_pixels'] for r in good)
    complete = len(good) == expected
    return {'num_images': len(good), 'expected_images': expected,
            'unscorable_images': [r['image'] for r in rows if r['status'] != 'ok'],
            'full_subset': complete, 'valid_pixels': valid,
            'native_valid_pixel_retention': valid/baseline if baseline else 0.,
            'aggregation': aggregation, **metrics,
            'abs_rel_difference': metrics['abs_rel']-.066 if good else None,
            'delta1_difference': metrics['delta1']-.952 if good else None,
            'matches_both_paper_rounding': complete and all(
                float(f'{metrics[k]:.3f}') == v for k,v in REFERENCE.items()),
            'max_absolute_metric_gap': max(abs(metrics[k]-v) for k,v in REFERENCE.items()) if good else None}


def run(summary_path, output_dir, workers=1):
    summary_path, output_dir = Path(summary_path), Path(output_dir)
    s = json.loads(summary_path.read_text(encoding='utf-8'))
    if not s['full_validation_split'] or s['num_images'] != 771:
        raise ValueError('Require full 771-image prediction summary')
    if s['protocol']['encoder'] != 'vitl':
        raise ValueError('This comparison targets the ViT-L paper reference')
    flip = s['protocol']['flip_augmentation']
    tag = 'flip' if flip else 'no-flip'
    stem = 'vitl-native-all' + ('-flip' if flip else '')
    manifest_path = summary_path.parent/f'{stem}-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (manifest_hash != s['manifest_sha256'] or len(manifest) != 771
            or len({sample['image'] for sample in manifest}) != 771):
        raise ValueError('Invalid or incomplete input manifest')
    identity = {'summary_sha256': sha256(summary_path), 'source_sha256': sha256(Path(__file__)),
                'manifest_sha256': s['manifest_sha256'], 'prediction_cache': s['prediction_cache']}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    dest = output_dir/f'{tag}-{key}'
    dest.mkdir(parents=True, exist_ok=True)
    rows_path = dest/'per-image.csv'
    if rows_path.exists():
        with rows_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for k in ['valid_pixels', 'baseline_valid_pixels', 'nonpositive_predictions']:
                row[k] = int(row[k])
            for k in ['abs_rel', 'delta1', 'scale', 'shift']:
                row[k] = float(row[k]) if row[k] else None
    else:
        rows = []
        cache = Path(s['prediction_cache'])
        data = Path(s.get('protocol', {}).get('data_root', ROOT/'data/diode/val'))
        def score_sample(sample):
            sample_rows = []
            rgb = sample['files'][0]
            image_key = hashlib.sha256(json.dumps(rgb, sort_keys=True).encode()).hexdigest()[:16]
            prediction_file = cache/f'{Path(sample["image"]).stem}-{image_key}.npy'
            pred = np.load(prediction_file, allow_pickle=False)
            depth = np.load(data/sample['files'][1]['path'], allow_pickle=False).squeeze()
            supplied = np.load(data/sample['files'][2]['path'], allow_pickle=False).squeeze()
            base_valid = evaluation_mask(depth, supplied, {})
            target = np.zeros(depth.shape, dtype=np.float64)
            np.divide(1., depth, out=target, where=base_valid)
            native_fit = None
            for config in CONFIGS:
                mask = evaluation_mask(depth, supplied, config)
                row = {'configuration': config['name'], 'image': sample['image'], 'domain': sample['domain'],
                       'status': 'ok', 'valid_pixels': int(mask.sum()), 'baseline_valid_pixels': int(base_valid.sum()),
                       'nonpositive_predictions': 0, 'abs_rel': None, 'delta1': None, 'scale': None, 'shift': None}
                if not mask.any():
                    row['status'] = 'empty-mask'
                else:
                    x, y = pred[mask].astype(np.float64), target[mask]
                    try:
                        if config['alignment'] == 'ols':
                            if config['name'] in ('native', 'native-clamp-gt', 'native-depth-metric') and native_fit:
                                scale, shift = native_fit
                            else:
                                _, scale, shift = align_disparity(pred, target, mask)
                            if config['name'] == 'native':
                                native_fit = (scale, shift)
                        else:
                            scale, shift = alternative_fit(x, y, config['alignment'])
                        a = scale*x+shift
                        floor = config.get('floor')
                        if floor is not None:
                            a = np.maximum(a, 1/float(depth[mask].max()) if floor == 'gt-max' else floor)
                        if config.get('space') == 'depth':
                            a, y = 1/a, depth[mask]
                        row.update(score_vectors(a, y), scale=scale, shift=shift)
                    except ValueError as error:
                        row['status'] = str(error)
                sample_rows.append(row)
            return sample_rows

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, sample_rows in enumerate(pool.map(score_sample, manifest), 1):
                rows.extend(sample_rows)
                if i % 25 == 0 or i == len(manifest):
                    print(f'{tag} protocol grid: {i}/771 images, {len(rows)} rows', flush=True)
        temporary = rows_path.with_suffix('.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(rows_path)
    table = []
    for config in CONFIGS:
        for subset, count in [('all', 771), ('indoors', 325), ('outdoor', 446)]:
            selected = [r for r in rows if r['configuration'] == config['name']
                        and (subset == 'all' or r['domain'] == subset)]
            for aggregation in (['image-mean', 'pixel-mean'] if config['name'] == 'native' else ['image-mean']):
                table.append({'flip': flip, 'configuration': config['name'], 'subset': subset,
                              **summarize(selected, count, aggregation)})
    report = {'scope': 'Exploratory post-hoc protocol comparison; not author-confirmed Table2 reproduction',
              'paper_reference': REFERENCE, 'configs': CONFIGS, 'identity': identity,
              'per_image_csv': str(rows_path), 'completed_prediction_files': len(manifest),
              'results': table}
    (dest/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/diode-relative/variants/protocol-grid')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    reports = [run(p, args.output_dir, args.workers) for p in args.summary]
    result = {'scope': reports[0]['scope'], 'paper_reference': REFERENCE,
              'configs': CONFIGS, 'results': [row for report in reports for row in report['results']]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = [{k:v for k,v in row.items() if k != 'unscorable_images'} for row in result['results']]
    with (args.output_dir/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    candidates = sorted([r for r in result['results'] if r['full_subset']], key=lambda r:r['max_absolute_metric_gap'])
    print(json.dumps({'num_results': len(rows), 'closest_full_subset_rows': candidates[:8]}, indent=2))
