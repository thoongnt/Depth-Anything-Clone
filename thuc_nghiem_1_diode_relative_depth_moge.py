"""Thực nghiệm 1 trên DIODE: zero-shot relative depth (CVPR 2024 Table 2).

Biến thể dùng mask MoGe và các đối chứng trên cùng 771 ảnh. Chạy chế độ
`resolution` để tái chạy cấu hình gần paper nhất `moge-mask-short-side-392`
(ViT-L no-flip, AbsRel 0.064582 / delta1 0.952898). Xem docs/table2/diode-followup.md
để biết lệnh, dữ liệu, cache cần có và đường dẫn kết quả.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

from depth_anything.dpt import load_depth_anything
from evaluate_diode_relative import PAPER, aggregate, discover_samples
from evaluate_nyuv2 import align_disparity, make_transform, sha256

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/table2/diode/followup'
ARCHIVE = ROOT / '.cache/moge-diode/DIODE.zip'
BASE = ROOT / 'results/table2/diode/audit-rerun'
NATIVE = ROOT / 'data/diode/val'
MODES = ('mask', 'mask-flip', 'mask-vits', 'mask-vitb', 'bundle', 'resolution')


def bundle_depth(archive, key):
    # https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py#L82-L102
    with Image.open(io.BytesIO(archive.read(f'DIODE/{key}/depth.png'))) as image:
        code = np.asarray(image)
        power = (code.astype(np.float32) - 1) / 65533
        near, far = float(image.info['near']), float(image.info['far'])
        depth = near ** (1 - power) * far ** power * float(image.info.get('unit', 1))
    valid = (code != 0) & (code != 65535) & np.isfinite(depth) & (depth > 0)
    return depth, valid


def score_pair(prediction, depth, mask):
    # Same fit and pixels for both metric spaces. The second metric follows the
    # MoGe affine-disparity branch, including its epsilon and prediction clamp.
    valid = mask & np.isfinite(depth) & (depth > 0)
    disparity = np.zeros(depth.shape, dtype=np.float64)
    np.divide(1.0, depth, out=disparity, where=valid)
    aligned, scale, shift = align_disparity(prediction, disparity, valid)
    aligned, target = aligned[valid], disparity[valid]
    positive = aligned > 0
    ratio = np.full(aligned.shape, np.inf)
    ratio[positive] = np.maximum(aligned[positive]/target[positive], target[positive]/aligned[positive])
    common = {'valid_pixels': int(valid.sum()), 'scale': scale, 'shift': shift}
    raw = {**common, 'abs_rel': float(np.mean(abs(aligned-target)/target)),
           'delta1': float(np.mean(ratio < 1.25)),
           'nonpositive_predictions': int(np.count_nonzero(~positive))}
    gt = depth[valid].astype(np.float64)
    predicted_depth = 1 / np.maximum(aligned, 1 / gt.max())
    depth_ratio = np.maximum(gt/(predicted_depth+1e-6), predicted_depth/(gt+1e-6))
    clamped = {**common, 'abs_rel': float(np.mean(abs(predicted_depth-gt)/(gt+1e-6))),
               'delta1': float(np.mean(depth_ratio < 1.25)), 'nonpositive_predictions': 0}
    return [('disparity', raw), ('clamped-depth', clamped)]


def load_native_cache(sample, cache):
    item = sample['files'][0]
    key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
    return np.load(cache / f'{Path(sample["image"]).stem}-{key}.npy', allow_pickle=False)


def infer(model, transform, bgr, shape):
    if bgr is None or bgr.shape[:2] != shape:
        raise ValueError('Bundle RGB and native grid differ')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    device = next(model.parameters()).device
    tensor = torch.from_numpy(transform({'image': rgb})['image'])[None].to(device)
    return interpolate(model(tensor)[:, None], size=shape, mode='bilinear',
                       align_corners=False)[0, 0].cpu().numpy()


def save_results(mode, rows, samples, archive_identity, encoder, checkpoint_sha256):
    RESULTS.mkdir(parents=True, exist_ok=True)
    groups = sorted({(x['configuration'], x['metric_space']) for x in rows})
    output = []
    for config, space in groups:
        for subset in ('all', 'indoors', 'outdoor'):
            selected = [r for r in rows if r['configuration'] == config and r['metric_space'] == space
                        and (subset == 'all' or r['domain'] == subset)]
            expected = sum(subset == 'all' or s['domain'] == subset for s in samples)
            if len(selected) != expected:
                raise ValueError(f'{config}: scored {len(selected)}/{expected} images')
            summary = aggregate(selected)
            native_pixels = sum(r['native_valid_pixels'] for r in selected)
            intersection = sum(r['native_overlap_pixels'] for r in selected)
            output.append({'configuration': config, 'metric_space': space, 'subset': subset,
                           **summary, 'native_valid_pixels': native_pixels,
                           'native_overlap_fraction': intersection / native_pixels,
                           'scored_pixels_over_native': summary['valid_pixels'] / native_pixels})
    per_image = RESULTS / f'{mode}-per-image.csv'
    with per_image.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'mode': mode, 'encoder': encoder, 'checkpoint_sha256': checkpoint_sha256,
              'paper_reference': dict(zip(('abs_rel', 'delta1'), PAPER[encoder])),
              'archive': archive_identity,
              'note': 'Bundle scored on the stored 1024x768 grid, without a separate MoGe loader warp or depth truncation.',
              'results': output}
    (RESULTS/f'{mode}-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in output:
        print(row['configuration'], row['metric_space'], row['subset'], row['metrics'],
              'overlap', round(row['native_overlap_fraction'], 5), flush=True)


@torch.inference_mode()
def run(mode):
    samples = discover_samples(NATIVE)
    summary_path = (BASE.parent / 'variants/vitl-native-all-flip-summary.json'
                    if mode == 'mask-flip' else BASE/'fresh-inference-summary.json')
    encoder = mode.removeprefix('mask-') if mode in ('mask-vits', 'mask-vitb') else 'vitl'
    if encoder != 'vitl':
        summary_path = RESULTS / f'{encoder}-native-all-summary.json'
    baseline = json.loads(summary_path.read_text())
    if (not baseline['full_validation_split'] or baseline['protocol']['encoder'] != encoder
            or baseline['protocol']['flip_augmentation'] != (mode == 'mask-flip')):
        raise ValueError('Unexpected prediction baseline')
    manifest_hash = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    if manifest_hash != baseline['manifest_sha256']:
        raise ValueError('Native sample manifest changed')
    cache = ROOT / baseline['prediction_cache']
    model = None
    if mode in ('bundle', 'resolution'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_depth_anything('vitl', str(ROOT/'checkpoints'),
                                    'cuda' if torch.cuda.is_available() else 'cpu')
        if sha256(ROOT/'checkpoints/depth_anything_vitl14.pth') != baseline['protocol']['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from native baseline')
        transform = make_transform()
        if mode == 'resolution':
            resize = transform.transforms[0]
            transform.transforms[0] = type(resize)(width=392, height=392, resize_target=False,
                keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC)
    rows = []
    with ZipFile(ARCHIVE) as archive:
        ids = archive.read('DIODE/.index.txt').decode().splitlines()
        if len(ids) != 771 or set(ids) != {'val/'+Path(s['image']).with_suffix('').as_posix() for s in samples}:
            raise ValueError('Bundle and native validation IDs differ')
        archive_identity = {'path': str(ARCHIVE.relative_to(ROOT)), 'sha256': sha256(ARCHIVE),
                            'identical_sample_ids': len(ids)}
        for sample in tqdm(samples, desc=f'DIODE {mode}', mininterval=10):
            rgb_path, depth_path, mask_path = [NATIVE/f['path'] for f in sample['files']]
            depth = np.load(depth_path, allow_pickle=False).squeeze()
            native_mask = np.load(mask_path, allow_pickle=False).squeeze() > 0
            native_mask &= np.isfinite(depth) & (depth > 0)
            key = 'val/'+Path(sample['image']).with_suffix('').as_posix()
            processed, processed_mask = bundle_depth(archive, key)
            if processed.shape != depth.shape:
                raise ValueError(f'MoGe and native depth grids differ: {key}')
            overlap = processed_mask & native_mask
            if mode == 'resolution':
                baseline_prediction = load_native_cache(sample, cache)
                small = infer(model, transform, cv2.imread(str(rgb_path)), depth.shape)
                # Register both prediction scales using prediction only, with no GT.
                registered, _, _ = align_disparity(small, baseline_prediction, np.ones(depth.shape, bool))
                ensemble = (registered + baseline_prediction)*.5
                cases = [('native-short-side-392', small, depth, native_mask),
                         ('native-392-plus-518', ensemble, depth, native_mask),
                         ('moge-mask-short-side-392', small, depth, overlap),
                         ('moge-mask-392-plus-518', ensemble, depth, overlap)]
            else:
                if mode.startswith('mask'):
                    prediction = load_native_cache(sample, cache)
                    cases = [('native-pred-native-depth-moge-mask', prediction, depth, overlap)]
                    if mode == 'mask':
                        cases.extend([('native-pred-moge-depth-common-mask', prediction, processed, overlap),
                                      ('native-pred-moge-depth-moge-mask', prediction, processed, processed_mask)])
                else:
                    rgb = cv2.imdecode(np.frombuffer(archive.read(f'DIODE/{key}/image.jpg'), np.uint8), cv2.IMREAD_COLOR)
                    prediction = infer(model, transform, rgb, depth.shape)
                    cases = [('moge-rgb-moge-depth-moge-mask', prediction, processed, processed_mask)]
            for config, pred, target, mask in cases:
                for space, result in score_pair(pred, target, mask):
                    rows.append({'configuration': config, 'metric_space': space, 'image': sample['image'],
                                 'domain': sample['domain'], 'native_valid_pixels': int(native_mask.sum()),
                                 'native_overlap_pixels': int((native_mask & mask).sum()), **result})
    save_results(mode, rows, samples, archive_identity, encoder, baseline['protocol']['checkpoint_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modes', nargs='+', choices=MODES)
    for mode in parser.parse_args().modes:
        run(mode)
