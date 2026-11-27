"""Experiment 1: ETH3D zero-shot relative depth, CVPR 2024 Table 2."""
import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
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
from depth_anything.util.transform import Resize
from evaluate_nyuv2 import make_transform, sha256

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/eth3d-relative'
RESULTS = ROOT / 'results/eth3d-relative'
NATIVE = ROOT / 'data/eth3d/multi_view_training_dslr_undistorted'
OFFICIAL = CACHE / 'official-stride3'
PAPER = {'vitl': (0.127, 0.882), 'vitb': (0.126, 0.884), 'vits': (0.127, 0.885)}
INDOOR = {'delivery_area', 'kicker', 'office', 'pipes', 'relief', 'relief_2', 'terrains'}
CAPS = (None, 50, 80, 200)


def discover(archive, native_root):
    names = set(archive.namelist())
    keys = sorted(name.removeprefix('ETH3D/').removesuffix('/image.jpg')
                  for name in archive.namelist() if name.endswith('/image.jpg'))
    if len(keys) != 454 or len(set(keys)) != 454:
        raise ValueError('Expected 454 unique ETH3D images')
    if any(f'ETH3D/{key}/{file}' not in names for key in keys
           for file in ('image.jpg', 'depth.png', 'meta.json')):
        raise ValueError('Incomplete RGB/depth/calibration pairs')
    if native_root.is_dir():
        native_keys = {f'{path.parents[2].name}/{path.stem}'
                       for path in native_root.glob('*/images/dslr_images_undistorted/*')
                       if path.suffix.lower() == '.jpg'}
        if set(keys) != native_keys:
            raise ValueError('MoGe image IDs do not match local ETH3D DSLR image IDs')
    return keys


def load_depth(archive, key):
    # MoGe stores finite depth as log-quantized uint16. 0 and 65535 are invalid.
    with Image.open(BytesIO(archive.read(f'ETH3D/{key}/depth.png'))) as image:
        codes = np.array(image)
        near, far = float(image.info['near']), float(image.info['far'])
        unit = float(image.info.get('unit', 1))
    valid = (codes != 0) & (codes != 65535)
    depth = np.zeros(codes.shape, dtype=np.float32)
    power = (codes[valid].astype(np.float32) - 1) / 65533
    depth[valid] = near ** (1 - power) * far ** power * unit
    if not np.all(np.isfinite(depth[valid]) & (depth[valid] > 0)):
        raise ValueError(f'Invalid decoded ground truth for {key}')
    return depth, valid


def score(x, gt, cap, fit='ols'):
    """Fit and score raw disparity, plus clamped metric depth as a control."""
    target = 1 / gt
    if fit == 'median-mad':
        center_x, center_y = np.median(x), np.median(target)
        xc = x - center_x
        spread = np.mean(np.abs(xc))
        if spread <= 0 or not np.isfinite(spread):
            raise ValueError('Constant or non-finite prediction in evaluation mask')
        aligned = xc * (np.mean(np.abs(target - center_y)) / spread) + center_y
    else:
        xc = x - x.mean()
        variance = np.sum(xc * xc)
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError('Constant or non-finite prediction in evaluation mask')
        scale = np.sum(xc * (target - target.mean())) / variance
        aligned = scale * xc + target.mean()
    abs_rel = np.mean(np.abs(aligned - target) / target)
    delta1 = np.mean((aligned > target / 1.25) & (aligned < target * 1.25))
    predicted_depth = 1 / np.maximum(aligned, 1 / (cap or gt.max()))
    return {
        'valid_pixels': int(x.size), 'abs_rel': float(abs_rel), 'delta1': float(delta1),
        'depth_abs_rel': float(np.mean(np.abs(predicted_depth - gt) / gt)),
        'depth_delta1': float(np.mean((predicted_depth > gt / 1.25)
                                      & (predicted_depth < gt * 1.25))),
        'nonpositive_predictions': int(np.count_nonzero(aligned <= 0)),
    }


def summarize(rows, encoder):
    paper_abs_rel, paper_delta1 = PAPER[encoder]
    summary = []
    for variant in dict.fromkeys(row['variant'] for row in rows):
        for cap in CAPS:
            for subset in ('all', 'indoor', 'outdoor'):
                selected = [r for r in rows if r['variant'] == variant
                            and r['cap'] == cap and (subset == 'all' or r['subset'] == subset)]
                if not selected:
                    continue
                weights = [r['valid_pixels'] for r in selected]
                for space, fields in (('disparity', ('abs_rel', 'delta1')),
                                      ('clamped-depth', ('depth_abs_rel', 'depth_delta1'))):
                    for weighting in ('image', 'pixel', 'scene'):
                        values = []
                        for field in fields:
                            if weighting == 'scene':
                                scenes = {r['scene'] for r in selected}
                                value = np.mean([np.mean([r[field] for r in selected
                                                if r['scene'] == scene]) for scene in sorted(scenes)])
                            else:
                                value = np.average([r[field] for r in selected],
                                                   weights=weights if weighting == 'pixel' else None)
                            values.append(float(value))
                        abs_rel, delta1 = values
                        summary.append({
                            'variant': variant, 'cap': cap, 'subset': subset, 'space': space,
                            'weighting': weighting, 'images': len(selected),
                            'valid_pixels': sum(weights), 'abs_rel': abs_rel, 'delta1': delta1,
                            'max_paper_gap': max(abs(abs_rel - paper_abs_rel),
                                                 abs(delta1 - paper_delta1)),
                        })
    return summary


@torch.inference_mode()
def run(args):
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This ETH3D evaluation requires CUDA for the validated attention backend')
    checkpoint = ROOT / 'checkpoints' / f'depth_anything_{args.encoder}14.pth'
    with ZipFile(args.archive) as archive:
        keys = discover(archive, args.native_root)
    scene_counts = dict(Counter(key.split('/')[0] for key in keys))
    selected_keys = keys[:args.limit]
    if args.images_per_scene:
        selected_keys = []
        for scene, scene_count in scene_counts.items():
            scene_keys = [key for key in keys if key.startswith(f'{scene}/')]
            count = min(args.images_per_scene, scene_count)
            # Use the middle index of each equal-size bin in name order.
            indices = ((np.arange(count) + 0.5) * scene_count / count).astype(int)
            selected_keys.extend(scene_keys[i] for i in indices)
    manifests = {}
    if args.source == 'official':
        for scene in sorted({key.split('/')[0] for key in selected_keys}):
            manifest = json.loads((args.official_root / scene / 'manifest.json').read_text(
                encoding='utf-8'))
            expected = {key.split('/')[1] for key in keys if key.startswith(f'{scene}/')}
            if set(manifest['images']) != expected:
                raise ValueError(f'Official ETH3D manifest differs for {scene}')
            manifests[scene] = manifest

    identity = {
        'data_source': args.source, 'archive_sha256': sha256(args.archive),
        'official_manifests': manifests, 'checkpoint_sha256': sha256(checkpoint),
        'transform_sha256': sha256(ROOT / 'depth_anything/util/transform.py'),
        'model_sha256': sha256(ROOT / 'depth_anything/dpt.py'),
        'preprocessing': 'v1-rgb-f32-cubic-lower-bound-multiple14-no-exif',
        'torch': torch.__version__, 'opencv': cv2.__version__, 'device': args.device,
    }
    # Preserve the keys of completed MoGe predictions.
    cache_identity = {k: v for k, v in identity.items()
                      if k not in ('data_source', 'official_manifests')}
    scene_cache_keys = {}
    if manifests:
        cache_identity['data_source'] = 'official-stride3'
        scene_cache_keys = {
            scene: hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
            for scene, manifest in manifests.items()
        }
    cache_id = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = CACHE / f'{args.encoder}-{args.size}-{cache_id}'
    if args.cache_predictions:
        cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    transform = make_transform()
    if args.size != 518:
        transform.transforms[0] = Resize(
            width=args.size, height=args.size, resize_target=False, keep_aspect_ratio=True,
            ensure_multiple_of=14, resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC)
    rows = []
    caps = CAPS if args.fit == 'ols' else (None,)
    model = None
    with ZipFile(args.archive) as archive:
        for key in tqdm(selected_keys, desc=f'{args.source} {args.encoder} {args.size}',
                        mininterval=15):
            scene, name = key.split('/')
            flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            if args.source == 'official':
                image_path = args.official_root / key
                with np.load(image_path.with_suffix('.npz'), allow_pickle=False) as data:
                    depth = data['depth']
                mask = np.isfinite(depth) & (depth > 0)
                rgb = cv2.imread(str(image_path.with_suffix('.jpg')), flags)
            else:
                depth, mask = load_depth(archive, key)
                encoded = np.frombuffer(archive.read(f'ETH3D/{key}/image.jpg'), np.uint8)
                rgb = cv2.imdecode(encoded, flags)
            if rgb is None or rgb.shape[:2] != depth.shape or not mask.any():
                raise ValueError(f'Invalid RGB/depth pair for {key}')
            variants = {}
            tensor = None
            for flipped in (False, True) if args.flip else (False,):
                scene_key = scene_cache_keys.get(scene, scene)
                path = cache / f'{scene_key}-{name}-{int(flipped)}.npy'
                if args.cache_predictions and path.is_file():
                    raw = np.load(path, allow_pickle=False)
                else:
                    if model is None:
                        model = load_depth_anything(args.encoder, checkpoint.parent, args.device)
                    if tensor is None:
                        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                        tensor = torch.from_numpy(transform({'image': image})['image'])
                        tensor = tensor.unsqueeze(0).to(args.device)
                    prediction = model(tensor.flip((-1,)) if flipped else tensor)[0]
                    raw = (prediction.flip((-1,)) if flipped else prediction).cpu().numpy()
                    if args.cache_predictions:
                        temporary = path.with_suffix('.part')
                        with temporary.open('wb') as stream:
                            np.save(stream, raw, allow_pickle=False)
                        temporary.replace(path)
                if not flipped:
                    variants['no-flip'] = raw
                else:
                    variants['flip-average'] = (variants['no-flip'] + raw) / 2

            gt = depth[mask].astype(np.float64)
            selections = {cap: gt < cap for cap in caps if cap is not None and gt.max() >= cap}
            for variant, raw in variants.items():
                prediction = interpolate(torch.from_numpy(raw)[None, None], size=depth.shape,
                                         mode='bilinear', align_corners=False)[0, 0].numpy()
                x = prediction[mask].astype(np.float64)
                baseline = score(x, gt, None, args.fit)
                for cap in caps:
                    if cap in selections:
                        selected = selections[cap]
                        if not selected.any():
                            continue
                        metrics = score(x[selected], gt[selected], cap)
                    elif cap is not None:
                        metrics = {**baseline}
                        # The capped fit has the same pixels. Its depth diagnostic
                        # still uses its own prediction clamp.
                        controls = score(x, gt, cap)
                        metrics.update(depth_abs_rel=controls['depth_abs_rel'],
                                       depth_delta1=controls['depth_delta1'])
                    else:
                        metrics = baseline
                    rows.append({'image': key, 'scene': scene, 'variant': variant, 'cap': cap,
                                 'subset': 'indoor' if scene in INDOOR else 'outdoor', **metrics})

    summary = summarize(rows, args.encoder)
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.encoder}-{args.size}' + ('-flip' if args.flip else '')
    if args.fit != 'ols':
        tag += f'-{args.fit}'
    if args.source == 'official':
        tag = f'official-stride3-{tag}'
    if args.limit is not None:
        tag += f'-limit{args.limit}'
    if args.images_per_scene:
        tag += f'-{args.images_per_scene}per-scene'
    for kind, data in (('per-image', rows), ('comparison', summary)):
        with (args.output / f'{tag}-{kind}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    metadata = {
        'identity': identity, 'encoder': args.encoder, 'short_side': args.size,
        'flip': args.flip, 'fit': args.fit, 'limit': args.limit,
        'images_per_scene': args.images_per_scene, 'archive': str(args.archive),
        'archive_scenes': scene_counts, 'native_ids_verified': args.native_root.is_dir(),
        'paper': dict(zip(('abs_rel', 'delta1'), PAPER[args.encoder])),
        'comparisons': summary,
    }
    (args.output / f'{tag}-summary.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                     encoding='utf-8')
    for row in summary:
        if row['subset'] == 'all' and row['space'] == 'disparity' and row['weighting'] == 'image':
            print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('moge', 'official'), default='moge')
    parser.add_argument('--official-root', type=Path, default=OFFICIAL)
    parser.add_argument('--archive', type=Path, default=CACHE / 'ETH3D.zip')
    parser.add_argument('--native-root', type=Path, default=NATIVE)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--encoder', choices=PAPER, default='vitl')
    parser.add_argument('--size', type=int, choices=(392, 518, 700), default=518)
    parser.add_argument('--fit', choices=('ols', 'median-mad'), default='ols',
                        help='Median-MAD is an uncapped alignment diagnostic')
    parser.add_argument('--flip', action='store_true', help='Evaluate both single pass and flip mean')
    parser.add_argument('--cache-predictions', action='store_true')
    parser.add_argument('--device', default='cuda', help='CUDA device, for example cuda:0')
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--limit', type=int, choices=range(1, 455), metavar='1..454')
    sampling.add_argument('--images-per-scene', type=int, choices=range(1, 77), metavar='1..76',
                          help='Run a diagnostic sampled evenly by name within each scene')
    run(parser.parse_args())
