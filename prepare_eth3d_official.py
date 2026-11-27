"""Prepare paired ETH3D DSLR RGB and official rendered depth at stride 3."""
import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np

from thuc_nghiem_1_eth3d_relative_depth import CACHE, discover, sha256

BASE_URL = 'https://www.eth3d.net/data'
ARCHIVES = CACHE / 'official-downloads'
OUTPUT = CACHE / 'official-stride3'
NATIVE_SHAPE = (4032, 6048)


def download(scene, product):
    name = f'{scene}_dslr_{product}.7z'
    path = ARCHIVES / name
    if not path.exists():
        temporary = path.with_suffix('.partial')
        subprocess.run(['curl.exe', '-fLsS', '--retry', '2', '--continue-at', '-',
                        '--output', str(temporary), f'{BASE_URL}/{name}'], check=True)
        temporary.replace(path)
    return path


def tar_members(archive, prefix):
    listing = subprocess.check_output(['tar.exe', '-tvf', str(archive)], text=True)
    members = {}
    for line in listing.splitlines():
        name = line.rsplit(maxsplit=1)[-1]
        if name.startswith(prefix) and name.lower().endswith('.jpg'):
            members[Path(name).stem] = (name, int(line.split()[4]))
    return members


def prepare_scene(scene, expected):
    folder = OUTPUT / scene
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.is_file() and all((folder / f'{key}.{suffix}').is_file()
                                  for key in expected for suffix in ('jpg', 'npz')):
        print(f'{scene}: already prepared', flush=True)
        return
    manifest.unlink(missing_ok=True)

    depth_archive = download(scene, 'depth')
    rgb_archive = download(scene, 'jpg')
    depth_members = tar_members(depth_archive, f'{scene}/ground_truth_depth/dslr_images/')
    rgb_members = tar_members(rgb_archive, f'{scene}/images/dslr_images/')
    if set(depth_members) != expected or set(rgb_members) != expected:
        raise ValueError(f'Official RGB/depth IDs differ for {scene}')
    if list(depth_members) != list(rgb_members):
        raise ValueError(f'Official archives list their members in different order for {scene}')
    h, w = NATIVE_SHAPE
    with (
        subprocess.Popen(['tar.exe', '-xOf', str(depth_archive),
                          *(name for name, _ in depth_members.values())],
                         stdout=subprocess.PIPE) as depth_stream,
        subprocess.Popen(['tar.exe', '-xOf', str(rgb_archive),
                          *(name for name, _ in rgb_members.values())],
                         stdout=subprocess.PIPE) as rgb_stream,
    ):
        for key, (_, depth_size) in depth_members.items():
            if depth_size != h * w * 4:
                raise ValueError(f'Unexpected depth record length for {scene}/{key}')
            encoded_depth = depth_stream.stdout.read(depth_size)
            encoded_rgb = rgb_stream.stdout.read(rgb_members[key][1])
            if len(encoded_depth) != depth_size or len(encoded_rgb) != rgb_members[key][1]:
                raise ValueError(f'Incomplete archive record for {scene}/{key}')
            native = np.frombuffer(encoded_depth, dtype='<f4').reshape(h, w)
            # Pixel centers of 3x3 blocks match the RGB area-resize grid.
            depth = native[1::3, 1::3].copy()
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            rgb = cv2.imdecode(np.frombuffer(encoded_rgb, np.uint8),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if rgb is None or rgb.shape[:2] != NATIVE_SHAPE or not depth.any():
                raise ValueError(f'Invalid official RGB/depth pair for {scene}/{key}')
            rgb = cv2.resize(rgb, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(folder / f'{key}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise OSError(f'Could not save {scene}/{key}')
            np.savez_compressed(folder / f'{key}.npz', depth=depth)
        for stream in (depth_stream, rgb_stream):
            if stream.stdout.read(1) or stream.wait() != 0:
                raise ValueError(f'Unequal archive record lengths for {scene}')
    metadata = {
        'scene': scene, 'images': sorted(expected),
        'rgb_archive_sha256': sha256(rgb_archive), 'depth_archive_sha256': sha256(depth_archive),
        'rgb_archive': f'{BASE_URL}/{rgb_archive.name}',
        'depth_archive': f'{BASE_URL}/{depth_archive.name}',
        'sampling': 'RGB INTER_AREA and JPEG quality 96; finite positive native depth [1::3,1::3]',
    }
    temporary = manifest.with_suffix('.partial')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest)
    # These are staging files created by this script. Keep the separately fetched audit pair.
    for archive in (depth_archive, rgb_archive):
        archive.unlink()
    print(f'{scene}: prepared {len(expected)} pairs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', help='Prepare this scene only, repeatable')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    with ZipFile(CACHE / 'ETH3D.zip') as archive:
        keys = discover(archive, CACHE / 'unused')
    scenes = sorted({key.split('/')[0] for key in keys})
    if args.scene:
        if not set(args.scene).issubset(scenes):
            parser.error('--scene must be one of: ' + ', '.join(scenes))
        scenes = args.scene
    for scene in scenes:
        expected = {key.split('/')[1] for key in keys if key.startswith(f'{scene}/')}
        prepare_scene(scene, expected)


if __name__ == '__main__':
    main()

