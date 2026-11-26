"""Verify local DIODE RGB/depth/mask bytes against an authenticated official archive.

First authenticate the downloaded validation archive by its published MD5, then
compare each member's SHA256. Later --reference-report rechecks local bytes
without decompressing the archive. No extraction or dataset changes occur.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

OFFICIAL_URL = 'https://diode-dataset.s3.amazonaws.com/val.tar.gz'
OFFICIAL_MD5 = '5c895d09201b88973c8fe4552a67dd85'
EXPECTED_FILES = 771 * 3
ROOT = Path(__file__).resolve().parent


def hash_stream(stream, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        digest.update(block)
    return digest.hexdigest()


def safe_local_path(val_dir, relative):
    rel = PurePosixPath(relative)
    root = Path(val_dir).resolve()
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Unsafe dataset path: {relative}')
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path outside dataset: {relative}')
    return path


def verify_reference_report(val_dir, report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('source') != OFFICIAL_URL or report.get('archive_md5_actual') != OFFICIAL_MD5
            or report.get('archive_md5_expected') != OFFICIAL_MD5
            or not report.get('all_files_identical')):
        raise ValueError('Reference report does not describe an authenticated official DIODE archive')
    rows = report['files']
    names = [r['path'] for r in rows]
    if len(names) != EXPECTED_FILES or len(set(names)) != EXPECTED_FILES:
        raise ValueError('Expected 2313 unique official RGB/depth/mask files')
    for row in rows:
        path = safe_local_path(val_dir, row['path'])
        with path.open('rb') as stream:
            if hash_stream(stream) != row['official_sha256']:
                raise ValueError(f'DIODE content differs from official data: {row["path"]}')
    local = {p.relative_to(val_dir).as_posix() for p in Path(val_dir).rglob('*')
             if p.is_file() and p.suffix in ('.png', '.npy')}
    if local != set(names):
        raise ValueError('Dataset contains missing or extra RGB/depth/mask files')
    with report_path.open('rb') as stream:
        report_hash = hash_stream(stream)
    return {'verified': True, 'source': OFFICIAL_URL, 'archive_md5': OFFICIAL_MD5,
            'files_checked': len(rows), 'reference_report_sha256': report_hash}


def audit_archive(archive, val_dir):
    archive, val_dir = Path(archive), Path(val_dir)
    with archive.open('rb') as stream:
        md5 = hash_stream(stream, 'md5')
    if md5 != OFFICIAL_MD5:
        raise ValueError(f'Archive MD5 mismatch: {md5}')
    rows, missing, different = [], [], []
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = PurePosixPath(member.name)
            if rel.parts[0] != 'val' or rel.suffix not in ('.png', '.npy'):
                continue
            relative = PurePosixPath(*rel.parts[1:]).as_posix()
            local = safe_local_path(val_dir, relative)
            with tar.extractfile(member) as stream:
                official_hash = hash_stream(stream)
            local_hash = None
            if not local.is_file():
                missing.append(relative)
            else:
                with local.open('rb') as stream:
                    local_hash = hash_stream(stream)
                if local_hash != official_hash:
                    different.append(relative)
            rows.append({'path': relative, 'official_sha256': official_hash,
                         'local_sha256': local_hash, 'identical': local_hash == official_hash})
    names = {r['path'] for r in rows}
    local_names = {p.relative_to(val_dir).as_posix() for p in val_dir.rglob('*')
                   if p.is_file() and p.suffix in ('.png', '.npy')}
    extra = sorted(local_names - names)
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'source': OFFICIAL_URL,
            'archive_md5_expected': OFFICIAL_MD5, 'archive_md5_actual': md5,
            'archive_size': archive.stat().st_size, 'compared_files': len(rows),
            'missing': missing, 'different': different, 'extra': extra,
            'all_files_identical': not missing and not different and not extra
                and len(rows) == len(names) == EXPECTED_FILES, 'files': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path)
    group.add_argument('--reference-report', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/diode-relative/audit/official-archive-verification.json')
    args = parser.parse_args()
    val_dir = (args.data_dir/'diode/val').resolve()
    if args.archive:
        result = audit_archive(args.archive, val_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))
        if not result['all_files_identical']:
            raise SystemExit(1)
    else:
        print(json.dumps(verify_reference_report(val_dir, args.reference_report), indent=2))
