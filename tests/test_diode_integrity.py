import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from audit_diode import hash_stream, safe_local_path, verify_reference_report, OFFICIAL_URL, OFFICIAL_MD5


class DIODEIntegrityTests(unittest.TestCase):
    def test_hash_stream_covers_contents(self):
        self.assertEqual(hash_stream(io.BytesIO(b'official sample')),
                         hashlib.sha256(b'official sample').hexdigest())

    def test_path_cannot_escape_dataset(self):
        root = Path.cwd() / 'data/diode/val'
        for name in ['../outside', '/absolute', 'indoors/../../escape', 'C:/Windows/file']:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    safe_local_path(root, name)
        self.assertEqual(safe_local_path(root, 'indoors/image.png'), root/'indoors/image.png')

    def report(self):
        return {'source': OFFICIAL_URL, 'archive_md5_actual': OFFICIAL_MD5,
                'archive_md5_expected': OFFICIAL_MD5, 'all_files_identical': True,
                'files': [{'path': 'indoors/image.png',
                           'official_sha256': hashlib.sha256(b'correct').hexdigest()}]}

    def test_tampered_sample_fails_before_inference(self):
        report = self.report()
        with patch('audit_diode.EXPECTED_FILES', 1), \
             patch.object(Path, 'read_text', return_value=json.dumps(report)), \
             patch.object(Path, 'open', return_value=io.BytesIO(b'changed')):
            with self.assertRaisesRegex(ValueError, 'content differs'):
                verify_reference_report(Path.cwd()/'data/diode/val', 'report.json')

    def test_wrong_archive_identity_is_rejected(self):
        report = self.report(); report['archive_md5_actual'] = 'wrong'
        with patch.object(Path, 'read_text', return_value=json.dumps(report)):
            with self.assertRaisesRegex(ValueError, 'authenticated'):
                verify_reference_report('data/diode/val', 'report.json')

    def test_missing_or_duplicate_manifest_rows_are_rejected(self):
        report = self.report()
        for rows in [report['files'], report['files'] * 2313]:
            report['files'] = rows
            with patch.object(Path, 'read_text', return_value=json.dumps(report)):
                with self.assertRaisesRegex(ValueError, '2313 unique'):
                    verify_reference_report('data/diode/val', 'report.json')


if __name__ == '__main__':
    unittest.main()
