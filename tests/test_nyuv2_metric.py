import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch
from evaluate_nyuv2_metric import score_metric_depth, metric_mask, infer_metric, prepare_bts_nyu_image, prepare_bts_nyu_depth


@contextmanager
def exported_fixture(suffix):
    path = Path(__file__).resolve().parents[1] / '.cache' / ('nyu-test-' + uuid.uuid4().hex + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield str(path)
    finally:
        path.unlink(missing_ok=True)


class NYUv2MetricTests(unittest.TestCase):
    def test_in_memory_image_matches_bts_disk_export(self):
        rgb = np.random.default_rng(7).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        bgr = rgb[:, :, ::-1]
        exported = np.zeros_like(rgb)
        exported[7:474, 7:632] = bgr[7:474, 7:632]
        with exported_fixture('.jpg') as path:
            self.assertTrue(cv2.imwrite(path, exported))
            with Image.open(path) as image:
                expected = np.asarray(image, dtype=np.float32) / 255.0
        np.testing.assert_array_equal(prepare_bts_nyu_image(rgb), expected)

    def test_depth_quantization_matches_bts_png_roundtrip(self):
        raw = np.full((480, 640), 2.1239, dtype=np.float64)
        raw[50, 50] = 0.0
        exported = (raw * 1000.0).astype(np.uint16)
        with exported_fixture('.png') as path:
            self.assertTrue(cv2.imwrite(path, exported))
            with Image.open(path) as image:
                expected = np.asarray(image, dtype=np.float32) / 1000.0
        np.testing.assert_array_equal(prepare_bts_nyu_depth(raw), expected)
        self.assertAlmostEqual(float(prepare_bts_nyu_depth(raw)[100, 100]), 2.123, places=6)

    def test_perfect_metric_depth(self):
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        result = score_metric_depth(depth, depth)
        for key in ('delta1', 'delta2', 'delta3'):
            self.assertEqual(result[key], 1.0)
        for key in ('abs_rel', 'rmse', 'log10'):
            self.assertEqual(result[key], 0.0)
        self.assertEqual(result['valid_pixels'], 426 * 560)

    def test_scale_error_is_not_removed(self):
        gt = np.full((480, 640), 2.0, dtype=np.float32)
        result = score_metric_depth(gt * 1.3, gt)
        self.assertAlmostEqual(result['abs_rel'], 0.3, places=5)
        self.assertAlmostEqual(result['rmse'], 0.6, places=5)
        self.assertAlmostEqual(result['log10'], np.log10(1.3), places=5)
        self.assertEqual(result['delta1'], 0.0)
        self.assertEqual(result['delta2'], 1.0)
        self.assertEqual(result['delta3'], 1.0)

    def test_crop_and_invalid_pixels(self):
        gt = np.full((480, 640), 2.0, dtype=np.float32)
        gt[50, 50] = 0
        gt[51, 50] = 10
        gt[52, 50] = np.nan
        self.assertEqual(metric_mask(gt).sum(), 426 * 560 - 3)
        self.assertFalse(metric_mask(gt)[10, 10])

    def test_empty_gt_or_bad_predictions_fail(self):
        gt = np.full((480, 640), 2.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            score_metric_depth(gt, np.zeros_like(gt))
        prediction = gt.copy()
        prediction[0, 0] = np.nan
        with self.assertRaises(ValueError):
            score_metric_depth(prediction, gt)

    def test_clipping_does_not_mutate_prediction(self):
        gt = np.full((480, 640), 2.0, dtype=np.float32)
        prediction = np.full_like(gt, 20.0)
        result = score_metric_depth(prediction, gt)
        self.assertTrue(np.all(prediction == 20))
        self.assertAlmostEqual(result['rmse'], 8.0)

    def test_flip_returns_predictions_in_original_orientation(self):
        class DummyModel:
            def __call__(self, image, **kwargs):
                return {'metric_depth': image[:, :1] + 1}
        image = torch.rand(1, 3, 480, 640)
        result = infer_metric(DummyModel(), image, flip=True)
        np.testing.assert_allclose(result, image[0, 0].numpy() + 1)


if __name__ == '__main__':
    unittest.main()
