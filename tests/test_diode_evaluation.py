import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evaluate_diode_relative import score_diode, aggregate, discover_samples


class DIODEEvaluationTests(unittest.TestCase):
    def test_perfect_affine_disparity_across_far_depths(self):
        depth = np.array([[0.3, 1, 20], [50, 100, 299]], dtype=np.float64)
        prediction = (1 / depth - 0.2) / 3
        row = score_diode(prediction, depth, np.ones_like(depth))
        self.assertLess(row['abs_rel'], 1e-12)
        self.assertEqual(row['delta1'], 1)
        self.assertEqual(row['valid_pixels'], 6)
        self.assertAlmostEqual(row['scale'], 3)
        self.assertAlmostEqual(row['shift'], 0.2)

    def test_supplied_mask_excludes_positive_invalid_depth(self):
        depth = np.array([[1., 2., 4., 99.]])
        pred = 1 / depth
        pred[0, 3] = np.nan
        row = score_diode(pred, depth, np.array([[1, 1, 1, 0]]))
        self.assertLess(row['abs_rel'], 1e-12)
        self.assertEqual(row['valid_pixels'], 3)

    def test_nonfinite_and_nonpositive_gt_are_excluded(self):
        depth = np.array([[1., 2., 4., 0., -1., np.nan, np.inf]])
        prediction = np.array([[1., .5, .25, np.nan, np.nan, np.nan, np.nan]])
        row = score_diode(prediction, depth, np.ones_like(depth))
        self.assertEqual(row['valid_pixels'], 3)
        self.assertLess(row['abs_rel'], 1e-12)

    def test_negative_aligned_prediction_fails_delta_without_dropping_pixel(self):
        q = np.array([[1., 1., 20.]])
        row = score_diode(np.array([[1., 2., 3.]]), 1/q, np.ones_like(q))
        self.assertEqual(row['nonpositive_predictions'], 1)
        self.assertEqual(row['valid_pixels'], 3)
        self.assertAlmostEqual(row['delta1'], 1/3)

    def test_empty_mask_constant_prediction_and_bad_shapes_fail(self):
        depth = np.array([[1., 2., 3.]])
        for pred, mask in [(1/depth, np.zeros_like(depth)),
                           (np.ones_like(depth), np.ones_like(depth)),
                           (np.ones((3, 1)), np.ones_like(depth))]:
            with self.subTest(shape=pred.shape, mask=mask.tolist()):
                with self.assertRaises(ValueError):
                    score_diode(pred, depth, mask)

    def test_nonfinite_prediction_on_valid_pixel_fails(self):
        with self.assertRaises(ValueError):
            score_diode(np.array([[1., np.nan, 3.]]),
                        np.array([[1., 2., 3.]]), np.ones((1, 3)))

    def test_aggregation_weights_images_not_domains_or_pixels(self):
        rows = [{'abs_rel': a, 'delta1': 1-a, 'valid_pixels': n,
                 'nonpositive_predictions': 0} for a, n in [(0, 1000), (0, 1000), (.6, 1)]]
        self.assertAlmostEqual(aggregate(rows)['metrics']['abs_rel'], .2)
        self.assertEqual(aggregate(rows)['valid_pixels'], 2001)
        with self.assertRaises(ValueError):
            aggregate([])

    def test_missing_dataset_and_wrong_subset_fail(self):
        with patch.object(Path, 'rglob', return_value=iter([])):
            with self.assertRaisesRegex(ValueError, 'expected 325'):
                discover_samples('missing-data')
        with self.assertRaises(ValueError):
            discover_samples('missing-data', 'indoor')

    def test_missing_pair_is_not_silently_skipped(self):
        with patch('evaluate_diode_relative.COUNTS', {'indoors': 1}), \
             patch.object(Path, 'rglob', return_value=[Path('val/indoors/sample.png')]), \
             patch.object(Path, 'is_file', side_effect=[True, False]):
            with self.assertRaises(FileNotFoundError):
                discover_samples('val')


if __name__ == '__main__':
    unittest.main()
