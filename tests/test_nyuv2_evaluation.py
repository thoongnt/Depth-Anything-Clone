import unittest
import numpy as np
from evaluate_nyuv2 import align_disparity, score_disparity, nyu_mask


class NYUv2EvaluationTests(unittest.TestCase):
    def test_affine_disparity_prediction_is_perfect(self):
        depth = np.linspace(0.5, 9.5, 480 * 640).reshape(480, 640)
        prediction = (1.0 / depth - 0.2) / 3.0
        result = score_disparity(prediction, depth)
        self.assertLess(result['abs_rel'], 1e-12)
        self.assertEqual(result['delta1'], 1.0)
        self.assertAlmostEqual(result['scale'], 3.0)
        self.assertAlmostEqual(result['shift'], 0.2)

    def test_invalid_and_border_pixels_do_not_change_alignment(self):
        depth = np.full((480, 640), 2.0)
        depth[100:300, 100:300] = 4.0
        prediction = 1.0 / depth
        depth[50, 50] = np.nan
        depth[51, 50] = 0.0
        depth[52, 50] = 10.0
        prediction[0:45] = np.nan
        prediction[50:53, 50] = np.nan
        result = score_disparity(prediction, depth, eigen_crop=True)
        self.assertLess(result['abs_rel'], 1e-12)
        self.assertEqual(result['valid_pixels'], 426 * 560 - 3)

    def test_midas_mask_keeps_valid_points_outside_eigen_crop(self):
        depth = np.zeros((480, 640))
        depth[10, 10] = 0.05
        self.assertTrue(nyu_mask(depth)[10, 10])
        self.assertFalse(nyu_mask(depth, eigen_crop=True)[10, 10])

    def test_constant_prediction_fails_instead_of_silent_fallback(self):
        a = np.ones((2, 2))
        with self.assertRaises(ValueError):
            align_disparity(a, a, np.ones((2, 2), dtype=bool))

    def test_empty_mask_fails_instead_of_zero_error(self):
        with self.assertRaises(ValueError):
            score_disparity(np.ones((480, 640)), np.zeros((480, 640)))

    def test_absrel_is_scored_in_disparity_not_depth(self):
        depth = np.tile(np.linspace(0.5, 5.0, 640), (480, 1))
        prediction = 1.0 / depth + 0.02 * np.sin(np.arange(640))[None, :]
        result = score_disparity(prediction, depth)
        mask = nyu_mask(depth)
        aligned, _, _ = align_disparity(prediction, 1.0 / depth, mask)
        expected = np.mean(np.abs(aligned[mask] - 1.0 / depth[mask]) / (1.0 / depth[mask]))
        self.assertAlmostEqual(result['abs_rel'], expected)
        self.assertGreater(abs(result['abs_rel'] - result['depth_abs_rel_diagnostic']), 1e-5)


if __name__ == '__main__':
    unittest.main()
