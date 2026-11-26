import unittest
import numpy as np
from research_diode_variants import alternative_fit, evaluation_mask, score_vectors, summarize


class DIODEVariantTests(unittest.TestCase):
    def test_all_fit_methods_recover_affine_relation(self):
        x=np.linspace(.1,3.,101);y=2*x+.4
        for method in ['median-mad','relative-ls','huber']:
            with self.subTest(method=method):
                scale,shift=alternative_fit(x,y,method)
                self.assertAlmostEqual(scale,2.)
                self.assertAlmostEqual(shift,.4)

    def test_relative_ls_matches_direct_weighted_design(self):
        x=np.array([.1,.3,1.,2.,4.]);y=np.array([.3,.4,1.5,2.3,7.])
        expected=np.linalg.lstsq(np.column_stack([x/y,1/y]),np.ones(5),rcond=None)[0]
        self.assertTrue(np.allclose(alternative_fit(x,y,'relative-ls'),expected))

    def test_huber_resists_one_large_target_outlier(self):
        x=np.linspace(0,1,101);clean=2*x+1;noisy=clean.copy();noisy[50]=50
        scale,shift=alternative_fit(x,noisy,'huber')
        self.assertLess(np.mean(np.abs(scale*x+shift-clean)),.01)

    def test_depth_filter_keeps_supplied_mask_and_excludes_boundaries(self):
        depth=np.array([[.2,.5,1.,9.,10.,20.,np.nan]])
        provided=np.array([[1,1,0,1,1,1,1]])
        mask=evaluation_mask(depth,provided,{'min_depth':.5,'max_depth':10})
        self.assertEqual(mask.tolist(),[[False,False,False,True,False,False,False]])

    def test_nonpositive_predictions_fail_delta(self):
        result=score_vectors(np.array([-1.,1.]),np.array([1.,1.]))
        self.assertEqual(result['delta1'],.5)
        self.assertEqual(result['abs_rel'],1.)

    def test_empty_masks_cannot_count_as_full_or_matching(self):
        rows=[{'status':'ok','image':'a','valid_pixels':10,'baseline_valid_pixels':20,
               'abs_rel':.066,'delta1':.952},
              {'status':'empty-mask','image':'b','valid_pixels':0,'baseline_valid_pixels':20}]
        result=summarize(rows,2)
        self.assertFalse(result['full_subset'])
        self.assertFalse(result['matches_both_paper_rounding'])
        self.assertEqual(result['unscorable_images'],['b'])
        self.assertEqual(result['native_valid_pixel_retention'],.25)

    def test_pixel_and_image_aggregation_are_distinguished(self):
        rows=[{'status':'ok','image':'a','valid_pixels':9,'baseline_valid_pixels':9,'abs_rel':0.,'delta1':1.},
              {'status':'ok','image':'b','valid_pixels':1,'baseline_valid_pixels':1,'abs_rel':1.,'delta1':0.}]
        self.assertAlmostEqual(summarize(rows,2)['abs_rel'],.5)
        self.assertAlmostEqual(summarize(rows,2,'pixel-mean')['abs_rel'],.1)


if __name__=='__main__':unittest.main()
