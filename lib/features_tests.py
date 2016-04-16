import unittest
import numpy as np

import features

class TestFeatures(unittest.TestCase):
    def test_final_size(self):
        self.assertEqual(363, features.transformed_size())

    def test_feature_35(self):
        r = features._transform_35(-1.1)
        self.assertEqual(23, len(r))
        self.assertAlmostEqual(1, sum(r))
        self.assertAlmostEqual(1.0, r[0])
        self.assertAlmostEqual(0, r[1])

        r = features._transform_35(0.0)
        self.assertEqual(23, len(r))
        self.assertAlmostEqual(1, sum(r))
        self.assertAlmostEqual(0, r[0])
        self.assertAlmostEqual(1, r[11])

        # bad range is an exception
        self.assertRaises(AssertionError, lambda: features._transform_35(-1.2))
        self.assertRaises(AssertionError, lambda: features._transform_35(1.2))
        self.assertRaises(AssertionError, lambda: features._transform_35(-20))

    def test_feature_00(self):
        v = -0.7275277972002714
        r = features._transform_00(v)
        self.assertEqual(len(r), 181)
        self.assertAlmostEqual(r[18], 1.0)
        self.assertAlmostEqual(v, features._reverse_00(r), places=5)
        v = -1.5
        r = features._transform_00(v)
        self.assertAlmostEqual(r[0], -1.5)
        self.assertAlmostEqual(v, features._reverse_00(r), places=5)

    def test_feature_01(self):
        r = features._split_bound_func(0.0)(-1.0)
        self.assertAlmostEqual(r[0], -1.0)
        self.assertAlmostEqual(r[1], 0.0)
        self.assertAlmostEqual(-1.0, features._unsplit_bound(r))
        r = features._split_bound_func(0.0)(1.0)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(1.0, features._unsplit_bound(r))

    def test_feature_05(self):
        r = features._transform_05(-0.65676)
        self.assertEqual(len(r), 122)
        self.assertAlmostEqual(r[0], 1.0)
        self.assertAlmostEqual(sum(r[1:]), 0.0)
        r = features._transform_05(0.5661479235)  # 5'th entry
        self.assertAlmostEqual(r[5], 1.0)
        self.assertAlmostEqual(sum(r[:5]), 0.0)
        self.assertAlmostEqual(sum(r[6:]), 0.0)

    def test_result(self):
        f = features.transform(np.zeros((features.ORIGIN_N_FEATURES, )))
        self.assertEqual(features.transformed_size(), len(f))

    def test_transform_striped(self):
        opts = {
            'delta': 0.1,
            'start': -0.5,
            'stop': 0.5
        }

        filled, r = features._transform_striped(0.5, **opts)
        self.assertTrue(filled)
        self.assertEqual(len(r), 11)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[-1], 1.0)

        filled, r = features._transform_striped(-1, **opts)
        self.assertFalse(filled)
        self.assertAlmostEqual(0.0, r.sum())