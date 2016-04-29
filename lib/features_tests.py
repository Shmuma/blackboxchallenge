import unittest
import numpy as np

import features


class TestFeatures(unittest.TestCase):
    def test_final_size(self):
        self.assertEqual(4510, features.transformed_size())

    def test_to_dense(self):
        idx = np.array([0, 3])
        val = np.array([1, 2])
        v = features.to_dense((idx, val))
        res = np.zeros((features.RESULT_N_FEATURES,))
        res[0] = 1
        res[3] = 2
        d = np.subtract(v, res)
        self.assertEqual(len(d), features.RESULT_N_FEATURES)
        self.assertAlmostEqual(0.0, d.sum())

    def test_feature_35(self):
        v = -1.1
        sparse = features._transform_35(v)
        self.assertEqual(sparse[0], 0)
        self.assertAlmostEqual(sparse[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_35(sparse))

        v = 0.0
        r = features._transform_35(v)
        self.assertEqual(r[0], 11)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_35(r))

        # bad range is an exception
        self.assertRaises(AssertionError, lambda: features._transform_35(-1.2))
        self.assertRaises(AssertionError, lambda: features._transform_35(1.2))
        self.assertRaises(AssertionError, lambda: features._transform_35(-20))

        v = 0.69999999
        r = features._transform_35(v)
        self.assertEqual(r[0], 18)
        self.assertAlmostEqual(v, features._reverse_35(r))

    def test_feature_00(self):
        v = -0.7275277972002714
        idx, val = features._transform_stripes_func(0)(v)
        self.assertEqual(idx, 17)
        self.assertAlmostEqual(val, 1.0)
        self.assertAlmostEqual(v, features._reverse_00((idx, val)), places=5)
        v = -1.5
        idx, val = features._transform_stripes_func(0)(v)
        self.assertAlmostEqual(val, -1.5)
        self.assertAlmostEqual(v, features._reverse_00((idx, val)), places=5)

    def test_feature_01(self):
        r = features._split_bound_func(0.0)(-1.0)
        self.assertEqual(r[0], 0)
        self.assertAlmostEqual(r[1], -1.0)
        self.assertAlmostEqual(-1.0, features._unsplit_bound(r))
        r = features._split_bound_func(0.0)(1.0)
        self.assertEqual(r[0], 1)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(1.0, features._unsplit_bound(r))

    def test_feature_05(self):
        v = -0.65676
        r = features._transform_stripes_func(5)(v)
        self.assertEqual(r[0], 0)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_stripe_func(5)(r), places=5)

        v = 0.5661479235
        r = features._transform_stripes_func(5)(v)  # 5'th entry
        self.assertEqual(r[0], 5)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_stripe_func(5)(r), places=5)

    def test_feature_09(self):
        v = -.7955922120711471674   # 7'th entry in first stripe
        r = features._transform_stripes_func(9)(v)
        self.assertEqual(r[0], 7)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_stripe_func(9)(r), places=5)

        v = -.7140044569994015538   # 60'th entry in first stripe
        r = features._transform_stripes_func(9)(v)
        self.assertEqual(r[0], 59)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_stripe_func(9)(r), places=5)

        v = 1.293757915496888       # single stripe value
        r = features._transform_stripes_func(9)(v)
        self.assertEqual(r[0], 60)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(v, features._reverse_stripe_func(9)(r), places=5)

    def test_result(self):
        idx, vals = features.transform(np.zeros((features.ORIGIN_N_FEATURES, )))
        self.assertEqual(features.ORIGIN_N_FEATURES, idx.shape[0])
        self.assertEqual(features.ORIGIN_N_FEATURES, vals.shape[0])

    def test_transform_striped(self):
        opts = {
            'delta': 0.1,
            'start': -0.5,
            'stop': 0.5
        }

        index, total = features._transform_striped(0.5, **opts)
        self.assertIsNotNone(index)
        self.assertEqual(total, 11)
        self.assertEqual(index, 10)

        index, total = features._transform_striped(-1, **opts)
        self.assertIsNone(index)
