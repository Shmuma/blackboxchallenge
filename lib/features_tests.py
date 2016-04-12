import unittest
import numpy as np

import features

class TestFeatures(unittest.TestCase):
    def test_final_size(self):
        self.assertEqual(58, features.transformed_size())

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

    def test_result(self):
        f = features.transform(np.zeros((features.ORIGIN_N_FEATURES, )))
        self.assertEqual(features.transformed_size(), len(f))
