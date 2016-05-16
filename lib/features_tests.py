from unittest import TestCase
import numpy as np
import array

import features

class TestTransform(TestCase):
    data = [-0.73334587, -0.93910491, -0.94270045, -1.36341822, -1.70008028, -0.65676075,
            -0.66000223, -0.61031222, -0.61547941, -0.73753935, -0.74450946, -2.11130285,
            -0.040695058, -1.24927127,  0.90754294, -0.21203995,  0.3009333 , -0.49948287,
            -0.646667,   -0.31782657,  0.25072268,  1.20941901,  0.24673782, -0.2735582 ,
            0.78459948,   0.47483093,  0.15551165, -0.09912075, -0.3076221 , -0.46290749,
            0.112813171,  -0.4509145,  1.5137006,   -0.99996543,  0.87362069,  0.
            ]

    def test_binary(self):
        s = array.array('f', self.data).tostring()
        a_ints = array.array('I')
        a_ints.fromstring(s)
        ints = a_ints.tolist()

        expect = [
            0xBF3BBC8E, 0xBF70692E,
            0xBF7154D1, 0xBFAE847D, 0xBFD99C3B, 0xBF282179,
            0xBF28F5E8, 0xBF1C3D6C, 0xBF1D900F, 0xBF3CCF61,
            0xBF3E982C, 0xC0071F96, 0xBD26AFDC, 0xBF9FE81F,
            0x3F6854BC, 0xBE592100, 0x3E9A13EE, 0xBEFFBC38,
            0xBF258BF8, 0xBEA2BA2A, 0x3E805EB9, 0x3F9ACE3E,
            0x3E7CA8D7, 0xBE8C0FD2, 0x3F48DB83, 0x3EF31D0A,
            0x3E1F3E72, 0xBDCAFFD2, 0xBE9D80A5, 0xBEED0236,
            0x3DE70A98, 0xBEE6DE44, 0x3FC1C0F1, 0xBF7FFDBC,
            0x3F5FA59B, 0x0
        ]

        self.assertEqual(ints, expect)

    def test_transform(self):
        bins = features.transform(self.data)

        self.assertEqual(len(bins), features.RESULT_N_FEATURES)

        first = bins[0:32]
        # 0xBF3BBC8E
        first_valid = [1., 0., 1., 1., 1., 1., 1., 1.,
                       0., 0., 1., 1., 1., 0., 1., 1.,
                       1., 0., 1., 1., 1., 1., 0., 0.,
                       1., 0., 0., 0., 1., 1., 1., 0.]
        self.assertTrue(np.allclose(first, first_valid))

        self.assertAlmostEqual(bins[-1], 0.0)
