# ----------------------------------------------------------------------------
# tests/sketchers/test_cw_sketch.py
import unittest
from synrfp.sketchers.cw_sketch import CWSketch
import numpy as np


class TestCWSketch(unittest.TestCase):
    def test_instantiation_and_invalid_m(self):
        # invalid m is always rejected
        with self.assertRaises(ValueError):
            CWSketch(m=0)
        # valid instantiation
        cw = CWSketch(m=8, seed=3)
        # check internal attributes populated
        self.assertEqual(cw._m, 8)
        self.assertEqual(cw._seed, 3)

    def test_build_empty(self):
        cw = CWSketch(m=8, seed=1)
        arr = cw.build({}, {})
        self.assertIsInstance(arr, np.ndarray)
        # fallback and datasketch path both return uint64 hashvalues
        self.assertEqual(arr.dtype, np.uint64)
        self.assertEqual(arr.size, 8)
        self.assertTrue((arr == 0).all())

    def test_build_weighted_variations(self):
        cw = CWSketch(m=4, seed=1)
        pos = {1: 2, 3: 1}
        neg = {2: 1}
        arr1 = cw.build(pos, neg)
        arr2 = cw.build(pos, neg)
        # deterministic
        np.testing.assert_array_equal(arr1, arr2)
        # ensure output length
        self.assertEqual(len(arr1), 4)
        # swapping pos and neg should typically change the signature
        arr3 = cw.build(neg, pos)
        self.assertFalse(np.array_equal(arr1, arr3))


if __name__ == "__main__":
    unittest.main()
