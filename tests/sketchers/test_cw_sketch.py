# ----------------------------------------------------------------------------
# tests/sketchers/test_cw_sketch.py
import unittest
from synrfp.sketchers.cw_sketch import CWSketch, _HAVE_DATASKETCH
import numpy as np


class TestCWSketch(unittest.TestCase):
    def test_instantiation_and_invalid_m(self):
        if not _HAVE_DATASKETCH:
            with self.assertRaises(RuntimeError):
                CWSketch()
        else:
            with self.assertRaises(ValueError):
                CWSketch(m=0)
            cw = CWSketch(m=8, seed=3)
            self.assertEqual(cw.m, 8)

    @unittest.skipUnless(_HAVE_DATASKETCH, "requires datasketch")
    def test_build_empty(self):
        cw = CWSketch(m=8, seed=1)
        arr = cw.build({}, {})
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.uint64)
        self.assertTrue((arr == 0).all())

    @unittest.skipUnless(_HAVE_DATASKETCH, "requires datasketch")
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
        # swapping pos and neg changes result
        arr3 = cw.build(neg, pos)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(arr1, arr3)
