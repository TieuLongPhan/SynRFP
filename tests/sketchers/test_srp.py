# tests/sketchers/test_srp.py
import unittest
import numpy as np
from synrfp.sketchers.srp_sketch import SRPSketch


class TestSRPSketch(unittest.TestCase):
    def test_invalid_m(self):
        with self.assertRaises(ValueError):
            SRPSketch(m=0)
        with self.assertRaises(ValueError):
            SRPSketch(m=-10)

    def test_repr_and_describe(self):
        srp = SRPSketch(m=64, seed=7, normalize=False, memory_safe_threshold=0)
        expected = "SRPSketch(m=64, seed=7, normalize=False, memory_safe_threshold=0)"
        self.assertEqual(repr(srp), expected)
        desc = srp.describe()
        self.assertIn("SRPSketch", desc)
        self.assertIn("sk.build", desc) or self.assertIn("SRPSketch", desc)

    def test_build_consistency_and_dtype(self):
        srp = SRPSketch(m=128, seed=1, normalize=True)
        pos = {1: 2, 3: 1, 5: 4}
        neg = {2: 1}
        sketch1 = srp.build(pos, neg)
        sketch2 = srp.build(pos, neg)
        # deterministic equality
        np.testing.assert_array_equal(sketch1, sketch2)
        # dtype and length
        self.assertEqual(sketch1.dtype, np.int8)
        self.assertEqual(sketch1.size, 128)
        # values should be +/-1
        unique_vals = set(np.unique(sketch1).tolist())
        self.assertTrue(unique_vals.issubset({-1, 1}))

    def test_approx_cosine_similarity(self):
        # identical inputs -> cosine ~ 1
        srp = SRPSketch(m=256, seed=42, normalize=True)
        pos_a = {1: 2, 3: 1, 7: 1}
        neg_a = {}
        a = srp.build(pos_a, neg_a)
        b = srp.build(pos_a, neg_a)
        cos_ab = SRPSketch.approx_cosine_from_sketch(a, b)
        self.assertAlmostEqual(cos_ab, 1.0, places=6)

        # different inputs -> cosine < 1 (and within [-1,1])
        pos_b = {10: 1}
        b2 = srp.build(pos_b, {})
        cos_a_b2 = SRPSketch.approx_cosine_from_sketch(a, b2)
        self.assertGreaterEqual(cos_a_b2, -1.0)
        self.assertLessEqual(cos_a_b2, 1.0)
        # should be meaningfully smaller than identical-case (not exactly 1)
        self.assertLess(cos_a_b2, 0.9)

    def test_rowwise_matches_full_matrix(self):
        # ensure row-wise (memory_safe) path yields same result as full-matrix path
        pos = {i: 1 for i in range(20)}
        neg = {}
        # full-matrix default (threshold large enough)
        srp_full = SRPSketch(
            m=64, seed=123, normalize=False, memory_safe_threshold=10_000_000
        )
        # force row-wise by setting threshold to 0 (always rowwise)
        srp_row = SRPSketch(m=64, seed=123, normalize=False, memory_safe_threshold=0)

        sig_full = srp_full.build(pos, neg)
        sig_row = srp_row.build(pos, neg)
        np.testing.assert_array_equal(sig_full, sig_row)


if __name__ == "__main__":
    unittest.main()
