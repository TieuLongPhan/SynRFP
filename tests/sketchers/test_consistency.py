# tests/sketchers/test_consistency.py
"""Cross-reaction consistency of the weighted sketches and bit encodings."""

import unittest

import numpy as np

from synrfp.sketchers.cw_sketch import CWSketch
from synrfp.sketchers.srp_sketch import SRPSketch
from synrfp.utils import signature_to_bits


def weighted_jaccard(a, b):
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den else 0.0


class TestCWSConsistency(unittest.TestCase):
    def setUp(self):
        self.cw = CWSketch(m=2048, seed=7)

    def agreement(self, a, b):
        return float(np.mean(self.cw.build(a, {}) == self.cw.build(b, {})))

    def test_disjoint_supports_never_agree(self):
        self.assertEqual(self.agreement({10: 1, 20: 1}, {50: 1, 60: 1}), 0.0)

    def test_identical_inputs_always_agree(self):
        self.assertEqual(self.agreement({10: 3, 20: 1}, {10: 3, 20: 1}), 1.0)

    def test_estimates_weighted_jaccard(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            ks = rng.choice(10**12, 40, replace=False).tolist()
            a = {k: int(rng.integers(1, 5)) for k in ks[:30]}
            b = {k: int(rng.integers(1, 5)) for k in ks[10:]}
            self.assertAlmostEqual(
                self.agreement(a, b), weighted_jaccard(a, b), delta=0.05
            )

    def test_count_magnitude_is_retained(self):
        # {x:1} vs {x:2} has weighted Jaccard 0.5, not 1
        self.assertAlmostEqual(
            self.agreement({10: 1, 20: 1}, {10: 2, 20: 2}), 0.5, delta=0.05
        )

    def test_sign_lifting_separates_production_and_consumption(self):
        pos = self.cw.build({10: 1}, {})
        neg = self.cw.build({}, {10: 1})
        self.assertEqual(float(np.mean(pos == neg)), 0.0)

    def test_does_not_depend_on_other_tokens_indexing(self):
        # adding a token with a smaller id must not change which samples
        # the shared tokens win against each other
        a = self.cw.build({100: 1, 200: 1}, {})
        b = self.cw.build({1: 1, 100: 1, 200: 1}, {})
        self.assertAlmostEqual(float(np.mean(a == b)), 2 / 3, delta=0.05)


class TestSRPConsistency(unittest.TestCase):
    def test_estimates_cosine(self):
        srp = SRPSketch(m=8192, seed=3)
        a, b = {1: 1, 2: 2}, {1: 1, 2: 1, 4: 2}
        va, vb = np.array([1, 2, 0.0]), np.array([1, 1, 2.0])
        cos = va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)
        agree = float(np.mean(srp.build(a, {}) == srp.build(b, {})))
        self.assertAlmostEqual(np.cos(np.pi * (1 - agree)), cos, delta=0.05)

    def test_consistent_across_supports(self):
        srp = SRPSketch(m=512, seed=3)
        a = srp.build({1: 1}, {})
        b = srp.build({1: 1, 999: 0}, {})  # zero-weight token is ignored
        np.testing.assert_array_equal(a, b)


class TestSignatureToBits(unittest.TestCase):
    def test_bbit_sets_one_bit_per_sample(self):
        sig = np.arange(256, dtype=np.uint64) * 7919
        bits = np.array(signature_to_bits(sig, 1024))
        self.assertEqual(bits.sum(), 256)
        self.assertTrue((bits.reshape(256, 4).sum(axis=1) == 1).all())

    def test_bbit_identical_signatures_identical_bits(self):
        sig = np.arange(128, dtype=np.uint64)
        self.assertEqual(signature_to_bits(sig, 512), signature_to_bits(sig, 512))

    def test_bbit_rejects_incompatible_length(self):
        with self.assertRaises(ValueError):
            signature_to_bits(np.arange(100), 1024)

    def test_legacy_mod(self):
        bits = signature_to_bits([1, 3], 4, method="mod")
        self.assertEqual(bits, [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
