# ----------------------------------------------------------------------------
# tests/sketchers/test_parity_fold.py
import unittest
from synrfp.sketchers.parity_fold import ParityFold


class TestParityFold(unittest.TestCase):
    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            ParityFold(bits=0)
        with self.assertRaises(ValueError):
            ParityFold(bits=-5)

    def test_repr_and_describe(self):
        pf = ParityFold(bits=16, seed=5)
        self.assertEqual(repr(pf), "ParityFold(bits=16, seed=5)")
        desc = pf.describe()
        self.assertIn("ParityFold", desc)
        self.assertIn("build(tokens)", desc)

    def test_build_consistency_and_parity(self):
        pf = ParityFold(bits=8, seed=1)
        tokens = [1, 2, 9, 1]
        sketch1 = pf.build(tokens)
        sketch2 = pf.build(tokens)
        # Consistency
        self.assertEqual(sketch1, sketch2)
        # Parity: bit index for token 1 toggled twice -> 0
        idx1 = hash((pf.seed, 1)) % pf.bits
        self.assertEqual(sketch1[idx1], 0)
        # token 2 toggled once -> 1
        idx2 = hash((pf.seed, 2)) % pf.bits
        self.assertEqual(sketch1[idx2], 1)

    def test_build_empty(self):
        pf = ParityFold(bits=8, seed=0)
        sketch = pf.build([])
        self.assertEqual(sketch, bytearray(8))
