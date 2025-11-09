# ----------------------------------------------------------------------------
# tests/sketchers/test_parity_fold.py
import unittest
import numpy as np
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
        # use a larger bit-space to avoid accidental collisions for small token ids
        pf = ParityFold(bits=1024, seed=1)
        tokens = [1, 2, 9, 1]
        sketch1 = pf.build(tokens)
        sketch2 = pf.build(tokens)
        # Consistency (use numpy equality)
        np.testing.assert_array_equal(sketch1, sketch2)
        # dtype and length
        self.assertEqual(sketch1.dtype, np.uint8)
        self.assertEqual(sketch1.size, 1024)
        # Parity: token 1 toggled twice -> 0
        idx1 = hash((pf.seed, 1)) % pf.bits
        self.assertEqual(int(sketch1[idx1]), 0)
        # token 2 toggled once -> 1
        idx2 = hash((pf.seed, 2)) % pf.bits
        self.assertEqual(int(sketch1[idx2]), 1)

    def test_build_empty(self):
        pf = ParityFold(bits=8, seed=0)
        sketch = pf.build([])
        self.assertIsInstance(sketch, np.ndarray)
        np.testing.assert_array_equal(sketch, np.zeros(8, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
