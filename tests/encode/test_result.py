# tests/encode/test_result.py
import unittest
from collections import Counter

import numpy as np

from synrfp.encode.result import SynRFPResult


class TestSynRFPResult(unittest.TestCase):
    def setUp(self):
        self.tokens_R = Counter({0: 2, 1: 1})
        self.tokens_P = Counter({1: 1, 2: 3})
        self.delta = Counter({0: 2, 2: 3})
        self.support = [0, 2]
        # simple binary sketch as uint8 array
        self.sketch = np.array([0, 1, 0, 1], dtype=np.uint8)
        self.res = SynRFPResult(
            tokens_R=self.tokens_R,
            tokens_P=self.tokens_P,
            delta=self.delta,
            support=self.support,
            sketch=self.sketch,
            mode="delta",
        )

    def test_repr_contains_key_info(self):
        r = repr(self.res)
        self.assertIn("SynRFPResult", r)
        self.assertIn("mode='delta'", r)
        self.assertIn("sketch_type=", r)

    def test_describe(self):
        desc = SynRFPResult.describe()
        self.assertIn("SynRFPResult", desc)
        self.assertIn("mode='delta'|'union'", desc)

    def test_to_binary(self):
        bits = self.res.to_binary()
        self.assertIsInstance(bits, list)
        self.assertEqual(len(bits), len(self.sketch))
        self.assertTrue(all(b in (0, 1) for b in bits))

    def test_as_array(self):
        arr = self.res.as_array()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, self.sketch)


if __name__ == "__main__":
    unittest.main()
