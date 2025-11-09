# ----------------------------------------------------------------------------
# tests/sketchers/test_minhash_sketch.py
import unittest
from synrfp.sketchers.minhash_sketch import MinHashSketch, _HAVE_DS


class TestMinHashSketch(unittest.TestCase):
    def test_instantiation_and_invalid_m(self):
        if not _HAVE_DS:
            with self.assertRaises(RuntimeError):
                MinHashSketch()
        else:
            # valid
            mh = MinHashSketch(m=4, seed=2)
            self.assertEqual(mh.m, 4)
            # invalid m
            with self.assertRaises(ValueError):
                MinHashSketch(m=0, seed=1)

    @unittest.skipUnless(_HAVE_DS, "requires datasketch")
    def test_build_determinism_and_seed_variation(self):
        mh1 = MinHashSketch(m=8, seed=42)
        tokens = [10, 20, 30]
        sketch1 = mh1.build(tokens)
        sketch2 = mh1.build(tokens)
        self.assertEqual(len(sketch1), 8)
        self.assertEqual(sketch1, sketch2)
        # different seed yields different
        mh2 = MinHashSketch(m=8, seed=0)
        sketch3 = mh2.build(tokens)
        self.assertNotEqual(sketch1, sketch3)
