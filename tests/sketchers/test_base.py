# tests/sketchers/test_base.py
import unittest
from typing import Iterable
from collections import Counter
from synrfp.sketchers.base import BaseSketch


class DummySketch(BaseSketch):
    def build(self, support: Iterable[int]):
        # Return Counter of support
        return self._as_counter(support)


class TestBaseSketch(unittest.TestCase):
    def test_seed_validation(self):
        # Negative seed invalid
        with self.assertRaises(ValueError):
            DummySketch(seed=-1)
        # Zero and positive seeds valid
        ds0 = DummySketch(seed=0)
        self.assertEqual(ds0.seed, 0)
        ds1 = DummySketch(seed=42)
        self.assertEqual(ds1.seed, 42)

    def test_build_and_counter(self):
        ds = DummySketch(seed=7)
        data = [1, 2, 2, 3, 3, 3]
        result = ds.build(data)
        expected = Counter({3: 3, 2: 2, 1: 1})
        self.assertEqual(result, expected)

    def test_describe(self):
        ds = DummySketch()
        desc = ds.describe()
        self.assertIn("DummySketch", desc)
        self.assertIn("build", desc)
        self.assertTrue(isinstance(desc, str))

    def test_build_invalid_input(self):
        ds = DummySketch()
        with self.assertRaises(TypeError):
            ds.build(123)  # non-iterable should raise
