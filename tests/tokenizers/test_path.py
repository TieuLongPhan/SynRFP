# tests/tokenizers/test_path.py
import unittest
from collections import Counter
from synrfp.tokenizers.path import PathTokenizer
from synrfp.graph.molecule import Molecule


class TestPathTokenizer(unittest.TestCase):
    def setUp(self):
        # triangle graph 0-1-2-0
        self.G = Molecule.from_dicts(
            {0: {}, 1: {}, 2: {}},
            {(0, 1): {"order": 1}, (1, 2): {"order": 1}, (0, 2): {"order": 1}},
        )
        # default min_path=1
        self.tokenizer = PathTokenizer()

    def test_repr(self):
        self.assertIn("PathTokenizer", repr(self.tokenizer))

    def test_describe(self):
        desc = PathTokenizer.describe()
        self.assertIn("PathTokenizer", desc)
        self.assertIn("tokens_graph", desc)

    def test_tokens_graph_radius1(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=1)
        # Each node emits two simple paths of length 1 -> total 6 tokens
        self.assertIsInstance(tokens, Counter)
        self.assertEqual(sum(tokens.values()), 6)

    def test_tokens_graph_radius2(self):
        tokens2 = self.tokenizer.tokens_graph(self.G, radius=2)
        # radius=2 includes length-1 paths (6) and length-2 paths (6) => 12 total
        self.assertIsInstance(tokens2, Counter)
        self.assertEqual(sum(tokens2.values()), 12)

    def test_invalid_radius_raises(self):
        with self.assertRaises(ValueError):
            self.tokenizer.tokens_graph(self.G, radius=-1)


if __name__ == "__main__":
    unittest.main()
