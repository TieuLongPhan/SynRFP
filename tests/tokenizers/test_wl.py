# ----------------------------------------------------------------------------
# tests/tokenizers/test_wl.py
import unittest
from collections import Counter
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.graph.graph_data import GraphData


class TestWLTokenizer(unittest.TestCase):
    def setUp(self):
        # triangle graph 0-1-2-0
        self.G = GraphData.from_dicts(
            {0: {}, 1: {}, 2: {}},
            {(0, 1): {"order": 1}, (1, 2): {"order": 1}, (0, 2): {"order": 1}},
        )
        self.tokenizer = WLTokenizer()

    def test_repr(self):
        self.assertIn("WLTokenizer", repr(self.tokenizer))

    def test_describe(self):
        desc = WLTokenizer.describe()
        self.assertIn("WLTokenizer", desc)
        self.assertIn("tokens_graph", desc)

    def test_tokens_graph_radius0(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=0)
        # Expect a Counter-like result and 3 initial node labels
        self.assertIsInstance(tokens, Counter)
        self.assertEqual(sum(tokens.values()), 3)

    def test_tokens_graph_radius1(self):
        tokens1 = self.tokenizer.tokens_graph(self.G, radius=1)
        # radius=1 yields k=0 and k=1 tokens: 3 + 3 = 6
        self.assertIsInstance(tokens1, Counter)
        self.assertEqual(sum(tokens1.values()), 6)

    def test_invalid_radius_raises(self):
        with self.assertRaises(ValueError):
            self.tokenizer.tokens_graph(self.G, radius=-1)


if __name__ == "__main__":
    unittest.main()
