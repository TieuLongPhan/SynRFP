# tests/tokenizers/test_nauty.py
import unittest
from synrfp.tokenizers.nauty import NautyTokenizer
from synrfp.graph.graph_data import GraphData


class TestNautyTokenizer(unittest.TestCase):
    def setUp(self):
        # path graph 0-1
        self.G = GraphData.from_dicts({0: {}, 1: {}}, {(0, 1): {}})
        # allow fallback
        self.tokenizer = NautyTokenizer(require_pynauty=False)

    def test_repr(self):
        self.assertIn("NautyTokenizer", repr(self.tokenizer))

    def test_describe(self):
        self.assertIn("NautyTokenizer", NautyTokenizer.describe())

    def test_tokens_graph_radius0(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=0)
        # radius=0 yields one token per node => 2 tokens
        self.assertEqual(sum(tokens.values()), 2)

    def test_tokens_graph_radius1(self):
        tokens1 = self.tokenizer.tokens_graph(self.G, radius=1)
        # each node has ego of size2 for k=1, so two radius=0 + two radius=1 => total 4
        self.assertEqual(sum(tokens1.values()), 4)


if __name__ == "__main__":
    unittest.main()
