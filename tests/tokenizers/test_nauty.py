# tests/tokenizers/test_nauty.py
import unittest
from synrfp.tokenizers.nauty import NautyTokenizer
from synrfp.graph.molecule import Molecule


class TestNautyTokenizer(unittest.TestCase):
    def setUp(self):
        # path graph 0-1
        self.G = Molecule.from_dicts({0: {}, 1: {}}, {(0, 1): {}})
        # use default NautyTokenizer (pure-networkx canonicalizer)
        self.tokenizer = NautyTokenizer()

    def test_repr(self):
        self.assertIn("NautyTokenizer", repr(self.tokenizer))

    def test_tokens_graph_radius0(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=0)
        # radius=0 yields one token per node => 2 tokens
        self.assertEqual(sum(tokens.values()), 2)

    def test_tokens_graph_radius1(self):
        tokens1 = self.tokenizer.tokens_graph(self.G, radius=1)
        # each node has ego of size 2 for k=1, so two radius=0 + two radius=1 => total 4
        self.assertEqual(sum(tokens1.values()), 4)


if __name__ == "__main__":
    unittest.main()
