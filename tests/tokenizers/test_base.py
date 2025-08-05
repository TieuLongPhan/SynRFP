# tests/tokenizers/test_base.py
import unittest
from collections import Counter

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.graph.graph_data import GraphData


class DummyTokenizer(BaseTokenizer):
    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        # simple implementation: return degree counts
        super().tokens_graph(G, radius)
        return Counter({v: G.degree(v) for v in G.nodes})


class TestBaseTokenizer(unittest.TestCase):
    def setUp(self):
        # build simple graph: path of length 2
        self.G = GraphData.from_dicts({0: {}, 1: {}, 2: {}}, {(0, 1): {}, (1, 2): {}})
        self.tokenizer = DummyTokenizer()

    def test_init_defaults(self):
        # default node_attrs and edge_attrs
        self.assertIsInstance(self.tokenizer.node_attrs, list)
        self.assertIsInstance(self.tokenizer.edge_attrs, list)
        self.assertIn("element", self.tokenizer.node_attrs)

    def test_tokens_graph_valid(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=1)
        # degrees: node 0 and 2 have degree 1, node1 has degree 2
        expected = Counter({0: 1, 1: 2, 2: 1})
        self.assertEqual(tokens, expected)

    def test_tokens_graph_invalid_G(self):
        with self.assertRaises(TypeError):
            self.tokenizer.tokens_graph(graph=None, radius=1)

    def test_tokens_graph_invalid_radius(self):
        with self.assertRaises(ValueError):
            self.tokenizer.tokens_graph(self.G, radius=-1)

    def test_tokens_side(self):
        # using two graphs
        G2 = GraphData.from_dicts({0: {}}, {})
        combined = self.tokenizer.tokens_side([self.G, G2], radius=1)
        # combine tokens from both
        expected = Counter({0: 1, 1: 2, 2: 1}) + Counter({0: 0})
        self.assertEqual(combined, Counter({0: 1, 1: 2, 2: 1}))
        self.assertEqual(combined, expected)

    def test_tokens_side_invalid(self):
        with self.assertRaises(TypeError):
            self.tokenizer.tokens_side([self.G, 123], radius=1)
