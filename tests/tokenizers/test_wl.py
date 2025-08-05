# tests/tokenizers/test_wl.py
import unittest
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
        self.assertIn("tokenizer = WLTokenizer", WLTokenizer.describe())

    def test_tokens_graph_radius0(self):
        tokens = self.tokenizer.tokens_graph(self.G, radius=0)
        # radius=0 yields only initial atom labels count=3
        self.assertEqual(sum(tokens.values()), 3)

    def test_tokens_graph_radius1(self):
        tokens1 = self.tokenizer.tokens_graph(self.G, radius=1)
        # radius=1 adds one iteration per node, total tokens = 3 + 3 =6
        self.assertEqual(sum(tokens1.values()), 6)
