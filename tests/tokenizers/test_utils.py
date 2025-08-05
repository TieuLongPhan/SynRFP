# tests/tokenizers/test_utils.py
import unittest
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.graph_data import GraphData


class TestUtils(unittest.TestCase):
    def setUp(self):
        # graph: two nodes 0--1
        self.G = GraphData.from_dicts({0: {"a": 1}, 1: {"a": 2}}, {(0, 1): {"w": 3}})

    def test_h64_stability(self):
        h1 = _h64(("a", 1))
        h2 = _h64(("a", 1))
        self.assertEqual(h1, h2)
        self.assertIsInstance(h1, int)

    def test_atom_label_tuple(self):
        t0 = atom_label_tuple(self.G, 0, ["a"])
        # returns (1, degree)
        self.assertEqual(t0, (1, self.G.degree(0)))

    def test_bond_label_tuple(self):
        t = bond_label_tuple(self.G, 0, 1, ["w"])
        self.assertEqual(t, (3,))
