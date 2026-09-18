# tests/tokenizers/test_map.py
import unittest
from collections import Counter

from synrfp import synrfp
from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.map import MAPTokenizer


def _chain(ids):
    """Path graph over ``ids`` in the given order, carbon atoms, single bonds."""
    nodes = {i: {"element": "C"} for i in ids}
    edges = {(a, b): {"order": 1} for a, b in zip(ids, ids[1:])}
    return Molecule.from_dicts(nodes, edges)


class TestMAPTokenizer(unittest.TestCase):
    def test_returns_counter(self):
        tokens = MAPTokenizer().tokens_graph(_chain([0, 1, 2]), radius=1)
        self.assertIsInstance(tokens, Counter)
        self.assertGreater(sum(tokens.values()), 0)

    def test_node_relabelling_without_edge_attrs(self):
        tok = MAPTokenizer(edge_attrs=[])
        self.assertEqual(
            tok.tokens_graph(_chain([0, 1, 2, 3]), radius=2),
            tok.tokens_graph(_chain([7, 3, 9, 1]), radius=2),
        )

    def test_isolated_atom_is_tokenized(self):
        G = Molecule.from_dicts({0: {"element": "Cl", "charge": -1}}, {})
        self.assertEqual(sum(MAPTokenizer().tokens_graph(G, radius=2).values()), 2)

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            MAPTokenizer().tokens_graph(_chain([0, 1]), radius=0)

    def test_atom_order_invariance_end_to_end(self):
        for edge_attrs in (None, []):
            with self.subTest(edge_attrs=edge_attrs):
                self.assertEqual(
                    synrfp("CCO>>CC=O", tokenizer="map", edge_attrs=edge_attrs),
                    synrfp("OCC>>O=CC", tokenizer="map", edge_attrs=edge_attrs),
                )

    def test_ion_exchange_is_not_empty(self):
        self.assertGreater(sum(synrfp("[Cl-]>>[Br-]", tokenizer="map")), 0)


if __name__ == "__main__":
    unittest.main()
