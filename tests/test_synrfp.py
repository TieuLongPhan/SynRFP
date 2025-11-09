# ----------------------------------------------------------------------------
# tests/test_synrfp.py
import unittest
import numpy as np
from collections import Counter

from synrfp import build_graph_from_printout, tanimoto_bits, jaccard_minhash, SynRFP
from synrfp.graph.graph_data import GraphData
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.sketchers.parity_fold import ParityFold


class TestConvenienceFunctions(unittest.TestCase):

    def test_build_graph_from_printout(self):
        nodes = {0: {"element": "C"}, 1: {"element": "O"}}
        edges = {(0, 1): {"order": 1.5}}
        G = build_graph_from_printout(nodes, edges)
        self.assertIsInstance(G, GraphData)
        # should preserve dicts exactly (GraphData.from_dicts uses same structure)
        self.assertEqual(G.nodes, nodes)
        self.assertEqual(G.edges, {(0, 1): {"order": 1.5}})

    def test_tanimoto_bits(self):
        b1 = bytearray([1, 0, 1, 0])
        b2 = bytearray([1, 1, 0, 0])
        # Only bit 0 is common; union = bits 0,1,2 => 3
        self.assertAlmostEqual(tanimoto_bits(b1, b2), 1 / 3)
        # empty arrays => union=0 => return 0.0
        b_empty = bytearray([0, 0, 0])
        self.assertEqual(tanimoto_bits(b_empty, b_empty), 0.0)

    def test_jaccard_minhash(self):
        # identical arrays => 1.0
        h1 = [1, 2, 3]
        h2 = [1, 2, 3]
        self.assertEqual(jaccard_minhash(h1, h2), 1.0)
        # two out of three match => 2/3
        h3 = [1, 2, 4]
        self.assertAlmostEqual(jaccard_minhash(h1, h3), 2 / 3)


class TestSynRFP(unittest.TestCase):

    def test_empty_reaction(self):
        # reactant == product => empty delta
        G = GraphData.from_dicts({}, {})
        fp = SynRFP(
            tokenizer=WLTokenizer(), radius=1, sketch=ParityFold(bits=8, seed=0)
        )
        res = fp.fingerprint(G, G)

        self.assertIsInstance(res.tokens_R, Counter)
        self.assertIsInstance(res.tokens_P, Counter)
        # no tokens at all
        self.assertEqual(sum(res.tokens_R.values()), 0)
        self.assertEqual(sum(res.tokens_P.values()), 0)
        self.assertEqual(res.delta, Counter())
        self.assertEqual(res.support, [])

        # sketch should be a numpy uint8 array of zeros
        self.assertIsInstance(res.sketch, np.ndarray)
        self.assertEqual(res.sketch.dtype, np.uint8)
        self.assertTrue(bool((res.sketch == 0).all()))

    def test_simple_reaction_delta_and_sketch(self):
        # Reactant: two-node edge; Product: single isolated node
        nodes_r, edges_r = {0: {}, 1: {}}, {(0, 1): {}}
        nodes_p, edges_p = {0: {}}, {}
        G_r = GraphData.from_dicts(nodes_r, edges_r)
        G_p = GraphData.from_dicts(nodes_p, edges_p)

        fp = SynRFP(
            tokenizer=WLTokenizer(),
            radius=0,  # only atom‐level tokens
            sketch=ParityFold(bits=16, seed=42),
        )
        res = fp.fingerprint(G_r, G_p)

        # radius=0 → one token per node
        self.assertEqual(sum(res.tokens_R.values()), 2)
        self.assertEqual(sum(res.tokens_P.values()), 1)

        # delta has +1 for product, -2 for reactant
        pos = sum(v for v in res.delta.values() if v > 0)
        neg = sum(-v for v in res.delta.values() if v < 0)
        self.assertEqual(pos, 1)
        self.assertEqual(neg, 2)

        # support length == number of nonzero entries in delta
        self.assertEqual(len(res.support), len(res.delta))

        # sketch is a numpy array of uint8 bits of length bits
        self.assertIsInstance(res.sketch, np.ndarray)
        self.assertEqual(res.sketch.dtype, np.uint8)
        self.assertEqual(res.sketch.size, 16)


if __name__ == "__main__":
    unittest.main()
