# tests/encode/test_engine.py
import unittest
from collections import Counter

from synrfp.encode.engine import SynRFP
from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.base import BaseTokenizer
from synrfp.sketchers.base import BaseSketch, WeightedSketch


class DummyTokenizer(BaseTokenizer):
    """Very simple tokenizer: returns one token per node (the node id itself)."""

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        # keep BaseTokenizer validation
        super().tokens_graph(G, radius)
        # iterate G.nodes (works if nodes is a dict or an iterable)
        nodes = list(G.nodes)
        return Counter(nodes)


class DummySketch(BaseSketch):
    """Records the last set of tokens it was called with."""

    def __init__(self):
        self.last_tokens = None

    def build(self, tokens):
        self.last_tokens = list(tokens)
        return list(tokens)


class DummyWeightedSketch(WeightedSketch):
    """Records the last (pos, neg) maps it was called with."""

    def __init__(self):
        self.last_pos = None
        self.last_neg = None

    def build(self, pos, neg):
        self.last_pos = dict(pos)
        self.last_neg = dict(neg)
        # return something, content does not matter for the engine
        return {"pos": dict(pos), "neg": dict(neg)}


class TestSynRFPEngine(unittest.TestCase):
    def setUp(self):
        # Reactant: nodes {0, 1}
        self.reactant = Molecule.from_dicts({0: {}, 1: {}}, {(0, 1): {}})
        # Product: nodes {1, 2}
        self.product = Molecule.from_dicts({1: {}, 2: {}}, {(1, 2): {}})
        self.tokenizer = DummyTokenizer()

    def test_invalid_constructor_both_or_none_sketchers(self):
        with self.assertRaises(ValueError):
            SynRFP(self.tokenizer, radius=0, sketch=None, weighted_sketch=None)
        with self.assertRaises(ValueError):
            SynRFP(
                self.tokenizer,
                radius=0,
                sketch=DummySketch(),
                weighted_sketch=DummyWeightedSketch(),
            )

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            SynRFP(self.tokenizer, radius=-1, sketch=DummySketch())

    def test_delta_mode_unweighted_encodes_sign(self):
        sk = DummySketch()
        engine = SynRFP(self.tokenizer, radius=0, sketch=sk)

        res = engine.fingerprint(self.reactant, self.product, mode="delta")

        # tokens_R = {0:1,1:1}, tokens_P = {1:1,2:1}
        self.assertEqual(res.tokens_R, Counter({0: 1, 1: 1}))
        self.assertEqual(res.tokens_P, Counter({1: 1, 2: 1}))

        # delta = P - R = {2:1, 0:-1}
        self.assertEqual(res.delta, Counter({2: 1, 0: -1}))

        # encoded support: token 0 consumed -> 2*0+1=1; token 2 created ->2*2+0=4
        self.assertIsNotNone(sk.last_tokens)
        self.assertEqual(set(sk.last_tokens), {1, 4})
        # support in result matches what was passed to sketcher
        self.assertEqual(set(res.support), set(sk.last_tokens))

    def test_union_mode_unweighted(self):
        sk = DummySketch()
        engine = SynRFP(self.tokenizer, radius=0, sketch=sk)

        res = engine.fingerprint(self.reactant, self.product, mode="union")

        # union_counts = {0:1, 1:2, 2:1}
        self.assertEqual(res.delta, Counter({0: 1, 1: 2, 2: 1}))
        self.assertEqual(set(res.support), {0, 1, 2})
        self.assertEqual(set(sk.last_tokens), {0, 1, 2})

    def test_delta_mode_weighted(self):
        sk = DummyWeightedSketch()
        engine = SynRFP(self.tokenizer, radius=0, weighted_sketch=sk)

        res = engine.fingerprint(self.reactant, self.product, mode="delta")

        # delta = {2:1,0:-1}
        self.assertEqual(res.delta, Counter({2: 1, 0: -1}))
        # pos_map = {2:1}, neg_map = {0:1}
        self.assertEqual(sk.last_pos, {2: 1})
        self.assertEqual(sk.last_neg, {0: 1})

    def test_union_mode_weighted(self):
        sk = DummyWeightedSketch()
        engine = SynRFP(self.tokenizer, radius=0, weighted_sketch=sk)

        res = engine.fingerprint(self.reactant, self.product, mode="union")

        # union_counts = {0:1, 1:2, 2:1}
        self.assertEqual(res.delta, Counter({0: 1, 1: 2, 2: 1}))
        self.assertEqual(sk.last_pos, {0: 1, 1: 2, 2: 1})
        self.assertEqual(sk.last_neg, {})

    def test_invalid_mode_raises(self):
        sk = DummySketch()
        engine = SynRFP(self.tokenizer, radius=0, sketch=sk)
        with self.assertRaises(ValueError):
            engine.fingerprint(self.reactant, self.product, mode="invalid")


if __name__ == "__main__":
    unittest.main()
