# tests/test_invariance.py
"""Invariants stated in the paper, checked end to end on reaction SMILES."""

import unittest

import numpy as np

from synrfp import synrfp
from synrfp.encode.engine import SynRFP
from synrfp.graph.reaction import Reaction
from synrfp.sketchers.parity_fold import ParityFold
from synrfp.tokenizers.wl import WLTokenizer

ESTER = "CC(=O)O.OCC>>CC(=O)OCC.O"
SKETCHES = ("parity", "minhash", "cw", "srp")


class TestInvariance(unittest.TestCase):
    def test_component_permutation(self):
        for sk in SKETCHES:
            with self.subTest(sketch=sk):
                self.assertEqual(
                    synrfp(ESTER, sketch=sk),
                    synrfp("OCC.CC(=O)O>>O.CC(=O)OCC", sketch=sk),
                )

    def test_atom_order(self):
        for sk in SKETCHES:
            with self.subTest(sketch=sk):
                self.assertEqual(
                    synrfp(ESTER, sketch=sk),
                    synrfp("OC(C)=O.C(C)O>>C(C)OC(C)=O.O", sketch=sk),
                )

    def test_reversal_negates_delta_and_preserves_union(self):
        rxn = Reaction.from_rsmi(ESTER)
        eng = SynRFP(tokenizer=WLTokenizer(), radius=2, sketch=ParityFold(1024))
        fwd = eng.fingerprint(rxn.reactant, rxn.product, mode="delta").delta
        rev = eng.fingerprint(rxn.product, rxn.reactant, mode="delta").delta
        self.assertEqual({t: -c for t, c in fwd.items()}, dict(rev))
        u_fwd = eng.fingerprint(rxn.reactant, rxn.product, mode="union").delta
        u_rev = eng.fingerprint(rxn.product, rxn.reactant, mode="union").delta
        self.assertEqual(dict(u_fwd), dict(u_rev))

    def test_spectators_cancel_in_delta(self):
        for sk in SKETCHES:
            with self.subTest(sketch=sk):
                self.assertEqual(
                    synrfp(ESTER, sketch=sk),
                    synrfp("CC(=O)O.OCC.ClCCl>>CC(=O)OCC.O.ClCCl", sketch=sk),
                )

    def test_agents(self):
        with_agent = "CC(=O)O.OCC>[H+]>CC(=O)OCC.O"
        self.assertEqual(synrfp(ESTER), synrfp(with_agent, agents="both"))
        self.assertEqual(synrfp(ESTER), synrfp(with_agent, agents="drop"))
        self.assertNotEqual(synrfp(ESTER), synrfp(with_agent, agents="reactants"))

    def test_invalid_reaction_raises_value_error(self):
        with self.assertRaises(ValueError):
            synrfp("CCO")

    def test_fixed_length_all_sketches(self):
        for sk in SKETCHES:
            for bits in (512, 1024, 2048):
                with self.subTest(sketch=sk, bits=bits):
                    fp = np.asarray(synrfp(ESTER, sketch=sk, bits=bits))
                    self.assertEqual(fp.size, bits)
                    self.assertTrue(set(np.unique(fp)) <= {0, 1})


class TestProvenance(unittest.TestCase):
    def test_provenance_matches_tokens(self):
        rxn = Reaction.from_rsmi(ESTER)
        tok = WLTokenizer()
        prov = tok.tokens_with_provenance(rxn.reactant, 2)
        counts = tok.tokens_graph(rxn.reactant, 2)
        self.assertEqual({t: len(v) for t, v in prov.items()}, dict(counts))


class TestFingerprintFromTokens(unittest.TestCase):
    def test_matches_synrfp(self):
        from synrfp.encode.core import fingerprint_from_tokens

        rxn = Reaction.from_rsmi(ESTER)
        tok = WLTokenizer()
        tr, tp = tok.tokens_graph(rxn.reactant, 2), tok.tokens_graph(rxn.product, 2)
        for sk in SKETCHES:
            for mode in ("delta", "union"):
                with self.subTest(sketch=sk, mode=mode):
                    a = fingerprint_from_tokens(tr, tp, sketch=sk, mode=mode)
                    b = np.asarray(synrfp(ESTER, sketch=sk, mode=mode))
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
