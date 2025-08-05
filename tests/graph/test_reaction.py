# tests/graph/test_reaction.py

import unittest
import networkx as nx
import pandas as pd

from synrfp.graph.graph_data import GraphData
from synrfp.graph.reaction import Reaction


def make_test_graph(num_nodes: int, edge_list):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, element="X")
    for u, v in edge_list:
        G.add_edge(u, v, order=1.0)
    return G


class TestReaction(unittest.TestCase):

    def test_from_graph_and_repr(self):
        G_r = make_test_graph(3, [(0, 1), (1, 2)])
        G_p = make_test_graph(2, [(0, 1)])
        rxn = Reaction.from_graph(G_r, G_p)

        rep = repr(rxn)
        self.assertIn("reactant_nodes=3", rep)
        self.assertIn("product_nodes=2", rep)
        self.assertIsInstance(rxn.reactant, GraphData)
        self.assertIsInstance(rxn.product, GraphData)

    def test_len_iter_getitem_to_dataframe(self):
        G_r = make_test_graph(2, [(0, 1)])
        G_p = make_test_graph(1, [])
        rxn = Reaction.from_graph(G_r, G_p)

        # length
        self.assertEqual(len(rxn), 2)

        # iteration
        sides = list(rxn)
        self.assertIs(sides[0], rxn.reactant)
        self.assertIs(sides[1], rxn.product)

        # indexing
        self.assertIs(rxn[0], rxn.reactant)
        self.assertIs(rxn["reactant"], rxn.reactant)
        self.assertIs(rxn[1], rxn.product)
        self.assertIs(rxn["product"], rxn.product)
        with self.assertRaises(KeyError):
            _ = rxn["unknown"]

        # DataFrame summary
        df = rxn.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ["side", "n_nodes", "n_edges"])
        df2 = df.set_index("side")
        self.assertEqual(df2.loc["reactant", "n_nodes"], 2)
        self.assertEqual(df2.loc["product", "n_edges"], 0)

    def test_from_rsmi_integration(self):
        """Integration test: real synkit.IO.rsmi_to_graph on 'CCO>>C=C.O'."""
        rsmi = "CCO>>C=C.O"
        rxn = Reaction.from_rsmi(rsmi)

        # Expect reactant CCO: 3 atoms, 2 bonds (chain)
        self.assertEqual(len(rxn.reactant.nodes), 3)
        self.assertEqual(len(rxn.reactant.edges), 2)

        # Expect product C=C.O: 3 atoms, one double bond + one isolated
        self.assertEqual(len(rxn.product.nodes), 3)
        # Exactly one bond in the alkene fragment
        self.assertEqual(len(rxn.product.edges), 1)

        # And support the rest of the API
        self.assertEqual(len(rxn), 2)
        repr_str = repr(rxn)
        self.assertIn("reactant_nodes=3", repr_str)
        self.assertIn("product_nodes=3", repr_str)


if __name__ == "__main__":
    unittest.main()
