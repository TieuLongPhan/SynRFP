# tests/graph/test_graph_data.py

import unittest
import networkx as nx
from synrfp.graph.graph_data import GraphData


class TestGraphData(unittest.TestCase):

    def setUp(self):
        # Basic two‐node graph
        self.nodes = {
            0: {"element": "C", "aromatic": False},
            1: {"element": "O", "aromatic": True},
        }
        self.edges = {
            (0, 1): {"order": 2.0},
        }

    def test_empty_graph(self):
        G = GraphData.from_dicts({}, {})
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)
        self.assertEqual(G.adj, {})  # no adjacency
        self.assertEqual(G.degree(42), 0)  # nonexistent node
        with self.assertRaises(KeyError):
            _ = G.edge_attr(0, 1)

    def test_edge_key_normalization(self):
        # provide reversed key
        nodes = {0: {}, 1: {}}
        edges = {(1, 0): {"foo": "bar"}}
        G = GraphData.from_dicts(nodes, edges)
        # internal key must be (0,1)
        self.assertIn((0, 1), G.edges)
        self.assertNotIn((1, 0), G.edges)
        # retrieval works both ways
        self.assertEqual(G.edge_attr(0, 1), {"foo": "bar"})
        self.assertEqual(G.edge_attr(1, 0), {"foo": "bar"})

    def test_self_loop(self):
        nodes = {0: {"element": "X"}}
        edges = {(0, 0): {"order": 1.0}}
        G = GraphData.from_dicts(nodes, edges)
        # adjacency: self‐loop appears twice
        self.assertEqual(G.adj, {0: [0, 0]})
        # degree counts both ends
        self.assertEqual(G.degree(0), 2)
        self.assertEqual(G.edge_attr(0, 0), {"order": 1.0})

    def test_from_dicts_and_repr(self):
        G = GraphData.from_dicts(self.nodes, self.edges)
        # repr shows correct counts
        self.assertEqual(repr(G), "GraphData(nodes=2, edges=1)")
        # copies of inputs
        self.assertIsNot(G.nodes, self.nodes)
        self.assertIsNot(G.edges, self.edges)

    def test_adj_and_degree_caching(self):
        G = GraphData.from_dicts(self.nodes, self.edges)
        adj1 = G.adj
        adj2 = G.adj
        self.assertIs(adj1, adj2)  # same object cached

    def test_from_nx_graph_equivalence(self):
        # mirror with networkx
        nxg = nx.Graph()
        for n, attr in self.nodes.items():
            nxg.add_node(n, **attr)
        for (u, v), attr in self.edges.items():
            nxg.add_edge(u, v, **attr)

        G1 = GraphData.from_dicts(self.nodes, self.edges)
        G2 = GraphData.from_nx_graph(nxg)
        self.assertEqual(G1.nodes, G2.nodes)
        self.assertEqual(G1.edges, G2.edges)


if __name__ == "__main__":
    unittest.main()
