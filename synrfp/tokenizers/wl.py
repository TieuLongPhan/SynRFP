# synrfp/tokenizers/wl.py
from __future__ import annotations

from collections import Counter
from typing import Dict

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.graph_data import GraphData, NodeId


class WLTokenizer(BaseTokenizer):
    """
    Weisfeiler–Lehman subtree tokenizer (edge-aware; k=0..r tokens).

    Node labels: selected node attrs + degree.
    Edge labels: selected bond attrs (e.g., order, aromaticity).

    :example:
        >>> tok = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])
        >>> isinstance(tok, WLTokenizer)
        True
    """

    def __repr__(self) -> str:
        return (
            f"WLTokenizer(node_attrs={self.node_attrs}, edge_attrs={self.edge_attrs})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the WLTokenizer.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "WLTokenizer(node_attrs=['element'], edge_attrs=['order'])."
            "tokens_graph(graph, radius=2)\n"
        )

    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Tokenize a graph via edge-aware WL subtree hashing.

        :param G: Molecular graph to tokenize.
        :type G: GraphData
        :param radius: Number of WL iterations (>=0).
        :type radius: int
        :returns: Counter of subtree-hash tokens.
        :rtype: Counter
        """
        super().tokens_graph(G, radius)

        labels: Dict[NodeId, int] = {
            v: _h64(("a0",) + atom_label_tuple(G, v, self.node_attrs)) for v in G.nodes
        }
        out = Counter(labels.values())

        for k in range(1, radius + 1):
            new_labels: Dict[NodeId, int] = {}
            for v in G.nodes:
                neigh = []
                for u in G.adj.get(v, []):
                    bl = _h64(("b",) + bond_label_tuple(G, v, u, self.edge_attrs))
                    neigh.append((bl, labels[u]))
                neigh.sort()
                new_labels[v] = _h64(("wl", k, labels[v], tuple(neigh)))
            labels = new_labels
            out.update(labels.values())
        return out
