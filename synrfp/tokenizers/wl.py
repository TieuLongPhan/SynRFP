# ------------------------------------------
# synrfp/tokenizers/wl.py
from collections import Counter
from typing import Dict

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.graph_data import GraphData, NodeId


class WLTokenizer(BaseTokenizer):
    """
    Weisfeiler–Lehman subtree tokenizer.

    Applies r iterations of subtree hashing to encapsulate local structure.

    :param node_attrs: Attributes to include in initial node labels.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Attributes to include in bond labels.
    :type edge_attrs: Optional[Sequence[str]]

    :example:
    >>> tokenizer = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])
    >>> tokens = tokenizer.tokens_graph(graph, radius=2)
    """

    def __repr__(self) -> str:
        return (
            f"WLTokenizer(node_attrs={self.node_attrs}, "
            f"edge_attrs={self.edge_attrs})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the WLTokenizer.

        :returns: Example instantiation and usage code.
        :rtype: str
        """
        return (
            "tokenizer = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])\n"
            "tokens = tokenizer.tokens_graph(graph, radius=2)\n"
        )

    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Tokenize a graph via WL subtree hashing.

        :param G: Molecular graph to tokenize.
        :type G: GraphData
        :param radius: Number of WL iterations.
        :type radius: int
        :returns: Counter of subtree hash tokens.
        :rtype: Counter
        """
        super().tokens_graph(G, radius)
        # Initial node hashes
        labels: Dict[NodeId, int] = {
            v: _h64(("a0",) + atom_label_tuple(G, v, self.node_attrs)) for v in G.nodes
        }
        out = Counter(labels.values())

        # Iterative neighborhood hashing
        for k in range(1, radius + 1):
            new_labels: Dict[NodeId, int] = {}
            for v in G.nodes:
                neigh = []
                for u in G.adj.get(v, []):
                    neigh.append(
                        (
                            _h64(("b",) + bond_label_tuple(G, v, u, self.edge_attrs)),
                            labels[u],
                        )
                    )
                neigh.sort()
                new_labels[v] = _h64(("wl", k, labels[v], tuple(neigh)))
            labels = new_labels
            out.update(labels.values())
        return out
