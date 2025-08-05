# synrfp/tokenizers/nauty.py
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Sequence

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple
from synrfp.graph.graph_data import GraphData, NodeId

# module-level flag set at import
try:
    import pynauty

    _HAVE_PYNAUTY = True
except ImportError:
    _HAVE_PYNAUTY = False


class NautyTokenizer(BaseTokenizer):
    """
    Canonical ego-subgraph tokenizer using nauty/Traces.

    Computes canonical hash certificates for k-hop ego subgraphs.

    :param require_pynauty: Whether to enforce pynauty availability.
    :type require_pynauty: bool
    :param node_attrs: Attributes to include in node coloring.
    :type node_attrs: Optional[Sequence[str]]

    :raises RuntimeError: If pynauty is required but not installed.

    :example:
    >>> tokenizer = NautyTokenizer(require_pynauty=False, node_attrs=['element'])
    >>> tokens = tokenizer.tokens_graph(graph, radius=3)
    """

    def __init__(
        self,
        require_pynauty: bool = True,
        node_attrs: Optional[Sequence[str]] = None,
    ):
        super().__init__(node_attrs=node_attrs)
        self.require_pynauty = bool(require_pynauty)
        if self.require_pynauty and not _HAVE_PYNAUTY:
            raise RuntimeError("pynauty is not available; install `pynauty`.")

    def __repr__(self) -> str:
        return (
            f"NautyTokenizer(require_pynauty={self.require_pynauty},"
            + f" node_attrs={self.node_attrs})"
        )

    @staticmethod
    def describe() -> str:
        return (
            "tokenizer = NautyTokenizer(require_pynauty=False, node_attrs=['element'])"
            "tokens = tokenizer.tokens_graph(graph, radius=2)"
        )

    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Tokenize a graph by canonicalizing each ego subgraph up to radius r.

        :param G: Molecular graph to tokenize.
        :type G: GraphData
        :param radius: Hop-distance for ego subgraphs.
        :type radius: int
        :returns: Counter of canonical subgraph hash tokens.
        :rtype: Counter
        """
        super().tokens_graph(G, radius)
        out: Counter = Counter()

        def ego_nodes(center: NodeId, r: int) -> List[NodeId]:
            seen = {center}
            frontier = {center}
            for _ in range(r):
                nxt = set()
                for v in frontier:
                    for u in G.adj.get(v, []):
                        if u not in seen:
                            seen.add(u)
                            nxt.add(u)
                frontier = nxt
                if not frontier:
                    break
            return sorted(seen)

        def canonical_token(nodes: List[NodeId]) -> int:
            # Vertex coloring based on selected attributes
            colors = tuple(atom_label_tuple(G, v, self.node_attrs) for v in nodes)
            # degree sequence
            degs = tuple(sum(1 for u in G.adj.get(v, []) if u in nodes) for v in nodes)
            if not self.require_pynauty:
                return _h64(("fallback", colors, degs))
            # use pynauty for canonical certificate
            idx = {v: i for i, v in enumerate(nodes)}
            edges = [
                (idx[v], idx[u])
                for v in nodes
                for u in G.adj.get(v, [])
                if u in idx and idx[v] < idx[u]
            ]
            color_map: Dict[Tuple, List[int]] = defaultdict(list)
            for v in nodes:
                color_map[atom_label_tuple(G, v, self.node_attrs)].append(idx[v])
            blocks = list(color_map.values())
            H = pynauty.Graph(
                number_of_vertices=len(nodes),
                edges=edges,
                directed=False,
                vertex_coloring=blocks,
            )
            cert = pynauty.certificate(H)
            return _h64(cert)

        for v in G.nodes:
            for k in range(radius + 1):
                ego = ego_nodes(v, k)
                out[canonical_token(ego)] += 1
        return out
