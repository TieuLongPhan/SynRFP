# synrfp/tokenizers/nauty.py
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, List, Tuple, Optional, Sequence

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple
from synrfp.graph.graph_data import GraphData, NodeId

try:
    import pynauty

    _HAVE_PYNAUTY = True
except Exception:
    _HAVE_PYNAUTY = False


class NautyTokenizer(BaseTokenizer):
    """
    Canonical ego-subgraph tokenizer using nauty/Traces (colored graphs).

    Computes canonical certificates for k-hop ego subgraphs. When `pynauty`
    isn't available, falls back to a stable hash over node attribute tuples and
    induced degrees.

    :param require_pynauty: If True, raise if :mod:`pynauty` missing.
    :type require_pynauty: bool
    :param node_attrs: Node attributes to include in colors.
    :type node_attrs: Optional[Sequence[str]]
    :raises RuntimeError: If required and missing :mod:`pynauty`.

    Example
    -------
    >>> tok = NautyTokenizer(require_pynauty=False, node_attrs=['element'])
    >>> isinstance(tok, NautyTokenizer)
    True
    """

    def __init__(
        self, require_pynauty: bool = True, node_attrs: Optional[Sequence[str]] = None
    ):
        super().__init__(node_attrs=node_attrs)
        self.require_pynauty = bool(require_pynauty)
        if self.require_pynauty and not _HAVE_PYNAUTY:
            raise RuntimeError("pynauty is not available; install `pynauty`.")

    def __repr__(self) -> str:
        return (
            "NautyTokenizer(require_pynauty="
            + f"{self.require_pynauty}, node_attrs={self.node_attrs})"
        )

    @staticmethod
    def describe() -> str:
        return (
            "NautyTokenizer(require_pynauty=False, node_attrs=['element'])."
            "tokens_graph(graph, radius=2)\n"
        )

    # -------------------------------- helpers ---------------------------
    def _ego_nodes(self, G: GraphData, center: NodeId, r: int) -> List[NodeId]:
        seen = {center}
        q = deque([(center, 0)])
        while q:
            v, d = q.popleft()
            if d == r:
                continue
            for u in G.adj.get(v, []):
                if u not in seen:
                    seen.add(u)
                    q.append((u, d + 1))
        return sorted(seen)

    def _canonical_token(self, G: GraphData, nodes: List[NodeId]) -> int:
        colors = tuple(atom_label_tuple(G, v, self.node_attrs) for v in nodes)
        degs = tuple(sum(1 for u in G.adj.get(v, []) if u in nodes) for v in nodes)
        if not _HAVE_PYNAUTY or not self.require_pynauty:
            return _h64(("fallback", colors, degs))
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

    # -------------------------------- main ------------------------------
    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Tokenize by canonicalizing ego subgraphs up to radius ``r``.

        :param G: Molecular graph to tokenize.
        :type G: GraphData
        :param radius: Hop-distance for ego subgraphs (>=0).
        :type radius: int
        :returns: Counter of canonical subgraph hashes.
        :rtype: Counter
        """
        super().tokens_graph(G, radius)
        out: Counter = Counter()
        for v in G.nodes:
            for k in range(radius + 1):
                ego = self._ego_nodes(G, v, k)
                out[self._canonical_token(G, ego)] += 1
        return out
