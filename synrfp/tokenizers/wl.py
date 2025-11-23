# synrfp/tokenizers/wl.py
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence

try:
    import networkx as nx
    from networkx.algorithms.graph_hashing import (
        weisfeiler_lehman_subgraph_hashes as _wl_hashes,
    )

    _HAVE_NX_WL = True
except Exception:
    nx = None
    _wl_hashes = None
    _HAVE_NX_WL = False

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.molecule import Molecule, NodeId


class WLTokenizer(BaseTokenizer):
    """
    Weisfeiler–Lehman subtree tokenizer (edge-aware; k=0..r tokens).

    Node labels: selected node attrs + degree.
    Edge labels: selected bond attrs (e.g., order, aromaticity).

    The tokenizer supports two backends:

    * A lightweight in-house WL refinement loop (default).
    * An optional NetworkX-based WL implementation
      (:func:`networkx.algorithms.graph_hashing.weisfeiler_lehman_subgraph_hashes`),
      enabled via ``use_nx=True``.

    :example:
        >>> tok = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])
        >>> isinstance(tok, WLTokenizer)
        True
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        use_nx: bool = False,
        require_networkx: bool = False,
    ) -> None:
        """
        Initialize a WLTokenizer instance.

        :param node_attrs: Node attributes to include in the initial node labels
            (e.g. ``['element', 'charge', 'hcount']``). If ``None``, a default
            atom label tuple defined in :mod:`synrfp.tokenizers.utils` is used.
        :type node_attrs: Optional[Sequence[str]]
        :param edge_attrs: Edge attributes to include in the edge labels
            (e.g. ``['order', 'aromatic']``). If ``None``, a default bond label
            tuple is used.
        :type edge_attrs: Optional[Sequence[str]]
        :param use_nx: Whether to use NetworkX's Weisfeiler–Lehman
            implementation when available. If ``False`` (default), the
            in-house implementation is used.
        :type use_nx: bool
        :param require_networkx: If ``True`` and the NetworkX WL hashing
            functions are not available, a :class:`RuntimeError` is raised
            instead of falling back to the in-house implementation. Ignored
            when ``use_nx`` is ``False``.
        :type require_networkx: bool
        """
        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)
        self.use_nx: bool = bool(use_nx)
        self.require_networkx: bool = bool(require_networkx)

        if self.use_nx and self.require_networkx and not _HAVE_NX_WL:
            raise RuntimeError(
                "NetworkX WL hashing is required but not available. "
                "Install `networkx>=2.8` or set `use_nx=False`."
            )

    def __repr__(self) -> str:
        return (
            f"WLTokenizer(node_attrs={self.node_attrs}, "
            f"edge_attrs={self.edge_attrs}, use_nx={self.use_nx})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the :class:`WLTokenizer`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.tokenizers.wl import WLTokenizer\n"
            "tok = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])\n"
            "C = tok.tokens_graph(graph, radius=2)\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        """
        Tokenize a molecular graph via edge-aware WL subtree hashing.

        The behaviour is controlled by the ``use_nx`` flag:

        * If ``use_nx=True`` and NetworkX WL hashing is available, this method
          uses :func:`networkx.algorithms.graph_hashing.weisfeiler_lehman_subgraph_hashes`
          on a temporary NetworkX graph with precomputed atom/bond labels, and
          folds the resulting hex hashes into integer tokens using
          :func:`synrfp.tokenizers.utils._h64`.
        * Otherwise, a compact in-house WL implementation is used that performs
          the refinement directly on the :class:`Molecule` object.

        :param G: Molecular graph to tokenize.
        :type G: Molecule
        :param radius: Number of WL iterations (``k >= 0``). ``k=0`` returns
            only the initial atom-level subtree labels; higher values add
            increasingly larger neighbourhoods.
        :type radius: int
        :returns: Counter mapping integer subtree-hash tokens to their
            multiplicities.
        :rtype: collections.Counter
        :raises ValueError: If ``radius`` is negative.
        :raises RuntimeError: If ``use_nx=True``, ``require_networkx=True``
            and NetworkX WL hashing is not available.
        """
        # Basic validation via BaseTokenizer (type/radius checks, etc.).
        super().tokens_graph(G, radius)

        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius!r}")

        # NetworkX backend (if requested and available)
        if self.use_nx:
            if not _HAVE_NX_WL:
                if self.require_networkx:
                    raise RuntimeError(
                        "NetworkX WL hashing requested (use_nx=True) "
                        "but not available."
                    )
                # fall back silently
            else:
                return self._tokens_graph_networkx(G, radius)

        # Fallback / default: in-house WL implementation
        return self._tokens_graph_inhouse(G, radius)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_graph_inhouse(self, G: Molecule, radius: int) -> Counter:
        """
        In-house WL subtree hashing implementation (original version).

        This method implements the edge-aware WL refinement loop directly on
        the :class:`Molecule` adjacency structure using :func:`_h64`,
        :func:`atom_label_tuple`, and :func:`bond_label_tuple`.

        :param G: Molecular graph.
        :type G: Molecule
        :param radius: Number of WL iterations (>=0).
        :type radius: int
        :returns: Counter of integer subtree tokens.
        :rtype: collections.Counter
        """
        # k = 0: initial node labels (atomic labels)
        labels: Dict[NodeId, int] = {
            v: _h64(("a0",) + atom_label_tuple(G, v, self.node_attrs)) for v in G.nodes
        }
        out = Counter(labels.values())

        # k >= 1: WL refinement with edge-aware neighbour tuples
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

    def _tokens_graph_networkx(self, G: Molecule, radius: int) -> Counter:
        """
        NetworkX-based WL subtree hashing implementation.

        This method constructs a temporary :class:`networkx.Graph` that mirrors
        the :class:`Molecule`'s nodes and edges, attaches precomputed atom and
        bond labels, and calls NetworkX's WL subgraph hashing. The resulting
        hex hashes are folded into integer tokens using :func:`_h64`.

        :param G: Molecular graph.
        :type G: Molecule
        :param radius: Number of WL iterations (>=0).
        :type radius: int
        :returns: Counter of integer subtree tokens derived from NetworkX WL.
        :rtype: collections.Counter
        """
        if not _HAVE_NX_WL:
            # Should not happen if guarded by caller, but keep defensive.
            return self._tokens_graph_inhouse(G, radius)

        nx_graph = nx.Graph()

        # Attach composed node labels as a stable string representation.
        for v in G.nodes:
            lbl = atom_label_tuple(G, v, self.node_attrs)
            nx_graph.add_node(v, __synrfp_label__=repr(lbl))

        # Attach edge labels (if any) while avoiding duplicate undirected edges.
        if self.edge_attrs:
            seen = set()
            for v in G.nodes:
                for u in G.adj.get(v, []):
                    if (u, v) in seen or (v, u) in seen:
                        continue
                    seen.add((v, u))
                    bl = bond_label_tuple(G, v, u, self.edge_attrs)
                    nx_graph.add_edge(v, u, __synrfp_edge__=repr(bl))

        node_attr_name = "__synrfp_label__"
        edge_attr_name = "__synrfp_edge__" if self.edge_attrs else None

        # include_initial_labels=True -> depth 0..radius
        node_hash_lists = _wl_hashes(
            nx_graph,
            node_attr=node_attr_name,
            edge_attr=edge_attr_name,
            iterations=radius,
            include_initial_labels=True,
        )

        out: Counter = Counter()
        for _, hex_hashes in node_hash_lists.items():
            for depth_idx, hex_hash in enumerate(hex_hashes):
                # Map hex string into integer token using your global hasher.
                tok = _h64(("nxwl", depth_idx, hex_hash))
                out[tok] += 1
        return out
