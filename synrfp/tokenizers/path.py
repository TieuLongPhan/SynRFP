# synrfp/tokenizers/path.py
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, List

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.molecule import Molecule, NodeId


class PathTokenizer(BaseTokenizer):
    """
    Path-based tokenizer over molecular graphs.

    This tokenizer enumerates simple paths in the :class:`Molecule` up to a
    maximum length (in bonds) and hashes their labelled sequences into integer
    tokens. Each path is represented by an alternating sequence of atom and
    bond labels; to avoid double-counting reverse directions, the path label
    is canonicalized by comparing the forward and reverse sequences and
    selecting the lexicographically smaller one before hashing.

    The number of bonds in a path (path length) is bounded by ``min_path`` and
    the ``radius`` argument to :meth:`tokens_graph`.

    It is inspired by the :class:`PathFPs` example, but instead of directly
    producing a fixed-length bitstring fingerprint, it returns a multiset of
    hashed path descriptors suitable for downstream sketching.
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        min_path: int = 1,
    ) -> None:
        """
        Initialize a :class:`PathTokenizer` instance.

        :param node_attrs: Node attributes to include in atom labels used for
            path encoding, e.g. ``['element', 'charge']``. If ``None``,
            :func:`atom_label_tuple` uses its default atom labelling scheme.
        :type node_attrs: Optional[Sequence[str]]
        :param edge_attrs: Edge attributes to include in bond labels used for
            path encoding, e.g. ``['order', 'aromatic']``. If ``None``,
            :func:`bond_label_tuple` uses its default bond labelling scheme.
        :type edge_attrs: Optional[Sequence[str]]
        :param min_path: Minimum path length in bonds to be considered when
            generating tokens. Paths shorter than ``min_path`` bonds are
            ignored. Must be a positive integer.
        :type min_path: int
        :raises ValueError: If ``min_path`` is not positive.
        """
        if min_path <= 0:
            raise ValueError("min_path must be a positive integer.")

        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)
        self.min_path: int = int(min_path)

    def __repr__(self) -> str:
        return (
            f"PathTokenizer(node_attrs={self.node_attrs}, "
            f"edge_attrs={self.edge_attrs}, min_path={self.min_path})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the :class:`PathTokenizer`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.tokenizers.path import PathTokenizer\n"
            "tok = PathTokenizer(node_attrs=['element'], edge_attrs=['order'])\n"
            "C = tok.tokens_graph(graph, radius=7)\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        """
        Tokenize a molecular graph using path-based features.

        This method enumerates simple paths in the graph and emits one token
        per labelled path. A path of nodes ``[v0, v1, ..., vL]`` has
        length ``L`` in bonds, and is considered if
        ``min_path <= L <= max_path``, where
        ``max_path = max(self.min_path, radius)``.

        For each eligible path, an alternating sequence of atom and bond labels
        is constructed:

        .. code-block:: text

            atom(v0), bond(v0,v1), atom(v1), ..., bond(vL-1,vL), atom(vL)

        The forward and reverse sequences are compared lexicographically; the
        smaller one is used as canonical representation and hashed via
        :func:`_h64` into an integer token.

        :param G: Molecular graph to tokenize.
        :type G: Molecule
        :param radius: Effective maximum path length in bonds. A value of
            ``radius <= 0`` is treated as ``radius = min_path``.
        :type radius: int
        :returns: Counter mapping integer path-hash tokens to counts.
        :rtype: collections.Counter
        :raises ValueError: If ``radius`` is negative.
        """
        # Base validation (type & radius sanity checks).
        super().tokens_graph(G, radius)

        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius!r}")

        max_path = max(self.min_path, radius if radius > 0 else self.min_path)
        tokens: Counter = Counter()

        # DFS from each node to enumerate simple paths up to max_path bonds.
        for start in G.nodes:
            self._dfs_paths(
                G=G,
                current=start,
                parent=None,
                path=[start],
                depth=0,
                max_path=max_path,
                tokens=tokens,
            )

        return tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dfs_paths(
        self,
        G: Molecule,
        current: NodeId,
        parent: Optional[NodeId],
        path: List[NodeId],
        depth: int,
        max_path: int,
        tokens: Counter,
    ) -> None:
        """
        Depth-first enumeration of simple paths for internal use.

        :param G: Molecular graph.
        :type G: Molecule
        :param current: Current node at the end of the path.
        :type current: NodeId
        :param parent: Previous node in the path (used to avoid immediate
            backtracking). ``None`` for the starting node.
        :type parent: Optional[NodeId]
        :param path: Current path as a list of node IDs.
        :type path: List[NodeId]
        :param depth: Current path length in bonds (``len(path) - 1``).
        :type depth: int
        :param max_path: Maximum path length in bonds.
        :type max_path: int
        :param tokens: Counter to update with path tokens.
        :type tokens: collections.Counter
        """
        # depth == number of bonds = len(path) - 1
        if depth >= self.min_path:
            self._emit_path_token(G, path, tokens)

        if depth >= max_path:
            return

        v = current
        for u in G.adj.get(v, []):
            # avoid immediate backtracking and cycles (simple paths only)
            if parent is not None and u == parent:
                continue
            if u in path:
                continue

            path.append(u)
            self._dfs_paths(
                G=G,
                current=u,
                parent=v,
                path=path,
                depth=depth + 1,
                max_path=max_path,
                tokens=tokens,
            )
            path.pop()

    def _emit_path_token(
        self, G: Molecule, path: List[NodeId], tokens: Counter
    ) -> None:
        """
        Build a canonical labelled representation of a path and update tokens.

        :param G: Molecular graph.
        :type G: Molecule
        :param path: Simple path as a list of node IDs (length >= 2).
        :type path: List[NodeId]
        :param tokens: Counter to update.
        :type tokens: collections.Counter
        """
        # Forward label sequence: atom, (bond, atom)*
        seq_fwd = []
        # first atom
        seq_fwd.append(atom_label_tuple(G, path[0], self.node_attrs))
        for i in range(len(path) - 1):
            v = path[i]
            u = path[i + 1]
            seq_fwd.append(bond_label_tuple(G, v, u, self.edge_attrs))
            seq_fwd.append(atom_label_tuple(G, u, self.node_attrs))

        # Reverse path label sequence
        rev_nodes = list(reversed(path))
        seq_rev = []
        seq_rev.append(atom_label_tuple(G, rev_nodes[0], self.node_attrs))
        for i in range(len(rev_nodes) - 1):
            v = rev_nodes[i]
            u = rev_nodes[i + 1]
            seq_rev.append(bond_label_tuple(G, v, u, self.edge_attrs))
            seq_rev.append(atom_label_tuple(G, u, self.node_attrs))

        # Canonicalize orientation
        canonical_seq = tuple(seq_fwd) if seq_fwd <= seq_rev else tuple(seq_rev)

        depth = len(path) - 1  # number of bonds
        tok = _h64(("path", depth, canonical_seq))
        tokens[tok] += 1
