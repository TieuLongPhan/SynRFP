# synrfp/tokenizers/morgan.py
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.molecule import Molecule, NodeId


class MorganTokenizer(BaseTokenizer):
    """
    Morgan-style circular neighborhood tokenizer (ECFP-like).

    This tokenizer implements an extended-connectivity / Morgan-style
    neighborhood expansion directly on the :class:`~synrfp.graph.molecule.Molecule`
    graph, without relying on RDKit. It is conceptually similar to ECFP:

    * Each atom starts with an initial invariant derived from
      :func:`atom_label_tuple`.
    * At iteration ``k``, each atom aggregates its own previous label and a
      multiset of (bond-label, neighbour-label) tuples from its 1-hop
      neighbourhood (using :func:`bond_label_tuple`).
    * The resulting environment is hashed into a new integer identifier using
      :func:`_h64`.

    All identifiers (for radii ``0..radius``) are emitted as tokens.
    Compared to :class:`WLTokenizer`, this implementation uses different
    internal tags (``\"m0\"``, ``\"mb\"``, ``\"morgan\"``) but otherwise
    follows a very similar refinement scheme.

    :example:
        >>> tok = MorganTokenizer(node_attrs=['element'], edge_attrs=['order'])
        >>> isinstance(tok, MorganTokenizer)
        True
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize a :class:`MorganTokenizer` instance.

        :param node_attrs: Node attributes to include in the initial atom
            invariants, e.g. ``['element', 'charge', 'hcount']``. If ``None``,
            :func:`atom_label_tuple` applies its default atom labelling scheme.
        :type node_attrs: Optional[Sequence[str]]
        :param edge_attrs: Edge attributes to include in bond labels, e.g.
            ``['order', 'aromatic']``. If ``None``, :func:`bond_label_tuple`
            applies its default bond labelling scheme.
        :type edge_attrs: Optional[Sequence[str]]
        """
        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)

    def __repr__(self) -> str:
        return (
            f"MorganTokenizer(node_attrs={self.node_attrs}, "
            f"edge_attrs={self.edge_attrs})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the :class:`MorganTokenizer`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.tokenizers.morgan import MorganTokenizer\n"
            "tok = MorganTokenizer(node_attrs=['element'], edge_attrs=['order'])\n"
            "C = tok.tokens_graph(graph, radius=2)\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        """
        Tokenize a molecular graph using a Morgan-style circular algorithm.

        At radius ``0``, each atom contributes a token corresponding to its
        initial invariant. For each subsequent iteration ``k = 1..radius``,
        atom environments are updated by combining the previous atom label with
        a sorted multiset of (bond-label, neighbour-label) pairs. All
        environment identifiers are accumulated as tokens.

        This is analogous in spirit to :class:`WLTokenizer`, but follows the
        “extended connectivity” intuition of Morgan fingerprints.

        :param G: Molecular graph to tokenize.
        :type G: Molecule
        :param radius: Number of Morgan refinement iterations (``k >= 0``).
            ``radius=0`` yields only atom-level invariants; larger radii
            incorporate progressively larger circular neighborhoods.
        :type radius: int
        :returns: Counter mapping integer environment identifiers to counts.
        :rtype: collections.Counter
        :raises ValueError: If ``radius`` is negative.
        """
        # Basic validation via BaseTokenizer (type & radius sanity checks).
        super().tokens_graph(G, radius)

        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius!r}")

        # k = 0: initial atom invariants
        labels: Dict[NodeId, int] = {
            v: _h64(("m0",) + atom_label_tuple(G, v, self.node_attrs)) for v in G.nodes
        }
        tokens = Counter(labels.values())

        # k >= 1: circular refinement
        for k in range(1, radius + 1):
            new_labels: Dict[NodeId, int] = {}
            for v in G.nodes:
                neigh = []
                for u in G.adj.get(v, []):
                    bl = _h64(("mb",) + bond_label_tuple(G, v, u, self.edge_attrs))
                    neigh.append((bl, labels[u]))
                neigh.sort()
                env_id = _h64(("morgan", k, labels[v], tuple(neigh)))
                new_labels[v] = env_id
            labels = new_labels
            tokens.update(labels.values())

        return tokens
