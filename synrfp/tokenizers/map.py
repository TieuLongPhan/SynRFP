# synrfp/tokenizers/map.py
from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Optional, Sequence, Tuple

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.molecule import Molecule, NodeId


class MAPTokenizer(BaseTokenizer):
    """
    MAP-style circular atom-pair tokenizer (MinHashed Atom-Pair precursor).

    This tokenizer integrates circular (Morgan-like) environments with
    atom-pair distances in a MAP4-style fashion:

    1. Build circular environments around each atom via iterative refinement
       up to `radius` (analogous to ECFP / Morgan).
    2. Compute shortest-path distances d_ij between atoms i and j up to
       a cutoff `max_distance = distance_factor * radius`.
    3. For each atom pair (i, j), each environment radius r = 1..radius, and
       each valid distance d_ij, construct a "shingle":

           (r, env_lo, d_ij, env_hi)

       where env_lo/env_hi are the circular environments of i and j at
       radius r in sorted order (canonicalized).
    4. Hash shingles into 64-bit integer tokens via :func:`_h64`.

    These tokens are then intended to be consumed by SynRFP sketchers
    (e.g. MinHashSketch, CWSketch, parity fold) to produce MAP4-like
    fingerprints.

    Example (MAP4-like):
        >>> from synrfp.tokenizers.map import MAPTokenizer
        >>> tok = MAPTokenizer(distance_factor=2)  # distance <= 2 * radius
        >>> C = tok.tokens_graph(graph, radius=2)  # radius=2 -> distance<=4

    :param node_attrs: Node attributes used to build initial atom labels
        (e.g. ``['element', 'charge', 'hcount']``). If ``None``, the default
        atom label tuple defined in :mod:`synrfp.tokenizers.utils` is used.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge/bond attributes used in the circular refinement
        (e.g. ``['order', 'aromatic']``). If ``None``, a default bond label
        tuple is used.
    :type edge_attrs: Optional[Sequence[str]]
    :param distance_factor: Multiplier that controls the maximum atom-pair
        distance as ``max_distance = distance_factor * radius``. For MAP4-like
        behaviour, set ``distance_factor=2`` and call with ``radius=2``.
    :type distance_factor: int
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        distance_factor: int = 2,
    ) -> None:
        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)
        if distance_factor < 1:
            raise ValueError(f"distance_factor must be >= 1, got {distance_factor!r}")
        self.distance_factor: int = int(distance_factor)

    def __repr__(self) -> str:
        return (
            f"MAPTokenizer(node_attrs={self.node_attrs}, "
            f"edge_attrs={self.edge_attrs}, "
            f"distance_factor={self.distance_factor})"
        )

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for the :class:`MAPTokenizer`.

        :return: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.tokenizers.map import MAPTokenizer\n"
            "# MAP4-like: radius=2, max pair distance=4\n"
            "tok = MAPTokenizer(distance_factor=2)\n"
            "C = tok.tokens_graph(graph, radius=2)\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        """
        Tokenize a molecular graph via circular environments + atom pairs.

        This method produces MAP-style atom-pair tokens:

            (r, env_lo, d_ij, env_hi)  ->  int token via _h64

        where:
        * r      = circular radius (1..radius),
        * env_lo/env_hi are hashed circular environments of the two atoms,
        * d_ij   = shortest-path distance between the atoms, with
                   1 <= d_ij <= distance_factor * radius.

        Atoms without neighbours (ions, water, halides) instead contribute one
        ``(r, env)`` token per radius, so they are not lost.

        :param G: Molecular graph to tokenize.
        :type G: Molecule
        :param radius: Maximum circular radius (>=1). For MAP4-like behaviour,
            use ``radius=2`` (and ``distance_factor=2``).
        :type radius: int
        :return: Counter mapping integer MAP-style tokens to multiplicities.
        :rtype: collections.Counter
        :raises ValueError: If ``radius`` is not positive.
        """
        # Basic validation via BaseTokenizer (type/radius checks, etc.).
        super().tokens_graph(G, radius)

        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius!r}")

        return self._tokens_graph_inhouse(G, radius)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_graph_inhouse(self, G: Molecule, radius: int) -> Counter:
        """
        In-house MAP-style circular atom-pair implementation.

        1. Build circular node environments up to the given radius.
        2. Compute shortest-path distances up to `max_distance`.
        3. For each radius layer and each atom pair within cutoff, generate
           a MAP-style shingle and hash it into a token.

        :param G: Molecular graph.
        :type G: Molecule
        :param radius: Maximum circular radius (>=1).
        :type radius: int
        :return: Counter of integer MAP-style tokens.
        :rtype: collections.Counter
        """
        # --------------------------------------------------------------
        # 1) Base atom invariants (radius 0)
        # --------------------------------------------------------------
        labels: Dict[NodeId, int] = {
            v: _h64(("map0",) + atom_label_tuple(G, v, self.node_attrs))
            for v in G.nodes
        }

        # env[v][k-1] holds the environment of atom v at circular radius k
        env: Dict[NodeId, list[int]] = {v: [] for v in G.nodes}

        # --------------------------------------------------------------
        # 2) Circular refinement for k = 1..radius (Morgan-like)
        # --------------------------------------------------------------
        for k in range(1, radius + 1):
            new_labels: Dict[NodeId, int] = {}
            for v in G.nodes:
                neigh = []
                for u in G.adj.get(v, []):
                    if self.edge_attrs:
                        bl = _h64(
                            ("mapb",) + bond_label_tuple(G, v, u, self.edge_attrs)
                        )
                    else:
                        # No edge attributes: every bond gets the same label. It
                        # must not depend on the node ids, or the tokens would
                        # change with atom numbering.
                        bl = _h64(("mapb0",))
                    neigh.append((bl, labels[u]))
                neigh.sort()
                # Circular environment label at layer k
                new_labels[v] = _h64(("map_env", k, labels[v], tuple(neigh)))
            labels = new_labels
            for v in G.nodes:
                env[v].append(labels[v])

        # --------------------------------------------------------------
        # 3) Shortest-path distances up to max_distance
        # --------------------------------------------------------------
        max_distance: int = max(1, self.distance_factor * radius)
        distances: Dict[Tuple[NodeId, NodeId], int] = {}

        for start in G.nodes:
            q = deque([(start, 0)])
            seen = {start}
            while q:
                v, d = q.popleft()
                if d >= max_distance:
                    continue
                for u in G.adj.get(v, []):
                    if u in seen:
                        continue
                    seen.add(u)
                    d_new = d + 1
                    if 1 <= d_new <= max_distance:
                        a, b = (start, u) if start < u else (u, start)
                        # keep the smallest distance if multiple paths exist
                        prev = distances.get((a, b))
                        if prev is None or d_new < prev:
                            distances[(a, b)] = d_new
                    q.append((u, d_new))

        # --------------------------------------------------------------
        # 4) Build MAP-style shingles and hash to integer tokens
        # --------------------------------------------------------------
        out: Counter = Counter()
        # Isolated atoms (ions, water, ...) have no pair partner; give each
        # one a distance-0 shingle so it is not dropped from the multiset.
        for v in G.nodes:
            if G.degree(v) == 0:
                for idx, r in enumerate(range(1, radius + 1)):
                    out[_h64(("map_atom", r, env[v][idx]))] += 1
        # env[v] has length `radius`, indexed as 0..radius-1 corresponding
        # to circular radii r = 1..radius
        for (i, j), d_ij in distances.items():
            env_i = env[i]
            env_j = env[j]
            for idx, r in enumerate(range(1, radius + 1)):
                e_i = env_i[idx]
                e_j = env_j[idx]
                if e_i <= e_j:
                    env_lo, env_hi = e_i, e_j
                else:
                    env_lo, env_hi = e_j, e_i
                tok = _h64(("map_pair", r, d_ij, env_lo, env_hi))
                out[tok] += 1

        return out
