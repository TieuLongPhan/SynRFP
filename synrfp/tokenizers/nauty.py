# synrfp/tokenizers/nauty.py
from __future__ import annotations

import hashlib
import logging
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64, atom_label_tuple, bond_label_tuple
from synrfp.graph.molecule import Molecule, NodeId

logger = logging.getLogger(__name__)


class NautyCanonicalizer:
    """
    Nauty-style canonicalizer implemented with NetworkX primitives.

    This class computes a canonical labeling / signature for a NetworkX graph
    using equitable partition refinement + backtracking search similar to
    the nauty algorithm. It's designed to be reasonably robust for small ego
    subgraphs (typical use-case: ego-subgraphs extracted around a node).

    :param node_attrs: node attribute keys used for initial partitioning/refinement.
    :type node_attrs: list[str] | None
    :param edge_attrs: edge attribute keys used when including edge attributes
        in the canonical label.
    :type edge_attrs: list[str] | None
    """

    __slots__ = ("node_attrs", "edge_attrs")

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
    ) -> None:
        self.node_attrs = list(node_attrs) if node_attrs else []
        self.edge_attrs = list(edge_attrs) if edge_attrs else []

    # ----- helpers for freezing arbitrary values into comparable forms -----
    @staticmethod
    def _freeze(x):
        if isinstance(x, list):
            return tuple(NautyCanonicalizer._freeze(v) for v in x)
        if isinstance(x, dict):
            return frozenset(
                (k, NautyCanonicalizer._freeze(v)) for k, v in sorted(x.items())
            )
        return x

    # ----- canonicalization entry point -----
    def canonical_form(
        self,
        G: nx.Graph,
        return_aut: bool = False,
        remap_aut: bool = False,
        return_orbits: bool = False,
        return_perm: bool = False,
        max_depth: Optional[int] = None,
    ):
        """
        Compute canonical form of G.

        By default returns canonicalized graph `G_can`. Optionally can return
        permutation, automorphisms, orbits, and early-stop flag.

        The algorithm:
        - Build initial partition from node_attrs (or single cell if none)
        - Repeatedly refine partition by node signatures until stable
        - If partition refined to singletons, build label and update best
        - Otherwise pick a non-singleton cell and branch (backtracking) until
          best canonical label is found.

        This implementation is primarily intended for small graphs (ego subgraphs).
        """
        best = {"label": None, "perm": None}
        aut_perms: List[List[int]] = []

        initial_partition = self._initial_partition(G)
        early_stop_flag = self._search(
            G, initial_partition, [], best, aut_perms, depth=0, max_depth=max_depth
        )

        perm = best["perm"]
        if perm is None:
            # Search could have been stopped early or failed
            raise RuntimeError(
                "Canonical form not found (search stopped early or failed)."
            )

        # Build canonical graph (relabel nodes 1..n according to perm)
        mapping = {v: i + 1 for i, v in enumerate(perm)}
        G_can = nx.relabel_nodes(G, mapping, copy=True)

        results: List = [G_can]
        if return_perm:
            results.append(perm)
        if return_aut:
            if remap_aut:
                remapped = [[mapping[v] for v in p] for p in aut_perms]
                results.append(remapped)
            else:
                results.append(aut_perms)
        if return_orbits:
            orbits = self.compute_orbits(aut_perms)
            if remap_aut and return_aut:
                orbits = [set(mapping[v] for v in orbit) for orbit in orbits]
            results.append(orbits)

        results.append(early_stop_flag)
        return tuple(results) if len(results) > 2 else results[0]

    # create an initial partition (buckets) based on node_attrs or single cell
    def _initial_partition(self, G: nx.Graph) -> List[List[int]]:
        if not self.node_attrs:
            return [sorted(G.nodes())]
        buckets: Dict[Tuple, List[int]] = {}
        for v in G.nodes():
            key = tuple(
                self._freeze(G.nodes[v].get(attr, None)) for attr in self.node_attrs
            )
            buckets.setdefault(key, []).append(v)
        # sort buckets deterministically
        return [sorted(nodes) for _, nodes in sorted(buckets.items())]

    # compute signature for a node relative to current partition
    def _node_signature(self, G: nx.Graph, v: int, partition: List[List[int]]):
        node_attrs = tuple(
            self._freeze(G.nodes[v].get(a, None)) for a in self.node_attrs
        )
        degree = G.degree[v]
        # counts of neighbors falling into each cell
        nbr_part_counts = tuple(
            sum(1 for nbr in G.neighbors(v) if nbr in cell) for cell in partition
        )

        # build sorted multiset of edge attribute tuples
        edge_attr_multiset = []
        for nbr in G.neighbors(v):
            attrs = G[v][nbr]
            edge_attrs_vals = []
            for a in self.edge_attrs:
                val = attrs.get(a, None)
                # numeric normalization for 'order' tuples if present (compatibility)
                if a == "order" and isinstance(val, tuple):
                    try:
                        val = tuple(sorted(round(float(x), 3) for x in val))
                    except Exception:
                        pass
                edge_attrs_vals.append(self._freeze(val))
            edge_attr_multiset.append(tuple(edge_attrs_vals))
        edge_attr_multiset = tuple(sorted(edge_attr_multiset))

        return (node_attrs, degree, nbr_part_counts, edge_attr_multiset)

    # equitable partition refinement
    def _refine(self, G: nx.Graph, partition: List[List[int]]):
        changed = True
        while changed:
            changed = False
            new_partition: List[List[int]] = []
            sig_cache: Dict[int, object] = {}
            for cell in partition:
                if len(cell) <= 1:
                    new_partition.append(cell)
                    continue
                sigs: Dict[object, List[int]] = {}
                for v in cell:
                    if v not in sig_cache:
                        sig_cache[v] = self._node_signature(G, v, partition)
                    sig = sig_cache[v]
                    sigs.setdefault(sig, []).append(v)
                if len(sigs) > 1:
                    changed = True
                    for sig in sorted(sigs):
                        new_partition.append(sorted(sigs[sig]))
                else:
                    new_partition.append(cell)
            partition = new_partition
        return partition

    # backtracking search to produce canonical permutation(s)
    def _search(
        self,
        G: nx.Graph,
        partition: List[List[int]],
        prefix: List[int],
        best: Dict,
        aut_perms: List[List[int]],
        depth=0,
        max_depth: Optional[int] = None,
    ):
        if max_depth is not None and depth > max_depth:
            logger.debug("Early stop in canonical search due to depth limit.")
            return True  # early stop flag

        partition = self._refine(G, partition)
        # if fully refined to singletons -> candidate permutation
        if all(len(c) == 1 for c in partition):
            perm = prefix + [v for c in partition for v in c]
            label = self._build_label(G, perm)
            if best["label"] is None or label < best["label"]:
                best["label"], best["perm"] = label, perm
                aut_perms.clear()
                aut_perms.append(perm)
                logger.debug("Found new best canonical label.")
            elif label == best["label"]:
                aut_perms.append(perm)
                logger.debug("Found equivalent canonical label (automorphism).")
            return False  # did not early stop
        # choose first non-singleton cell
        idx = next(i for i, c in enumerate(partition) if len(c) > 1)
        cell = partition[idx]
        sorted_cell = sorted(cell, key=lambda n: G.nodes[n].get("atom_map", n))
        for v in sorted_cell:
            rest = [w for w in cell if w != v]
            new_partition = (
                partition[:idx]
                + [[v]]
                + ([sorted(rest)] if rest else [])
                + partition[idx + 1 :]  # noqa
            )
            candidate_prefix = prefix + [v]
            partial_label = self._build_partial_label(G, candidate_prefix)
            if best["label"] is not None and partial_label > best["label"]:
                continue  # prune
            if self._search(
                G,
                new_partition,
                candidate_prefix,
                best,
                aut_perms,
                depth=depth + 1,
                max_depth=max_depth,
            ):
                return True  # propagate early stop
        return False

    # build canonical string label for a permutation
    def _build_label(self, G: nx.Graph, perm: List[int]) -> str:
        node_segment = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(attr, ""))) for attr in self.node_attrs
            )
            for v in perm
        )
        n = len(perm)
        edge_bits = []
        for i in range(n):
            vi = perm[i]
            for j in range(i + 1, n):
                vj = perm[j]
                if G.has_edge(vi, vj):
                    attrs = G[vi][vj]
                    frozen_attrs = tuple(
                        self._freeze(attrs.get(a, "")) for a in self.edge_attrs
                    )
                    edge_bits.append("1:" + ":".join(str(x) for x in frozen_attrs))
                else:
                    edge_bits.append("0:" + ":".join("" for _ in self.edge_attrs))
        edge_segment = "|".join(edge_bits)
        return node_segment + "||" + edge_segment

    def _build_partial_label(self, G: nx.Graph, prefix: List[int]) -> str:
        node_segment = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(attr, ""))) for attr in self.node_attrs
            )
            for v in prefix
        )
        # suffix larger than any label to allow pruning by lexical comparison
        suffix = "{" * 1000
        return node_segment + suffix

    def compute_orbits(self, aut_perms: List[List[int]]):
        if not aut_perms:
            return []
        orbit_map = {}
        orbits = []

        def union_orbits(i, j):
            if i == j:
                return
            o1 = orbits[i]
            o2 = orbits[j]
            if len(o1) < len(o2):
                i, j = j, i
                o1, o2 = o2, o1
            o1.update(o2)
            orbits[j] = set()
            for v in o2:
                orbit_map[v] = i

        first_perm = aut_perms[0]
        for idx, node in enumerate(first_perm):
            orbit_map[node] = idx
            orbits.append({node})
        for perm in aut_perms:
            for idx, node in enumerate(perm):
                union_orbits(idx, orbit_map[node])
        return [o for o in orbits if o]

    def graph_signature(self, G: nx.Graph) -> str:
        # canonical_form + label -> hashed signature
        G_can = self.canonical_form(G)
        label = self._build_label(G_can, sorted(G_can.nodes()))
        return hashlib.sha256(label.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# NautyTokenizer: uses NautyCanonicalizer on ego-subgraphs built from Molecule
# ---------------------------------------------------------------------------
class NautyTokenizer(BaseTokenizer):
    """
    Nauty-style canonical ego-subgraph tokenizer using pure NetworkX canonicalizer.

    For each center node and each radius 0..r, the ego subgraph is canonicalized
    with respect to chosen node/edge attributes and the canonical signature is
    converted to an integer token via :func:`_h64`.

    :param node_attrs: list of node attribute keys to include in initial partitioning.
    :type node_attrs: list[str] | None
    :param edge_attrs: list of edge attribute keys to include for edge distinctions.
    :type edge_attrs: list[str] | None
    :param max_cache: maximum number of ego-canonicalizations cached
    (simple LRU via dict).
    :type max_cache: int
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        max_cache: int = 100000,
    ) -> None:
        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)
        self.canon = NautyCanonicalizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
        self._cache: Dict[frozenset, int] = {}
        self._max_cache = int(max_cache)

    def __repr__(self) -> str:
        return (
            f"NautyTokenizer(node_attrs={self.node_attrs},"
            + " edge_attrs={self.edge_attrs})"
        )

    def _ego_nodes(self, G: Molecule, center: NodeId, r: int) -> List[NodeId]:
        """BFS collect nodes up to radius r (inclusive) around center."""
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

    def _build_nx_ego(self, G: Molecule, nodes: List[NodeId]) -> nx.Graph:
        """Build a networkx.Graph for the ego-subgraph with
        stable node/edge attributes."""
        H = nx.Graph()
        # Add nodes with stable attributes (use repr of atom_label_tuple)
        for v in nodes:
            lbl = atom_label_tuple(G, v, self.node_attrs)
            H.add_node(v, synrfp_label=repr(lbl))
        # Add undirected edges with synrfp edge labels
        seen = set()
        for v in nodes:
            for u in G.adj.get(v, []):
                if u in nodes and (u, v) not in seen and (v, u) not in seen:
                    bl = bond_label_tuple(G, v, u, self.edge_attrs)
                    H.add_edge(v, u, synrfp_edge=repr(bl))
                    seen.add((v, u))
        return H

    def _canonical_token_for_nodes(self, G: Molecule, nodes: List[NodeId]) -> int:
        """Return integer token for canonical form of ego subgraph induced by `nodes`."""
        key = frozenset(nodes)
        if key in self._cache:
            return self._cache[key]

        H = self._build_nx_ego(G, nodes)

        # compute compact signature via canonicalizer (uses node_attrs/edge_attrs)
        try:
            sig = self.canon.graph_signature(H)
        except Exception as exc:
            # fallback deterministic hashing if canonicalizer fails
            logger.debug("Nauty canonicalization failed; falling back: %s", exc)
            # Build deterministic tuple for fallback
            node_colors = tuple(atom_label_tuple(G, v, self.node_attrs) for v in nodes)
            edge_colors = []
            for v in nodes:
                for u in G.adj.get(v, []):
                    if u in nodes and v < u:
                        bl = bond_label_tuple(G, v, u, self.edge_attrs)
                        edge_colors.append((nodes.index(v), nodes.index(u), bl))
            fallback_sig = _h64(("fallback", node_colors, tuple(edge_colors)))
            tok = fallback_sig
            # Cache and return
            if len(self._cache) >= self._max_cache:
                # simple eviction: pop first inserted
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = tok
            return tok

        # fold hex sig (sha256 hex) into numeric via _h64 for consistency
        tok = _h64(("nauty", sig))
        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = tok
        return tok

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        """
        Tokenize a molecular graph by canonicalizing ego-subgraphs up to `radius`.

        :param G: Molecular graph.
        :type G: Molecule
        :param radius: maximum radius (inclusive) for ego-subgraphs.
        :type radius: int
        :returns: Counter mapping canonical integer tokens to counts.
        :rtype: collections.Counter
        """
        super().tokens_graph(G, radius)

        out: Counter = Counter()
        for v in G.nodes:
            for k in range(0, radius + 1):
                ego_nodes = self._ego_nodes(G, v, k)
                if not ego_nodes:
                    continue
                tok = self._canonical_token_for_nodes(G, ego_nodes)
                out[tok] += 1
        return out
