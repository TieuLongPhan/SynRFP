# /mnt/data/graph_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx

# Basic aliases for readability
NodeId = int
Edge = Tuple[NodeId, NodeId]


@dataclass
class Molecule:
    """
    Lightweight labeled molecular graph container.

    Conceptually:
      - nodes: atoms with attributes (e.g. element, charge, aromatic)
      - edges: bonds with attributes (e.g. order, aromatic)

    :param nodes: Mapping from atom id to attribute dict.
    :type nodes: Dict[NodeId, Dict]
    :param edges: Mapping from bond tuple (u, v) with u<v to attribute dict.
    :type edges: Dict[Edge, Dict]
    :param _adj: Internal adjacency cache, computed lazily.
    :type _adj: Optional[Dict[NodeId, List[NodeId]]]
    """

    nodes: Dict[NodeId, Dict]
    edges: Dict[Edge, Dict]
    _adj: Optional[Dict[NodeId, List[NodeId]]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(cls, nodes: Dict[NodeId, Dict], edges: Dict[Edge, Dict]) -> Molecule:
        """
        Construct Molecule ensuring edge keys are ordered (u < v).

        :param nodes: Node attribute mapping.
        :param edges: Edge attribute mapping (may have u>v).
        :returns: Initialized Molecule.
        """
        normalized_edges: Dict[Edge, Dict] = {}
        for (u, v), attr in edges.items():
            a, b = (u, v) if u < v else (v, u)
            normalized_edges[(a, b)] = dict(attr)
        return cls(nodes=dict(nodes), edges=normalized_edges)

    @classmethod
    def from_nx_graph(cls, G: nx.Graph) -> Molecule:
        """
        Construct Molecule from a NetworkX Graph.

        :param G: NetworkX graph with node and edge attributes.
        :returns: Initialized Molecule.
        """
        nodes = {n: dict(G.nodes[n]) for n in G.nodes}
        edges: Dict[Edge, Dict] = {}
        for u, v, attr in G.edges(data=True):
            a, b = (u, v) if u < v else (v, u)
            edges[(a, b)] = dict(attr)
        return cls.from_dicts(nodes, edges)

    def to_nx_graph(self) -> nx.Graph:
        """
        Convert this Molecule back to a NetworkX Graph.

        :returns: networkx.Graph with the same nodes/edges/attributes.
        """
        G = nx.Graph()
        for n, attr in self.nodes.items():
            G.add_node(n, **attr)
        for (u, v), attr in self.edges.items():
            G.add_edge(u, v, **attr)
        return G

    def copy(self) -> Molecule:
        """
        Shallow copy of the molecule (attributes dicts are copied, not deep).

        :returns: New Molecule with copied node/edge mappings.
        """
        return Molecule.from_dicts(self.nodes, self.edges)

    # ------------------------------------------------------------------
    # Basic graph queries
    # ------------------------------------------------------------------
    @property
    def adj(self) -> Dict[NodeId, List[NodeId]]:
        """
        Lazily compute and cache adjacency list.

        :returns: Mapping from node id to sorted neighbor list.
        """
        if self._adj is None:
            A: Dict[NodeId, List[NodeId]] = defaultdict(list)
            for u, v in self.edges:
                A[u].append(v)
                A[v].append(u)
            self._adj = {k: sorted(vs) for k, vs in A.items()}
        return self._adj

    def degree(self, v: NodeId) -> int:
        """
        Get degree of node v.

        :param v: Node identifier.
        :returns: Degree count.
        """
        return len(self.adj.get(v, []))

    def edge_attr(self, u: NodeId, v: NodeId) -> Dict:
        """
        Retrieve attribute dict for edge (u, v).

        :param u: First node.
        :param v: Second node.
        :returns: Edge attributes.
        :raises KeyError: If edge not present.
        """
        a, b = (u, v) if u < v else (v, u)
        return self.edges[(a, b)]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        """Number of atoms (nodes)."""
        return len(self.nodes)

    @property
    def n_bonds(self) -> int:
        """Number of bonds (edges)."""
        return len(self.edges)

    def __repr__(self) -> str:
        """
        Representation showing number of nodes and edges.

        Tests expect the text 'Molecule(nodes=X, edges=Y)'.
        """
        return f"Molecule(nodes={len(self.nodes)}, edges={len(self.edges)})"


# ----------------------------------------------------------------------
# Backwards compatibility: GraphData is now just an alias for Molecule.
# All existing imports `from synrfp.graph.graph_data import GraphData`
# continue to work.
# ----------------------------------------------------------------------
GraphData = Molecule
