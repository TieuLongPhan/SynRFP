from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx

NodeId = int
Edge = Tuple[NodeId, NodeId]


@dataclass
class GraphData:
    """
    Lightweight labeled graph container.

    :param nodes: Mapping from node id to attribute dict (e.g., element, charge).
    :type nodes: Dict[NodeId, Dict]
    :param edges: Mapping from edge tuple (u, v) with u<v to attribute dict (e.g., order).
    :type edges: Dict[Edge, Dict]
    :param _adj: Internal adjacency cache, computed lazily.
    :type _adj: Optional[Dict[NodeId, List[NodeId]]]
    """

    nodes: Dict[NodeId, Dict]
    edges: Dict[Edge, Dict]
    _adj: Optional[Dict[NodeId, List[NodeId]]] = None

    @staticmethod
    def from_dicts(nodes: Dict[NodeId, Dict], edges: Dict[Edge, Dict]) -> GraphData:
        """
        Construct GraphData ensuring edge keys are ordered (u < v).

        :param nodes: Node attribute mapping.
        :type nodes: Dict[int, Dict]
        :param edges: Edge attribute mapping.
        :type edges: Dict[Tuple[int, int], Dict]
        :returns: Initialized GraphData.
        :rtype: GraphData
        """
        normalized_edges: Dict[Edge, Dict] = {}
        for (u, v), attr in edges.items():
            a, b = (u, v) if u < v else (v, u)
            normalized_edges[(a, b)] = dict(attr)
        return GraphData(nodes=dict(nodes), edges=normalized_edges)

    @staticmethod
    def from_nx_graph(G: nx.Graph) -> GraphData:
        """
        Construct GraphData from a NetworkX Graph.

        :param G: NetworkX graph with node and edge attributes.
        :type G: nx.Graph
        :returns: Initialized GraphData.
        :rtype: GraphData
        """
        nodes = {n: dict(G.nodes[n]) for n in G.nodes}
        edges: Dict[Edge, Dict] = {}
        for u, v, attr in G.edges(data=True):
            a, b = (u, v) if u < v else (v, u)
            edges[(a, b)] = dict(attr)
        return GraphData.from_dicts(nodes, edges)

    @property
    def adj(self) -> Dict[NodeId, List[NodeId]]:
        """
        Lazily compute and cache adjacency list.

        :returns: Mapping from node id to sorted neighbor list.
        :rtype: Dict[int, List[int]]
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
        :type v: int
        :returns: Degree count.
        :rtype: int
        """
        return len(self.adj.get(v, []))

    def edge_attr(self, u: NodeId, v: NodeId) -> Dict:
        """
        Retrieve attribute dict for edge (u, v).

        :param u: First node.
        :type u: int
        :param v: Second node.
        :type v: int
        :returns: Edge attributes.
        :rtype: Dict
        :raises KeyError: If edge not present.
        """
        a, b = (u, v) if u < v else (v, u)
        return self.edges[(a, b)]

    def __repr__(self) -> str:
        """
        Representation showing number of nodes and edges.

        :returns: String repr.
        :rtype: str
        """
        return f"GraphData(nodes={len(self.nodes)}, edges={len(self.edges)})"
