# ------------------------------------------
# synrfp/tokenizers/utils.py
from hashlib import blake2b
from typing import Tuple, List
from synrfp.graph.graph_data import GraphData, NodeId


def _h64(x) -> int:
    """
    Compute a stable 64-bit hash of the input using BLAKE2b.

    :param x: Any hashable object (e.g., tuple of labels).
    :type x: object
    :returns: 64-bit integer hash value.
    :rtype: int
    """
    h = blake2b(digest_size=8)
    h.update(repr(x).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def atom_label_tuple(G: GraphData, v: NodeId, node_attrs: List[str]) -> Tuple:
    """
    Build a node label tuple from selected node attributes and degree.

    :param G: GraphData instance containing node data.
    :type G: GraphData
    :param v: Node identifier.
    :type v: NodeId
    :param node_attrs: List of attribute keys to include.
    :type node_attrs: List[str]
    :returns: Tuple of attribute values followed by node degree.
    :rtype: Tuple
    """
    values = [G.nodes[v].get(attr) for attr in node_attrs]
    values.append(G.degree(v))
    return tuple(values)


def bond_label_tuple(
    G: GraphData, u: NodeId, v: NodeId, edge_attrs: List[str]
) -> Tuple:
    """
    Build an edge label tuple from selected edge attributes.

    :param G: GraphData instance containing edge data.
    :type G: GraphData
    :param u: First node of the edge.
    :type u: NodeId
    :param v: Second node of the edge.
    :type v: NodeId
    :param edge_attrs: List of attribute keys to include.
    :type edge_attrs: List[str]
    :returns: Tuple of attribute values.
    :rtype: Tuple
    """
    return tuple(G.edge_attr(u, v).get(attr) for attr in edge_attrs)
