# synrfp/tokenizers/utils.py
from __future__ import annotations

from hashlib import blake2b
from typing import Tuple, List, Any, Iterable

from synrfp.graph.graph_data import GraphData, NodeId


def _h64(obj: Any, *, seed: int = 0) -> int:
    """
    Stable 64-bit hash using BLAKE2b.

    :param obj: Any printable object (e.g., tuple of labels).
    :type obj: Any
    :param seed: Optional seed mixed into the hash.
    :type seed: int
    :returns: 64-bit integer hash.
    :rtype: int

    Example
    -------
    >>> isinstance(_h64(('C', 4)), int)
    True
    """
    h = blake2b(digest_size=8, person=b"synrfp")
    if seed:
        h.update(seed.to_bytes(8, "little", signed=False))
    h.update(repr(obj).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def batch_h64(items: Iterable[Any], *, seed: int = 0) -> List[int]:
    """
    Hash a sequence deterministically.

    :param items: Objects to hash.
    :type items: Iterable[Any]
    :param seed: Optional integer seed.
    :type seed: int
    :returns: List of 64-bit ints.
    :rtype: List[int]
    """
    return [_h64(x, seed=seed) for x in items]


def atom_label_tuple(G: GraphData, v: NodeId, node_attrs: List[str]) -> Tuple:
    """
    Build node label tuple from selected attributes and degree.

    :param G: GraphData with node data.
    :type G: GraphData
    :param v: Node id.
    :type v: NodeId
    :param node_attrs: Attribute keys to include.
    :type node_attrs: List[str]
    :returns: Label tuple.
    :rtype: Tuple
    """
    values = [G.nodes[v].get(attr) for attr in node_attrs]
    values.append(G.degree(v))
    return tuple(values)


def bond_label_tuple(
    G: GraphData, u: NodeId, v: NodeId, edge_attrs: List[str]
) -> Tuple:
    """
    Build edge label tuple from selected attributes.

    :param G: GraphData with edge data.
    :type G: GraphData
    :param u: First node.
    :type u: NodeId
    :param v: Second node.
    :type v: NodeId
    :param edge_attrs: Edge attribute keys to include.
    :type edge_attrs: List[str]
    :returns: Label tuple.
    :rtype: Tuple
    """
    return tuple(G.edge_attr(u, v).get(attr) for attr in edge_attrs)
