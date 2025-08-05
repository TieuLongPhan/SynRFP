# synrfp/tokenizers/base.py
from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence

from synrfp.graph.graph_data import GraphData


class BaseTokenizer(ABC):
    """
    Abstract base class for graph tokenizers: graph -> multiset of integer tokens.

    :param node_attrs: Optional list of node attribute keys to include in labels.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Optional list of edge attribute keys to include in labels.
    :type edge_attrs: Optional[Sequence[str]]
    """

    def __init__(self, node_attrs=None, edge_attrs=None):
        """
        Initialize tokenizer with optional attribute filters.

        :param node_attrs: List of node attributes to use (default molecular keys).
        :type node_attrs: Optional[Sequence[str]]
        :param edge_attrs: List of edge attributes to use (default molecular keys).
        :type edge_attrs: Optional[Sequence[str]]
        """
        # default attributes
        self.node_attrs = (
            list(node_attrs)
            if node_attrs is not None
            else ["element", "aromatic", "charge", "hcount"]
        )
        self.edge_attrs = list(edge_attrs) if edge_attrs is not None else ["order"]

    """
    Abstract base class for graph tokenizers: graph -> multiset of integer tokens.
    """

    @abstractmethod
    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Generate tokens for a single GraphData instance after validating inputs.

        :param G: GraphData instance to tokenize
        :type G: GraphData
        :param radius: non-negative neighborhood radius
        :type radius: int
        :returns: multiset of tokens (hashed neighborhood labels)
        :rtype: Counter
        :raises TypeError: if G is not a GraphData
        :raises ValueError: if radius is not a non-negative integer
        """
        # validate graph type
        if not isinstance(G, GraphData):
            raise TypeError(f"Expected GraphData, got {type(G).__name__}")
        # validate radius
        if not isinstance(radius, int) or radius < 0:
            raise ValueError(f"Radius must be a non-negative integer, got {radius}")
        # subclass must implement actual tokenization logic
        return Counter()

    def tokens_side(self, graphs: Sequence[GraphData], radius: int) -> Counter:
        """
        Generate tokens across multiple GraphData instances (e.g., reaction sides).

        :param graphs: sequence of GraphData instances
        :type graphs: Sequence[GraphData]
        :param radius: non-negative neighborhood radius
        :type radius: int
        :returns: combined multiset of tokens for all graphs
        :rtype: Counter
        :raises TypeError: if graphs is not a sequence of GraphData
        :raises ValueError: if radius is not a non-negative integer
        """
        # validate sequence
        if not isinstance(graphs, Sequence):
            raise TypeError(
                f"Expected sequence of GraphData, got {type(graphs).__name__}"
            )
        # validate radius
        if not isinstance(radius, int) or radius < 0:
            raise ValueError(f"Radius must be a non-negative integer, got {radius}")
        # aggregate tokens
        C = Counter()
        for g in graphs:
            if not isinstance(g, GraphData):
                raise TypeError(
                    f"Expected GraphData in sequence, got {type(g).__name__}"
                )
            C.update(self.tokens_graph(g, radius))
        return C
