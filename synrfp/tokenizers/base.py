# synrfp/tokenizers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence, Optional

from synrfp.graph.graph_data import GraphData


class BaseTokenizer(ABC):
    """
    Abstract base for graph tokenizers (graph → multiset of integer tokens).

    :param node_attrs: Node attribute keys to include in labels.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge attribute keys to include in labels.
    :type edge_attrs: Optional[Sequence[str]]

    Example
    -------
    >>> class Dummy(BaseTokenizer):
    ...     def tokens_graph(self, G, radius): return Counter({0: len(G.nodes)})
    ...
    """

    def __init__(
        self,
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
    ):
        self.node_attrs = (
            list(node_attrs)
            if node_attrs is not None
            else ["element", "aromatic", "charge", "hcount"]
        )
        self.edge_attrs = list(edge_attrs) if edge_attrs is not None else ["order"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(node_attrs="
            + f"{self.node_attrs}, edge_attrs={self.edge_attrs})"
        )

    @abstractmethod
    def tokens_graph(self, G: GraphData, radius: int) -> Counter:
        """
        Generate tokens for a single :class:`GraphData` instance.

        :param G: GraphData instance to tokenize.
        :type G: GraphData
        :param radius: Non-negative neighborhood radius.
        :type radius: int
        :returns: Multiset of tokens (hashed neighborhood labels).
        :rtype: Counter
        """
        if not isinstance(G, GraphData):
            raise TypeError(f"Expected GraphData, got {type(G).__name__}")
        if not isinstance(radius, int) or radius < 0:
            raise ValueError(f"Radius must be a non-negative integer, got {radius}")
        return Counter()

    def tokens_side(self, graphs: Sequence[GraphData], radius: int) -> Counter:
        """
        Generate tokens across multiple graphs (e.g., reaction sides).

        :param graphs: Sequence of GraphData objects.
        :type graphs: Sequence[GraphData]
        :param radius: Non-negative neighborhood radius.
        :type radius: int
        :returns: Combined multiset of tokens for all graphs.
        :rtype: Counter
        """
        if not isinstance(graphs, Sequence):
            raise TypeError(
                f"Expected sequence of GraphData, got {type(graphs).__name__}"
            )
        if not isinstance(radius, int) or radius < 0:
            raise ValueError(f"Radius must be a non-negative integer, got {radius}")
        out = Counter()
        for g in graphs:
            if not isinstance(g, GraphData):
                raise TypeError(
                    f"Expected GraphData in sequence, got {type(g).__name__}"
                )
            out.update(self.tokens_graph(g, radius))
        return out

    @staticmethod
    def describe() -> str:
        """
        Return a generic usage example for tokenizers.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "tokenizer = WLTokenizer(node_attrs=['element'], edge_attrs=['order'])\n"
            "C = tokenizer.tokens_graph(graph, radius=2)\n"
        )
