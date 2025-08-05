from __future__ import annotations
from typing import Tuple, Iterator, Union
from dataclasses import dataclass
import networkx as nx
import pandas as pd

from synrfp.graph.graph_data import GraphData

NodeId = int
Edge = Tuple[NodeId, NodeId]


@dataclass
class Reaction:
    """
    Represents a chemical reaction with a single reactant and a single product graph.

    :param reactant: GraphData for the reactant molecule.
    :type reactant: GraphData
    :param product: GraphData for the product molecule.
    :type product: GraphData
    """

    reactant: GraphData
    product: GraphData

    @staticmethod
    def from_rsmi(rsmi: str) -> Reaction:
        """
        Create a Reaction from an RSMI string using synkit IO.

        :param rsmi: Reaction SMILES string.
        :type rsmi: str
        :returns: Reaction with reactant and product GraphData.
        :rtype: Reaction
        :raises ValueError: If parsing fails.
        """
        try:
            from synkit.IO import rsmi_to_graph
        except ImportError as e:
            raise RuntimeError("synkit.IO.rsmi_to_graph not available") from e
        r_graph, p_graph = rsmi_to_graph(
            rsmi, drop_non_aam=False, use_index_as_atom_map=False
        )
        return Reaction(
            reactant=GraphData.from_nx_graph(r_graph),
            product=GraphData.from_nx_graph(p_graph),
        )

    @staticmethod
    def from_graph(reactant_graph: nx.Graph, product_graph: nx.Graph) -> Reaction:
        """
        Create a Reaction from two NetworkX graphs.

        :param reactant_graph: NetworkX Graph for reactant.
        :type reactant_graph: nx.Graph
        :param product_graph: NetworkX Graph for product.
        :type product_graph: nx.Graph
        :returns: Reaction instance.
        :rtype: Reaction
        """
        return Reaction(
            reactant=GraphData.from_nx_graph(reactant_graph),
            product=GraphData.from_nx_graph(product_graph),
        )

    def __repr__(self) -> str:
        """
        Representation showing reactant and product sizes.

        :returns: String repr.
        :rtype: str
        """
        return (
            f"Reaction(reactant_nodes={len(self.reactant.nodes)}, "
            f"reactant_edges={len(self.reactant.edges)}, "
            f"product_nodes={len(self.product.nodes)}, "
            f"product_edges={len(self.product.edges)})"
        )

    def help(self) -> str:
        """
        Show usage examples for Reaction.

        :returns: Usage guide.
        :rtype: str
        """
        return (
            "# Examples:\n"
            "# From reaction SMILES string:\n"
            "#   rxn = Reaction.from_rsmi('CCO>>CC=O')\n"
            "# From networkx Graphs:\n"
            "#   rxn = Reaction.from_graph(G_react, G_prod)\n"
        )

    def __len__(self) -> int:
        """
        Number of sides (reactant + product).

        :returns: Always 2.
        :rtype: int
        """
        return 2

    def __iter__(self) -> Iterator[GraphData]:
        """
        Iterate over reactant and product GraphData.

        :returns: Iterator yielding reactant then product.
        :rtype: Iterator[GraphData]
        """
        yield self.reactant
        yield self.product

    def __getitem__(self, key: Union[int, str]) -> GraphData:
        """
        Index reaction sides by 0/reactant or 1/product or keys.

        :param key: 0, 1, 'reactant', or 'product'.
        :type key: int or str
        :returns: Corresponding GraphData.
        :rtype: GraphData
        :raises KeyError: If key invalid.
        """
        if key in (0, "reactant"):
            return self.reactant
        if key in (1, "product"):
            return self.product
        raise KeyError(f"Invalid index {key}, use 0/reactant or 1/product.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Summarize reaction graphs as a pandas DataFrame.

        :returns: DataFrame with columns ['side','n_nodes','n_edges'].
        :rtype: pd.DataFrame
        """
        data = [
            {
                "side": "reactant",
                "n_nodes": len(self.reactant.nodes),
                "n_edges": len(self.reactant.edges),
            },
            {
                "side": "product",
                "n_nodes": len(self.product.nodes),
                "n_edges": len(self.product.edges),
            },
        ]
        return pd.DataFrame(data)
