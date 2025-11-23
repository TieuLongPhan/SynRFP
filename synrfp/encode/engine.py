# synrfp/encode/engine.py
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.base import BaseTokenizer
from synrfp.sketchers.base import BaseSketch, WeightedSketch
from .result import SynRFPResult


class SynRFP:
    """
    Build a SynRFP fingerprint for a single reaction:
      - one reactant :class:`Molecule`
      - one product  :class:`Molecule`

    Exactly one of ``sketch`` or ``weighted_sketch`` must be provided.

    :param tokenizer: Tokenizer instance (e.g. :class:`WLTokenizer`,
                      :class:`NautyTokenizer`, :class:`MorganTokenizer`,
                      :class:`PathTokenizer`).
    :type tokenizer: BaseTokenizer
    :param radius: Neighborhood radius for the tokenizer.
    :type radius: int
    :param sketch: Unweighted sketcher (e.g. :class:`ParityFold`,
                   :class:`MinHashSketch`).
    :type sketch: BaseSketch | None
    :param weighted_sketch: Weighted sketcher (e.g. :class:`CWSketch`,
                            :class:`SRPSketch`).
    :type weighted_sketch: WeightedSketch | None
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        radius: int = 2,
        sketch: Optional[BaseSketch] = None,
        weighted_sketch: Optional[WeightedSketch] = None,
    ):
        # exactly one of sketch / weighted_sketch
        if (sketch is None) == (weighted_sketch is None):
            raise ValueError(
                "Provide exactly one of `sketch` or `weighted_sketch`, "
                "not both or neither."
            )

        self.tokenizer = tokenizer
        if not isinstance(radius, int) or radius < 0:
            raise ValueError(f"radius must be a non-negative int, got {radius!r}")
        self.radius = radius

        self.sketch = sketch
        self.weighted_sketch = weighted_sketch

    def __repr__(self) -> str:
        skl = (
            type(self.sketch).__name__
            if self.sketch is not None
            else type(self.weighted_sketch).__name__
        )
        return (
            f"SynRFP(tokenizer={type(self.tokenizer).__name__}, "
            f"radius={self.radius}, sketcher={skl})"
        )

    @staticmethod
    def describe() -> str:
        """
        Example usage::

            >>> fp = SynRFP(tokenizer=WLTokenizer(), radius=2, sketch=ParityFold(1024))
            >>> res = fp.fingerprint(reactant_G, product_G)

        :returns: Example usage string.
        :rtype: str
        """
        return (
            ">>> fp = SynRFP(tokenizer=..., radius=2, sketch=...)\n"
            ">>> res = fp.fingerprint(reactant_G, product_G)\n"
        )

    # ------------------------ small helpers ------------------------ #
    def _validate_inputs(self, reactant: Molecule, product: Molecule) -> None:
        if not isinstance(reactant, Molecule) or not isinstance(product, Molecule):
            raise TypeError("reactant and product must be Molecule instances")

    def _maybe_override_tokenizer_attrs(
        self,
        node_attrs: Optional[Sequence[str]],
        edge_attrs: Optional[Sequence[str]],
    ) -> None:
        if node_attrs is not None:
            self.tokenizer.node_attrs = tuple(node_attrs)
        if edge_attrs is not None:
            self.tokenizer.edge_attrs = tuple(edge_attrs)

    def _compute_delta_union(
        self, tokens_R: Counter, tokens_P: Counter, mode: str
    ) -> Tuple[Counter, List[int]]:
        """
        Compute either signed delta (P - R) or union (R + P).

        Returns (delta_counter, support_list) where delta_counter semantics:
          - mode == 'delta': signed counts (may be negative)
          - mode == 'union': non-negative counts (R+P)
        """
        if mode == "delta":
            delta = Counter(tokens_P)
            # Subtract counts from R; remove zero entries
            for t, c in tokens_R.items():
                delta[t] -= c
                if delta[t] == 0:
                    del delta[t]
            support = list(delta.keys())
            return delta, support

        elif mode == "union":
            union_counts = Counter(tokens_R) + Counter(tokens_P)
            support = list(union_counts.keys())
            return union_counts, support

        else:
            raise ValueError("mode must be one of {'delta', 'union'}")

    def _encode_unweighted_support(self, delta: Counter, mode: str) -> List[int]:
        """
        For unweighted sketchers, produce the integer sequence to pass to
        sketch.build(). When mode == 'delta' encode sign into token id using:
            enc_t = 2 * int(t) + sign_bit
        where sign_bit = 0 for created (delta>0) and 1 for consumed (delta<0).

        When mode == 'union' we simply return the list of token ids.
        """
        if mode == "union":
            return list(delta.keys())

        # mode == "delta"
        encoded: List[int] = []
        for t, c in delta.items():
            if c == 0:
                continue
            sign_bit = 0 if c > 0 else 1
            enc_t = 2 * int(t) + sign_bit
            encoded.append(enc_t)
        return encoded

    def _build_weighted_maps(
        self, delta: Counter, mode: str
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create pos_map, neg_map expected by WeightedSketch.build(pos, neg).
        For mode 'delta': pos_map contains positive counts, neg_map contains magnitudes
        of negative counts. For mode 'union': pos_map contains union counts
        and neg_map is empty.
        """
        if mode == "union":
            return dict(delta), {}

        pos_map = {t: c for t, c in delta.items() if c > 0}
        neg_map = {t: -c for t, c in delta.items() if c < 0}
        return pos_map, neg_map

    # ------------------------ main method ------------------------- #
    def fingerprint(
        self,
        reactant: Molecule,
        product: Molecule,
        *,
        mode: str = "delta",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
    ) -> SynRFPResult:
        """
        Compute the reaction fingerprint for a pair of molecules.

        :param reactant: Reactant molecular graph.
        :type reactant: Molecule
        :param product: Product molecular graph.
        :type product: Molecule
        :param mode: Token combination mode:
                     - ``'delta'``: signed difference P−R (default)
                     - ``'union'``: union counts R+P
        :type mode: str
        :param node_attrs: Optional node attribute names for tokenizer.
        :type node_attrs: Sequence[str] | None
        :param edge_attrs: Optional edge attribute names for tokenizer.
        :type edge_attrs: Sequence[str] | None
        :returns: A :class:`SynRFPResult` with tokens, support and sketch.
        :rtype: SynRFPResult
        :raises TypeError: If inputs are not :class:`Molecule` instances.
        :raises ValueError: If ``mode`` is invalid.
        """
        # Lightweight validation / config override
        self._validate_inputs(reactant, product)
        self._maybe_override_tokenizer_attrs(node_attrs, edge_attrs)

        # 1) Tokenize both sides
        tokens_R: Counter = self.tokenizer.tokens_graph(reactant, self.radius)
        tokens_P: Counter = self.tokenizer.tokens_graph(product, self.radius)

        # 2) compute delta / union
        delta, support = self._compute_delta_union(tokens_R, tokens_P, mode)

        # 3) Build sketch
        if self.sketch is not None:
            # Unweighted sketchers expect an integer sequence
            encoded_support = self._encode_unweighted_support(delta, mode)
            sketch_obj = self.sketch.build(encoded_support)
            support_out = encoded_support
        else:
            # Weighted sketchers receive pos/neg maps
            pos_map, neg_map = self._build_weighted_maps(delta, mode)
            sketch_obj = self.weighted_sketch.build(pos_map, neg_map)  # type: ignore[arg-type]
            support_out = (
                list(pos_map.keys()) if mode == "union" else list(delta.keys())
            )

        return SynRFPResult(
            tokens_R=tokens_R,
            tokens_P=tokens_P,
            delta=delta,
            support=support_out,
            sketch=sketch_obj,
            mode=mode,
        )
