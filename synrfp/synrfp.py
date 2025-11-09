# synrfp.py
# -----------------------------------------------------------------------------
# SynRFP: Mapping-free reaction fingerprints with clean separation of
#          Tokenizers (graph -> multiset of tokens)
#          Sketchers   (multiset/delta -> fixed-size sketch)
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from collections import Counter
from synrfp.graph.graph_data import GraphData
from synrfp.tokenizers.base import BaseTokenizer
from synrfp.sketchers.base import BaseSketch, WeightedSketch
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.tokenizers.nauty import NautyTokenizer
from synrfp.sketchers.parity_fold import ParityFold
from synrfp.sketchers.minhash_sketch import MinHashSketch
from synrfp.sketchers.cw_sketch import CWSketch

import numpy as _np


def build_graph_from_printout(
    nodes: Dict[int, Dict],
    edges: Dict[tuple[int, int], Dict],
) -> GraphData:
    """
    Helper to convert “printout” dicts directly into a GraphData.

    :param nodes: Mapping from node ID to attribute dict.
    :type nodes: Dict[int, Dict]
    :param edges: Mapping from (u, v) edges (with u<v) to attribute dict.
    :type edges: Dict[tuple[int, int], Dict]
    :returns: A fresh GraphData instance.
    :rtype: GraphData

    :example:
        >>> nodes = {0: {'element': 'C'}, 1: {'element': 'O'}}
        >>> edges = {(0, 1): {'order': 1.5}}
        >>> G = build_graph_from_printout(nodes, edges)
    """
    return GraphData.from_dicts(nodes, edges)


def tanimoto_bits(
    b1: Union[bytearray, List[int], _np.ndarray],
    b2: Union[bytearray, List[int], _np.ndarray],
) -> float:
    """
    Compute the Tanimoto (Jaccard) similarity between two binary‐bit sketches.

    Accepts bytearray, list[int], or numpy array of 0/1.

    :param b1: First bit array (0/1 per position).
    :type b1: bytearray or List[int] or numpy.ndarray
    :param b2: Second bit array.
    :type b2: bytearray or List[int] or numpy.ndarray
    :returns: Intersection size divided by union size, or 0.0 if union is zero.
    :rtype: float
    """
    a1 = _np.asarray(b1, dtype=int)
    a2 = _np.asarray(b2, dtype=int)
    if a1.shape != a2.shape:
        raise ValueError("bit sketches must have identical shape")
    inter = int(_np.sum((a1 & 1) & (a2 & 1)))
    union = int(_np.sum(((a1 & 1) | (a2 & 1))))
    return 0.0 if union == 0 else inter / union


def jaccard_minhash(h1: Union[list, tuple], h2: Union[list, tuple]) -> float:
    """
    Estimate Jaccard similarity from two MinHash signature arrays.

    :param h1: First MinHash hash‐value sequence.
    :type h1: list or tuple of hash ints
    :param h2: Second MinHash sequence (must be same length).
    :type h2: list or tuple of hash ints
    :returns: Fraction of positions where h1[i] == h2[i].
    :rtype: float
    """
    import numpy as _np

    a1 = _np.asarray(h1)
    a2 = _np.asarray(h2)
    if a1.shape != a2.shape:
        raise ValueError("MinHash signatures must be the same length")
    return float((a1 == a2).mean())


def sketch_to_binary(sketch: Union[bytearray, List[int], _np.ndarray]) -> List[int]:
    """
    Convert a binary sketch (e.g. a ParityFold result) into a list of 0/1 bits.

    Supports bytearray, list[int], or numpy arrays with integer/boolean dtype.

    :param sketch: The sketch object containing bits as 0/1.
    :type sketch: bytearray or List[int] or numpy.ndarray
    :returns: List of integer bits (0 or 1).
    :rtype: List[int]
    :raises TypeError: If `sketch` is not a supported type or contains non-binary values.
    """
    # numpy array path
    if isinstance(sketch, _np.ndarray):
        if sketch.ndim != 1:
            raise TypeError("numpy sketch must be a 1D array of bits")
        # cast to integers and ensure only 0/1 values
        arr = sketch.astype(int).tolist()
        if not all((x == 0 or x == 1) for x in arr):
            raise TypeError("numpy sketch contains values other than 0/1")
        return arr

    # bytearray path
    if isinstance(sketch, bytearray):
        return list(sketch)

    # list-of-int path
    if isinstance(sketch, list) and all(isinstance(x, (int, (bool,))) for x in sketch):
        if not all((int(x) == 0 or int(x) == 1) for x in sketch):
            raise TypeError("list sketch contains values other than 0/1")
        return [0 if int(x) == 0 else 1 for x in sketch]

    raise TypeError(
        "sketch_to_binary expects a bytearray, list[int], or 1D numpy array"
    )


@dataclass
class SynRFPResult:
    """
    Container for outputs of a single fingerprinting call.

    :param tokens_R: Token multiset for the reactant graph.
    :type tokens_R: Counter[int]
    :param tokens_P: Token multiset for the product graph.
    :type tokens_P: Counter[int]
    :param delta: Signed difference P-R.
    :type delta: Counter[int]
    :param support: List of token keys with nonzero delta.
    :type support: List[int]
    :param sketch: Sketch object (bytes, list, or array) from the compressor.
    :type sketch: Union[bytes, List[int], object]
    """

    tokens_R: Counter
    tokens_P: Counter
    delta: Counter
    support: List[int]
    sketch: object

    def __repr__(self) -> str:
        return (
            f"SynRFPResult("
            f"tokens_R={sum(self.tokens_R.values())} tokens, "
            f"tokens_P={sum(self.tokens_P.values())} tokens, "
            f"support={len(self.support)}, "
            f"sketch_type={type(self.sketch).__name__}"
            f")"
        )

    @staticmethod
    def describe() -> str:
        """
        Example usage:

            >>> # assume `res` is a SynRFPResult
            >>> print(res)
            SynRFPResult(tokens_R=10 tokens, tokens_P=8 tokens, support=3,
            sketch_type=bytearray)
        """
        return (
            ">>> res = SynRFP(...).fingerprint(reactant_G, product_G)\n"
            ">>> print(res)\n"
            "SynRFPResult(tokens_R=..., tokens_P=..., support=..., sketch_type=...)\n"
        )

    def to_binary(self) -> List[int]:
        """
        Return the sketch stored in this result as a plain list of 0/1 bits.

        Only works for binary sketchers (e.g. ParityFold). For non-binary
        sketchers (MinHash, CWSketch) a TypeError is raised.

        :returns: Bit-vector corresponding to the sketch.
        :rtype: List[int]
        :raises TypeError: If the underlying sketch cannot be interpreted as bits.
        """
        # If already a recognized binary container, convert it
        return sketch_to_binary(self.sketch)


class SynRFP:
    """
    Build a SynRFP fingerprint for a single‐reaction:
      - one reactant GraphData
      - one product  GraphData

    Exactly one of `sketch` or `weighted_sketch` must be provided.

    :param tokenizer: Tokenizer instance (e.g. WLTokenizer, NautyTokenizer).
    :type tokenizer: BaseTokenizer
    :param radius: Neighborhood radius for the tokenizer.
    :type radius: int
    :param sketch: Unweighted sketcher (e.g. ParityFold, MinHashSketch).
    :type sketch: Optional[BaseSketch]
    :param weighted_sketch: Weighted sketcher (e.g. CWSketch).
    :type weighted_sketch: Optional[WeightedSketch]
    :raises ValueError: If neither or both `sketch` and `weighted_sketch` are provided.
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
                "Provide exactly one of `sketch` or `weighted_sketch`,"
                + " not both or neither."
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
        Example usage:

            >>> from synrfp import build_graph_from_printout, SynRFP
            >>> from synrfp.tokenizers.wl import WLTokenizer
            >>> from synrfp.sketchers.parity_fold import ParityFold
            >>> reactant_G = build_graph_from_printout(r_nodes, r_edges)
            >>> product_G  = build_graph_from_printout(p_nodes, p_edges)
            >>> fp = SynRFP(
            ...     tokenizer=WLTokenizer(),
            ...     radius=2,
            ...     sketch=ParityFold(bits=1024, seed=1)
            ... )
            >>> res = fp.fingerprint(reactant_G, product_G)
            >>> print(res)
        """
        return (
            ">>> fp = SynRFP(tokenizer=..., radius=2, sketch=...)\n"
            ">>> res = fp.fingerprint(reactant_G, product_G)\n"
        )

    def fingerprint(
        self,
        reactant: GraphData,
        product: GraphData,
    ) -> SynRFPResult:
        """
        Compute the reaction fingerprint.

        :param reactant: GraphData for the reactant.
        :type reactant: GraphData
        :param product: GraphData for the product.
        :type product: GraphData
        :returns: Dataclass containing tokens, delta, support, and sketch.
        :rtype: SynRFPResult
        :raises TypeError: If inputs are not GraphData.
        """
        if not isinstance(reactant, GraphData) or not isinstance(product, GraphData):
            raise TypeError("reactant and product must be GraphData instances")

        # 1) Tokenize each side
        tokens_R = self.tokenizer.tokens_graph(reactant, self.radius)
        tokens_P = self.tokenizer.tokens_graph(product, self.radius)

        # 2) Signed delta: P − R
        delta: Counter = Counter(tokens_P)
        for token, count in tokens_R.items():
            delta[token] -= count
            if delta[token] == 0:
                del delta[token]

        support: List[int] = list(delta.keys())

        # 3) Build sketch or weighted sketch
        if self.sketch is not None:
            sketch_obj = self.sketch.build(support)
        else:
            pos: Dict[int, int] = {t: c for t, c in delta.items() if c > 0}
            neg: Dict[int, int] = {t: -c for t, c in delta.items() if c < 0}
            sketch_obj = self.weighted_sketch.build(pos, neg)  # type: ignore

        return SynRFPResult(
            tokens_R=tokens_R,
            tokens_P=tokens_P,
            delta=delta,
            support=support,
            sketch=sketch_obj,
        )


def synrfp(
    rsmi: str,
    *,
    tokenizer: str = "wl",
    radius: int = 2,
    sketch: str = "parity",
    bits: int = 1024,
    m: int = 256,
    seed: int = 1,
    require_pynauty: bool = False,
) -> List[int]:
    """
    Convert a reaction SMILES (RSMI) into a binary fingerprint bit-vector.

    :param rsmi: Reaction SMILES, e.g. "CCO>>C=C.O".
    :type rsmi: str
    :param tokenizer: Which tokenizer to use: "wl" or "nauty".
    :type tokenizer: str
    :param radius: Neighborhood radius for the tokenizer.
    :type radius: int
    :param sketch: Which sketcher: "parity", "minhash", or "cw" (weighted).
    :type sketch: str
    :param bits: Number of bits for parity‐fold (only if sketch="parity").
    :type bits: int
    :param m: Number of hash samples for minhash or CWSketch.
    :type m: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :param require_pynauty: If `tokenizer="nauty"`, whether to enforce `pynauty` install.
    :type require_pynauty: bool

    :returns: Fingerprint as a list of 0/1 bits.
    :rtype: List[int]

    :raises ValueError: On invalid `tokenizer` or `sketch` names.
    :raises RuntimeError: If required dependencies (e.g. `pynauty` or `datasketch`)
    are missing.
    """
    # 1) Parse tokenizer
    tok_lower = tokenizer.lower()
    if tok_lower == "wl":
        tok = WLTokenizer()
    elif tok_lower == "nauty":
        tok = NautyTokenizer(require_pynauty=require_pynauty)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer!r} (choose 'wl' or 'nauty')")

    # 2) Parse sketcher
    sketch_lower = sketch.lower()
    if sketch_lower == "parity":
        sk = ParityFold(bits=bits, seed=seed)
        weighted = False
    elif sketch_lower == "minhash":
        sk = MinHashSketch(m=m, seed=seed)
        weighted = False
    elif sketch_lower == "cw":
        sk = CWSketch(m=m, seed=seed)
        weighted = True
    else:
        raise ValueError(
            f"Unknown sketch: {sketch!r} (choose 'parity', 'minhash', or 'cw')"
        )

    # 3) Parse reaction graphs
    reactant_G, product_G = Reaction.from_rsmi(rsmi)

    # 4) Build engine and fingerprint
    if not weighted:
        engine = SynRFP(tokenizer=tok, radius=radius, sketch=sk)
    else:
        engine = SynRFP(tokenizer=tok, radius=radius, weighted_sketch=sk)  # type: ignore

    res = engine.fingerprint(reactant_G, product_G)

    # 5) Return binary bits
    return res.to_binary()
