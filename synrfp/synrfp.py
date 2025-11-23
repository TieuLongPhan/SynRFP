# # synrfp/synrfp.py
# # -----------------------------------------------------------------------------
# # SynRFP: Mapping-free reaction fingerprints with clean separation of
# #          Tokenizers (graph -> multiset of tokens)
# #          Sketchers   (multiset/delta -> fixed-size sketch)
# # -----------------------------------------------------------------------------

# from dataclasses import dataclass
# from typing import Dict, List, Optional, Sequence

# from collections import Counter
# import numpy as np

# from synrfp.graph.molecule import Molecule
# from synrfp.tokenizers.base import BaseTokenizer
# from synrfp.sketchers.base import BaseSketch, WeightedSketch
# from synrfp.graph.reaction import Reaction
# from synrfp.tokenizers.wl import WLTokenizer
# from synrfp.tokenizers.nauty import NautyTokenizer
# from synrfp.sketchers.parity_fold import ParityFold
# from synrfp.sketchers.minhash_sketch import MinHashSketch
# from synrfp.sketchers.cw_sketch import CWSketch
# from .utils import sketch_to_array, sketch_to_binary, signature_to_bits

# # ---------------------------------------------------------------------------
# # Result container
# # ---------------------------------------------------------------------------


# @dataclass
# class SynRFPResult:
#     """
#     Container for outputs of a single fingerprinting call.

#     :param tokens_R: Token multiset for the reactant graph.
#     :type tokens_R: Counter
#     :param tokens_P: Token multiset for the product graph.
#     :type tokens_P: Counter
#     :param delta: Token counts summarising the transformation, depending on
#                   ``mode``:
#                   - if ``mode='delta'``: signed difference P−R
#                   - if ``mode='union'``: union counts (R+P)
#     :type delta: Counter
#     :param support: List of token keys with nonzero contribution (delta or union).
#     :type support: List[int]
#     :param sketch: Sketch object (bytes, list, or array) from the compressor.
#     :type sketch: object
#     :param mode: Fingerprint mode, either ``'delta'`` or ``'union'``.
#     :type mode: str
#     """

#     tokens_R: Counter
#     tokens_P: Counter
#     delta: Counter
#     support: List[int]
#     sketch: object
#     mode: str = "delta"

#     def __repr__(self) -> str:
#         return (
#             f"SynRFPResult("
#             f"tokens_R={sum(self.tokens_R.values())} tokens, "
#             f"tokens_P={sum(self.tokens_P.values())} tokens, "
#             f"support={len(self.support)}, "
#             f"mode={self.mode!r}, "
#             f"sketch_type={type(self.sketch).__name__}"
#             f")"
#         )

#     @staticmethod
#     def describe() -> str:
#         """
#         Example usage::

#             >>> # assume `res` is a SynRFPResult
#             >>> print(res)
#             SynRFPResult(tokens_R=10 tokens, tokens_P=8 tokens,
#             support=3, mode='delta', sketch_type=bytearray)

#         :returns: Example usage string.
#         :rtype: str
#         """
#         return (
#             ">>> res = SynRFP(...).fingerprint(reactant_G, product_G)\n"
#             ">>> print(res)\n"
#             "SynRFPResult(tokens_R=..., tokens_P=..., support=..., "
#             "mode='delta'|'union', sketch_type=...)\n"
#         )

#     def to_binary(self) -> List[int]:
#         """
#         Return the sketch stored in this result as a plain list of 0/1 bits.

#         Only works for binary sketchers (e.g. ParityFold). For non-binary
#         sketchers (MinHash, CWSketch) a :class:`TypeError` is raised.

#         :returns: Binary fingerprint as list of 0/1 bits.
#         :rtype: List[int]
#         :raises TypeError: If the underlying sketch cannot be interpreted as bits.
#         """
#         return sketch_to_binary(self.sketch)

#     def as_array(self) -> np.ndarray:
#         """
#         Return the underlying sketch as a 1D numpy integer array.

#         This works for all sketcher types:
#           - ParityFold: 0/1 array
#           - MinHashSketch: hash values
#           - CWSketch: sample indices

#         :returns: 1D numpy array representation of the sketch.
#         :rtype: numpy.ndarray
#         """
#         return sketch_to_array(self.sketch)


# # ---------------------------------------------------------------------------
# # Core engine
# # ---------------------------------------------------------------------------


# class SynRFP:
#     """
#     Build a SynRFP fingerprint for a single reaction:
#       - one reactant :class:`Molecule`
#       - one product  :class:`Molecule`

#     Exactly one of ``sketch`` or ``weighted_sketch`` must be provided.

#     :param tokenizer: Tokenizer instance (e.g. :class:`WLTokenizer`,
#                       :class:`NautyTokenizer`).
#     :type tokenizer: BaseTokenizer
#     :param radius: Neighborhood radius for the tokenizer.
#     :type radius: int
#     :param sketch: Unweighted sketcher (e.g. :class:`ParityFold`,
#                    :class:`MinHashSketch`).
#     :type sketch: Optional[BaseSketch]
#     :param weighted_sketch: Weighted sketcher (e.g. :class:`CWSketch`).
#     :type weighted_sketch: Optional[WeightedSketch]
#     """

#     def __init__(
#         self,
#         tokenizer: BaseTokenizer,
#         radius: int = 2,
#         sketch: Optional[BaseSketch] = None,
#         weighted_sketch: Optional[WeightedSketch] = None,
#     ):
#         # exactly one of sketch / weighted_sketch
#         if (sketch is None) == (weighted_sketch is None):
#             raise ValueError(
#                 "Provide exactly one of `sketch` or `weighted_sketch`, "
#                 "not both or neither."
#             )

#         self.tokenizer = tokenizer
#         if not isinstance(radius, int) or radius < 0:
#             raise ValueError(f"radius must be a non-negative int, got {radius!r}")
#         self.radius = radius

#         self.sketch = sketch
#         self.weighted_sketch = weighted_sketch

#     def __repr__(self) -> str:
#         skl = (
#             type(self.sketch).__name__
#             if self.sketch is not None
#             else type(self.weighted_sketch).__name__
#         )
#         return (
#             f"SynRFP(tokenizer={type(self.tokenizer).__name__}, "
#             f"radius={self.radius}, sketcher={skl})"
#         )

#     @staticmethod
#     def describe() -> str:
#         """
#         Example usage::

#             >>> fp = SynRFP(tokenizer=WLTokenizer(), radius=2, sketch=ParityFold(1024))
#             >>> res = fp.fingerprint(reactant_G, product_G)

#         :returns: Example usage string.
#         :rtype: str
#         """
#         return (
#             ">>> fp = SynRFP(tokenizer=..., radius=2, sketch=...)\n"
#             ">>> res = fp.fingerprint(reactant_G, product_G)\n"
#         )

#     def fingerprint(
#         self,
#         reactant: Molecule,
#         product: Molecule,
#         *,
#         mode: str = "delta",
#         node_attrs: Optional[Sequence[str]] = None,
#         edge_attrs: Optional[Sequence[str]] = None,
#     ) -> SynRFPResult:
#         """
#         Compute the reaction fingerprint for a pair of molecules.

#         :param reactant: Reactant molecular graph.
#         :type reactant: Molecule
#         :param product: Product molecular graph.
#         :type product: Molecule
#         :param mode: Token combination mode:

#                      - ``"delta"`` (default): signed difference P−R. Only tokens
#                        whose net count changes are kept in ``support``.
#                      - ``"union"``: union of tokens present in either side.
#                        ``delta`` then stores union counts (R+P).

#         :type mode: str
#         :param node_attrs: Optional sequence of node attribute names that the
#                            tokenizer should use when canonicalising/enumerating
#                            subgraphs (e.g. ``["element", "formal_charge"]``).
#                            If ``None``, tokenizer defaults are used.
#         :type node_attrs: Optional[Sequence[str]]
#         :param edge_attrs: Optional sequence of edge attribute names that the
#                            tokenizer should use (e.g. ``["order", "stereo"]``).
#                            If ``None``, tokenizer defaults are used.
#         :type edge_attrs: Optional[Sequence[str]]
#         :returns: A :class:`SynRFPResult` with tokens, support and sketch.
#         :rtype: SynRFPResult
#         :raises TypeError: If inputs are not :class:`Molecule` instances.
#         :raises ValueError: If ``mode`` is not one of ``{"delta", "union"}``.
#         """
#         if not isinstance(reactant, Molecule) or not isinstance(product, Molecule):
#             raise TypeError("reactant and product must be Molecule instances")

#         # Optionally override tokenizer attribute preferences
#         if node_attrs is not None:
#             self.tokenizer.node_attrs = tuple(node_attrs)
#         if edge_attrs is not None:
#             self.tokenizer.edge_attrs = tuple(edge_attrs)

#         # 1) Tokenize each side
#         tokens_R: Counter = self.tokenizer.tokens_graph(reactant, self.radius)
#         tokens_P: Counter = self.tokenizer.tokens_graph(product, self.radius)

#         # 2) Build token maps depending on mode
#         if mode == "delta":
#             # signed difference: P - R
#             delta: Counter = Counter(tokens_P)
#             for t, c in tokens_R.items():
#                 delta[t] -= c
#                 if delta[t] == 0:
#                     del delta[t]
#             # support initially in "raw" token-id space
#             support: List[int] = list(delta.keys())

#         elif mode == "union":
#             # union presence: set of tokens appearing on either side
#             union_counts: Counter = Counter(tokens_R) + Counter(tokens_P)
#             delta = union_counts  # reuse field; semantics documented in SynRFPResult
#             support = list(union_counts.keys())

#         else:
#             raise ValueError("mode must be one of {'delta', 'union'}")

#         # 3) Build sketch depending on sketcher type
#         if self.sketch is not None:
#             # Unweighted sketchers (ParityFold, MinHash) expect a sequence
#             # of token IDs. For mode="delta", we make this sign-aware by
#             # encoding direction directly into the token ID:
#             #
#             #   delta[t] > 0  (created)  ->  feature 2*t
#             #   delta[t] < 0  (consumed) ->  feature 2*t + 1
#             #
#             # This keeps the interface "set of ints" but avoids collisions
#             # between reactions that only differ by direction of change.
#             if mode == "delta":
#                 encoded_support: List[int] = []
#                 for t, c in delta.items():
#                     if c == 0:
#                         continue
#                     sign_bit = 0 if c > 0 else 1  # 0: created, 1: consumed
#                     enc_t = 2 * int(t) + sign_bit
#                     encoded_support.append(enc_t)
#                 support = encoded_support

#             sketch_obj = self.sketch.build(support)
#         else:
#             # Weighted sketchers (e.g. CWSketch) expect positive/negative maps
#             if mode == "delta":
#                 pos_map: Dict[int, int] = {t: c for t, c in delta.items() if c > 0}
#                 neg_map: Dict[int, int] = {t: -c for t, c in delta.items() if c < 0}
#             else:  # mode == "union": treat union counts as positive weights
#                 pos_map = dict(delta)
#                 neg_map = {}
#             sketch_obj = self.weighted_sketch.build(pos_map, neg_map)  # type: ignore

#         return SynRFPResult(
#             tokens_R=tokens_R,
#             tokens_P=tokens_P,
#             delta=delta,
#             support=support,
#             sketch=sketch_obj,
#             mode=mode,
#         )


# # ---------------------------------------------------------------------------
# # Convenience top-level API
# # ---------------------------------------------------------------------------


# def synrfp(
#     rsmi: str,
#     *,
#     tokenizer: str = "wl",
#     radius: int = 2,
#     sketch: str = "parity",
#     bits: int = 1024,
#     m: int = 256,
#     seed: int = 1,
#     mode: str = "delta",
#     node_attrs: Optional[Sequence[str]] = None,
#     edge_attrs: Optional[Sequence[str]] = None,
#     require_pynauty: bool = False,
# ) -> List[int]:
#     """
#     Convert a reaction SMILES (RSMI) into a binary fingerprint bit-vector.

#     Internally:

#     - A tokenizer (WL or Nauty) converts each side into multiset tokens.
#     - Depending on ``mode``, either token *delta* (P−R) or *union* (R+P)
#       is computed.
#     - A sketcher (parity, minhash, cw) converts the token set/weights into
#       a fixed-size sketch.
#     - The sketch is finally mapped into a binary vector of length ``bits``.

#     Parameters
#     ----------
#     :param rsmi: Reaction SMILES, e.g. ``"CCO>>C=C.O"``.
#     :type rsmi: str
#     :param tokenizer: Which tokenizer to use:

#                       - ``"wl"``: Weisfeiler–Lehman style tokenizer.
#                       - ``"nauty"``: Nauty-based canonical labeling tokenizer.

#     :type tokenizer: str
#     :param radius: Neighborhood radius for the tokenizer.
#     :type radius: int
#     :param sketch: Which sketcher to use:

#                    - ``"parity"`` (default): parity-folding into a binary vector.
#                    - ``"minhash"``: MinHash signature, then mapped to bits.
#                    - ``"cw"``: count-weighted sketch, then mapped to bits.

#     :type sketch: str
#     :param bits: Length of the final binary fingerprint.
#                  For ``sketch="parity"``, this is the internal bit-length of
#                  :class:`ParityFold`. For ``"minhash"``/``"cw"``, it controls
#                  the final bin count used by :func:`signature_to_bits`.
#     :type bits: int
#     :param m: Number of hash samples for MinHash or CWSketch.
#     :type m: int
#     :param seed: Random seed for reproducibility.
#     :type seed: int
#     :param mode: Token combination mode:

#                  - ``"delta"`` (default): signed difference P−R.
#                  - ``"union"``: union of tokens appearing on either side.

#     :type mode: str
#     :param node_attrs: Optional sequence of node attribute names passed to
#                        the tokenizer (e.g. ``["element", "formal_charge"]``).
#                        If ``None``, tokenizer defaults are used.
#     :type node_attrs: Optional[Sequence[str]]
#     :param edge_attrs: Optional sequence of edge attribute names passed to
#                        the tokenizer (e.g. ``["order", "stereo"]``).
#                        If ``None``, tokenizer defaults are used.
#     :type edge_attrs: Optional[Sequence[str]]
#     :param require_pynauty: If ``tokenizer="nauty"``, whether to enforce
#                             ``pynauty`` installation (raise if missing).
#     :type require_pynauty: bool

#     :returns: Fingerprint as a list of 0/1 bits of length ``bits``.
#     :rtype: List[int]

#     :raises ValueError: On invalid ``tokenizer``, ``sketch``, or ``mode`` names.
#     :raises RuntimeError: If required dependencies (e.g. ``pynauty`` or
#                           ``datasketch``) are missing.
#     """
#     # 1) Parse tokenizer
#     tok_lower = tokenizer.lower()
#     if tok_lower == "wl":
#         tok = WLTokenizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
#     elif tok_lower == "nauty":
#         tok = NautyTokenizer(
#             node_attrs=node_attrs,
#             edge_attrs=edge_attrs,
#             require_pynauty=require_pynauty,
#         )
#     else:
#         raise ValueError(f"Unknown tokenizer: {tokenizer!r} (choose 'wl' or 'nauty')")

#     # 2) Parse sketcher
#     sketch_lower = sketch.lower()
#     if sketch_lower == "parity":
#         sk = ParityFold(bits=bits, seed=seed)
#         weighted = False
#     elif sketch_lower == "minhash":
#         sk = MinHashSketch(m=m, seed=seed)
#         weighted = False
#     elif sketch_lower == "cw":
#         sk = CWSketch(m=m, seed=seed)
#         weighted = True
#     else:
#         raise ValueError(
#             f"Unknown sketch: {sketch!r} (choose 'parity', 'minhash', or 'cw')"
#         )

#     # 3) Parse reaction graphs
#     parsed = Reaction.from_rsmi(rsmi)
#     # Backwards compatibility: Reaction.from_rsmi may return either a Reaction
#     # instance or a (reactant, product) tuple.
#     if isinstance(parsed, Reaction):
#         reactant_G, product_G = parsed.reactant, parsed.product
#     else:
#         reactant_G, product_G = parsed  # type: ignore[assignment]

#     # 4) Build engine and fingerprint
#     if not weighted:
#         engine = SynRFP(tokenizer=tok, radius=radius, sketch=sk)
#     else:
#         engine = SynRFP(tokenizer=tok, radius=radius, weighted_sketch=sk)

#     res = engine.fingerprint(
#         reactant_G,
#         product_G,
#         mode=mode,
#         node_attrs=node_attrs,
#         edge_attrs=edge_attrs,
#     )

#     # 5) Return binary bits
#     if sketch_lower == "parity":
#         # parity-fold already produces binary bits internally
#         return res.to_binary()
#     else:
#         # MinHash / CWSketch: convert numeric signature -> bit-vector of length `bits`
#         sig = res.as_array()
#         return signature_to_bits(sig, bits)

# synrfp/synrfp.py  (shim / backwards-compat)
from synrfp.encode.core import synrfp, SynRFP
from synrfp.encode.result import SynRFPResult

__all__ = ["synrfp", "SynRFP", "SynRFPResult"]
