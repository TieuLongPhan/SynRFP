# synrfp/encode/core.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.tokenizers.nauty import NautyTokenizer
from synrfp.tokenizers.morgan import MorganTokenizer
from synrfp.tokenizers.path import PathTokenizer
from synrfp.tokenizers.map import MAPTokenizer
from synrfp.sketchers.parity_fold import ParityFold
from synrfp.sketchers.minhash_sketch import MinHashSketch
from synrfp.sketchers.cw_sketch import CWSketch
from synrfp.sketchers.srp_sketch import SRPSketch
from synrfp.utils import signature_to_bits
from .engine import SynRFP
from .result import SynRFPResult


def synrfp(
    rsmi: str,
    *,
    tokenizer: str = "wl",
    radius: int = 2,
    sketch: str = "parity",
    bits: int = 1024,
    m: Optional[int] = None,
    seed: int = 1,
    mode: str = "delta",
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
    bbit: int = 2,
    fold: str = "bbit",
    agents: str = "both",
) -> List[int]:
    """
    Convert a reaction SMILES (RSMI) into a binary fingerprint bit-vector.

    Internally:

    - A tokenizer (WL, Nauty, Morgan, Path, MAP) converts each side into
      multiset tokens.
    - Depending on ``mode``, either token *delta* (P−R) or *union* (R+P)
      is computed.
    - A sketcher (parity, minhash, cw, srp) converts the token set/weights into
      a fixed-size sketch.
    - The sketch is finally mapped into a binary vector of length ``bits``.

    :param rsmi: Reaction SMILES, e.g. ``"CCO>>C=C.O"``.
    :type rsmi: str
    :param tokenizer: ``"wl"`` (Weisfeiler–Lehman), ``"nauty"`` (canonical
        ego-subgraph labels), ``"morgan"`` (ECFP-style neighbourhoods),
        ``"path"`` (bond paths) or ``"map"``.
    :type tokenizer: str
    :param radius: Neighbourhood radius for the tokenizer (ignored where not
        applicable).
    :type radius: int
    :param sketch: ``"parity"`` (parity folding), ``"minhash"`` (MinHash),
        ``"cw"`` (consistent weighted sampling) or ``"srp"`` (signed random
        projection, cosine-oriented).
    :type sketch: str
    :param bits: Length of the final binary fingerprint. For ``"parity"`` it
        is the internal length of :class:`ParityFold`; for the other sketches it
        is the bin count used by :func:`signature_to_bits`.
    :type bits: int
    :param m: Number of MinHash/CWS samples. ``None`` sets
        ``m = bits // 2**bbit`` so that the b-bit encoding fills exactly
        ``bits`` positions. SRP always uses ``m = bits`` (one sign per bit).
    :type m: Optional[int]
    :param seed: Hash seed of the sketch.
    :type seed: int
    :param mode: ``"delta"`` (signed difference P−R) or ``"union"`` (tokens
        of either side).
    :type mode: str
    :param node_attrs: Node attribute names passed to the tokenizer
        (e.g. ``["element"]``).
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge attribute names passed to the tokenizer
        (e.g. ``["order"]``).
    :type edge_attrs: Optional[Sequence[str]]
    :param bbit: Bits per signature sample for the b-bit encoding of
        MinHash/CWS.
    :type bbit: int
    :param fold: Signature-to-bits mapping, ``"bbit"`` or ``"mod"`` (legacy).
    :type fold: str
    :param agents: Placement of agents in ``reactants>agents>products``:
        ``"both"``, ``"reactants"`` or ``"drop"``; see
        :meth:`synrfp.graph.reaction.Reaction.from_rsmi`.
    :type agents: str
    :return: Fingerprint as a list of 0/1 bits of length ``bits``.
    :rtype: List[int]
    :raises ValueError: On invalid ``tokenizer``, ``sketch`` or ``mode`` names.
    :raises RuntimeError: If an optional dependency (e.g. ``datasketch``) is
        missing.
    """
    # 1) Parse tokenizer
    tok_lower = tokenizer.lower()
    if tok_lower == "wl":
        tok = WLTokenizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
    elif tok_lower == "nauty":
        tok = NautyTokenizer(
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )
    elif tok_lower == "morgan":
        tok = MorganTokenizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
    elif tok_lower == "path":
        tok = PathTokenizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
    elif tok_lower == "map":
        tok = MAPTokenizer(node_attrs=node_attrs, edge_attrs=edge_attrs)
    else:
        raise ValueError(
            f"Unknown tokenizer: {tokenizer!r} "
            "(choose 'wl', 'nauty', 'morgan', 'path' or 'map')"
        )

    # 2) Parse sketcher
    sketch_lower = sketch.lower()
    sk, weighted, m = _make_sketcher(sketch, bits, m, seed, bbit)

    # 3) Parse reaction graphs
    parsed = Reaction.from_rsmi(rsmi, agents=agents)
    if isinstance(parsed, Reaction):
        reactant_G, product_G = parsed.reactant, parsed.product
    else:
        reactant_G, product_G = parsed  # type: ignore[assignment]

    # 4) Build engine and fingerprint
    if not weighted:
        engine = SynRFP(tokenizer=tok, radius=radius, sketch=sk)
    else:
        engine = SynRFP(tokenizer=tok, radius=radius, weighted_sketch=sk)

    res: SynRFPResult = engine.fingerprint(
        reactant_G,
        product_G,
        mode=mode,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
    )

    # 5) Return binary bits
    return _to_bits(res, sketch_lower, bits, fold)


def fingerprint_from_tokens(
    tokens_R,
    tokens_P,
    *,
    sketch: str = "parity",
    bits: int = 1024,
    m: Optional[int] = None,
    seed: int = 1,
    mode: str = "delta",
    bbit: int = 2,
    fold: str = "bbit",
) -> np.ndarray:
    """
    Bit fingerprint from precomputed side token counts.

    Gives the same result as :func:`synrfp` for the same tokens, without
    re-tokenizing. Intended for experiments that sweep sketch settings.

    :param tokens_R: Reactant-side token counts (``collections.Counter``).
    :param tokens_P: Product-side token counts (``collections.Counter``).
    :return: ``uint8`` array of length ``bits``.
    :rtype: numpy.ndarray
    """
    sk, weighted, m = _make_sketcher(sketch, bits, m, seed, bbit)
    if weighted:
        engine = SynRFP(tokenizer=None, radius=0, weighted_sketch=sk)
    else:
        engine = SynRFP(tokenizer=None, radius=0, sketch=sk)
    res = engine.fingerprint_from_tokens(tokens_R, tokens_P, mode=mode)
    return np.asarray(_to_bits(res, sketch.lower(), bits, fold), dtype=np.uint8)


def _make_sketcher(sketch: str, bits: int, m: Optional[int], seed: int, bbit: int):
    """Instantiate a sketcher; returns ``(sketcher, weighted, m)``."""
    s = sketch.lower()
    if s in ("minhash", "cw") and m is None:
        if bits % (2**bbit):
            raise ValueError(f"bits={bits} must be divisible by 2**bbit={2**bbit}")
        m = bits // (2**bbit)
    if s == "srp":
        m = bits
    if s == "parity":
        return ParityFold(bits=bits, seed=seed), False, m
    if s == "minhash":
        return MinHashSketch(m=m, seed=seed), False, m
    if s == "cw":
        return CWSketch(m=m, seed=seed), True, m
    if s == "srp":
        return SRPSketch(m=m, seed=seed, normalize=True), True, m
    raise ValueError(
        f"Unknown sketch: {sketch!r} (choose 'parity', 'minhash', 'cw', or 'srp')"
    )


def _to_bits(res: SynRFPResult, sketch_lower: str, bits: int, fold: str) -> List[int]:
    """Convert an engine result to a 0/1 list of length ``bits``."""
    if sketch_lower == "parity":
        return res.to_binary()
    sig = res.as_array()
    if sketch_lower == "srp":
        return (np.asarray(sig) > 0).astype(np.uint8).tolist()
    return signature_to_bits(sig, bits, method=fold)
