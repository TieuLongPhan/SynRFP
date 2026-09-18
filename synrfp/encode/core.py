# synrfp/encode/core.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.tokenizers.nauty import NautyTokenizer
from synrfp.tokenizers.morgan import MorganTokenizer
from synrfp.tokenizers.path import PathTokenizer
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

    - A tokenizer (WL, Nauty, Morgan, Path) converts each side into multiset
      tokens.
    - Depending on ``mode``, either token *delta* (P−R) or *union* (R+P)
      is computed.
    - A sketcher (parity, minhash, cw, srp) converts the token set/weights into
      a fixed-size sketch.
    - The sketch is finally mapped into a binary vector of length ``bits``.

    Parameters
    ----------
    rsmi : str
        Reaction SMILES, e.g. ``"CCO>>C=C.O"``.
    tokenizer : str, default "wl"
        Which tokenizer to use:

        - ``"wl"``     : Weisfeiler–Lehman style tokenizer.
        - ``"nauty"``  : Nauty-based canonical labeling tokenizer.
        - ``"morgan"`` : Morgan/ECFP-style neighborhood tokenizer.
        - ``"path"``   : simple path-based tokenizer.

    radius : int, default 2
        Neighborhood radius for the tokenizer (ignored by some tokenizers if
        not applicable).
    sketch : str, default "parity"
        Which sketcher to use:

        - ``"parity"``  : parity-folding into a binary vector.
        - ``"minhash"`` : MinHash signature, then mapped to bits.
        - ``"cw"``      : count-weighted sketch, then mapped to bits.
        - ``"srp"``     : signed random projection sketch (cosine-oriented).

    bits : int, default 1024
        Length of the final binary fingerprint.
        For ``sketch="parity"``, this is the internal bit-length of
        :class:`ParityFold`. For ``"minhash"``, ``"cw"`` and ``"srp"``, it
        controls the final bin count used by :func:`signature_to_bits`.

    m : int or None, default None
        Number of samples for MinHash or CWSketch. ``None`` sets
        ``m = bits // 2**bbit`` so that the b-bit encoding fills exactly
        ``bits`` positions. SRP always uses ``m = bits`` (one sign per bit).
    seed : int, default 1
        Random seed for reproducibility.
    mode : {"delta", "union"}, default "delta"
        Token combination mode:

        - ``"delta"`` : signed difference P−R.
        - ``"union"`` : union of tokens appearing on either side.

    node_attrs : Sequence[str] or None, optional
        Node attribute names passed to the tokenizer (e.g. ``["element"]``).
    edge_attrs : Sequence[str] or None, optional
        Edge attribute names passed to the tokenizer (e.g. ``["order"]``).
    bbit : int, default 2
        Bits per signature sample for the b-bit encoding of MinHash/CWS.
    fold : {"bbit", "mod"}, default "bbit"
        Signature-to-bits mapping; ``"mod"`` reproduces the legacy behaviour.
    agents : {"both", "reactants", "drop"}, default "both"
        Placement of agents in ``reactants>agents>products``; see
        :meth:`synrfp.graph.reaction.Reaction.from_rsmi`.

    Returns
    -------
    list[int]
        Fingerprint as a list of 0/1 bits of length ``bits``.

    Raises
    ------
    ValueError
        On invalid ``tokenizer``, ``sketch``, or ``mode`` names.
    RuntimeError
        If required dependencies (e.g. ``pynauty`` or ``datasketch``) are missing.
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
    else:
        raise ValueError(
            f"Unknown tokenizer: {tokenizer!r} "
            "(choose 'wl', 'nauty', 'morgan', or 'path')"
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
    :returns: ``uint8`` array of length ``bits``.
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
