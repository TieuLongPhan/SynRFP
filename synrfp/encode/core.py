# synrfp/encode/core.py
from __future__ import annotations

from typing import List, Optional, Sequence

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
    m: int = 256,
    seed: int = 1,
    mode: str = "delta",
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
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

    m : int, default 256
        Number of hash samples/projections for MinHash, CWSketch, or SRP.
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
    weighted = False
    if sketch_lower == "parity":
        sk = ParityFold(bits=bits, seed=seed)
        weighted = False
    elif sketch_lower == "minhash":
        sk = MinHashSketch(m=m, seed=seed)
        weighted = False
    elif sketch_lower == "cw":
        sk = CWSketch(m=m, seed=seed)
        weighted = True
    elif sketch_lower == "srp":
        sk = SRPSketch(m=m, seed=seed, normalize=True)
        weighted = True
    else:
        raise ValueError(
            f"Unknown sketch: {sketch!r} "
            "(choose 'parity', 'minhash', 'cw', or 'srp')"
        )

    # 3) Parse reaction graphs
    parsed = Reaction.from_rsmi(rsmi)
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
    if sketch_lower == "parity":
        # parity-fold already produces binary bits internally
        return res.to_binary()

    # MinHash / CWSketch / SRP: convert numeric/signature -> bit-vector of length `bits`
    sig = res.as_array()
    return signature_to_bits(sig, bits)
