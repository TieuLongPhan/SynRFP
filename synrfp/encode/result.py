# synrfp/encode/result.py
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import List

import numpy as np

from synrfp.utils import sketch_to_array, sketch_to_binary


@dataclass
class SynRFPResult:
    """
    Container for outputs of a single fingerprinting call.

    :param tokens_R: Token multiset for the reactant graph.
    :type tokens_R: collections.Counter
    :param tokens_P: Token multiset for the product graph.
    :type tokens_P: collections.Counter
    :param delta: Token counts summarising the transformation, depending on
                  ``mode``:
                  - if ``mode='delta'``: signed difference P−R
                  - if ``mode='union'``: union counts (R+P)
    :type delta: collections.Counter
    :param support: List of token keys with nonzero contribution (delta or union).
    :type support: list[int]
    :param sketch: Sketch object (bytes, list, or array) from the compressor.
    :type sketch: object
    :param mode: Fingerprint mode, either ``'delta'`` or ``'union'``.
    :type mode: str
    """

    tokens_R: Counter
    tokens_P: Counter
    delta: Counter
    support: List[int]
    sketch: object
    mode: str = "delta"

    def __repr__(self) -> str:
        return (
            f"SynRFPResult("
            f"tokens_R={sum(self.tokens_R.values())} tokens, "
            f"tokens_P={sum(self.tokens_P.values())} tokens, "
            f"support={len(self.support)}, "
            f"mode={self.mode!r}, "
            f"sketch_type={type(self.sketch).__name__}"
            f")"
        )

    @staticmethod
    def describe() -> str:
        """
        Example usage::

            >>> # assume `res` is a SynRFPResult
            >>> print(res)
            SynRFPResult(tokens_R=10 tokens, tokens_P=8 tokens,
            support=3, mode='delta', sketch_type=bytearray)

        :returns: Example usage string.
        :rtype: str
        """
        return (
            ">>> res = SynRFP(...).fingerprint(reactant_G, product_G)\n"
            ">>> print(res)\n"
            "SynRFPResult(tokens_R=..., tokens_P=..., support=..., "
            "mode='delta'|'union', sketch_type=...)\n"
        )

    def to_binary(self) -> List[int]:
        """
        Return the sketch stored in this result as a plain list of 0/1 bits.

        Only works for binary sketchers (e.g. ParityFold). For non-binary
        sketchers (MinHash, CWSketch, SRP) a :class:`TypeError` is raised.

        :returns: Binary fingerprint as list of 0/1 bits.
        :rtype: list[int]
        :raises TypeError: If the underlying sketch cannot be interpreted as bits.
        """
        return sketch_to_binary(self.sketch)

    def as_array(self) -> np.ndarray:
        """
        Return the underlying sketch as a 1D numpy integer array.

        This works for all sketcher types:
          - ParityFold: 0/1 array
          - MinHashSketch: hash values
          - CWSketch: sample indices
          - SRPSketch: sign pattern (+1/-1)

        :returns: 1D numpy array representation of the sketch.
        :rtype: numpy.ndarray
        """
        return sketch_to_array(self.sketch)
