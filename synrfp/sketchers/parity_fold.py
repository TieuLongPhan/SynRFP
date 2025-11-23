# synrfp/sketchers/parity_fold.py
from __future__ import annotations

from typing import Iterable

import numpy as np

from synrfp.sketchers.base import BaseSketch
from synrfp.tokenizers.utils import _h64


class ParityFold(BaseSketch):
    """
    Parity-based folding sketcher (unweighted tokens → binary bit vector).

    Each token ``t`` is mapped to a bit index via a deterministic 64-bit hash
    and the sketch's seed, then the bit is toggled (XOR). If a token appears
    an even number of times, its contribution cancels out; odd multiplicities
    flip the corresponding bit.

    The result is a compact binary fingerprint useful for fast similarity
    via Hamming distance or Tanimoto over bits.

    :param bits: Length of the binary sketch (number of bits).
    :type bits: int
    :param seed: Non-negative integer seed for the internal hash mapping.
    :type seed: int
    :raises ValueError: If ``bits`` is not positive or ``seed`` is negative.
    """

    def __init__(self, bits: int = 2048, seed: int = 0) -> None:
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits must be a positive integer")
        super().__init__(seed=seed)
        self.bits: int = int(bits)

    def __repr__(self) -> str:
        return f"ParityFold(bits={self.bits}, seed={self.seed})"

    def build(self, support: Iterable[int]) -> np.ndarray:
        """
        Build a parity-folded binary sketch from an unweighted token stream.

        Internally, this:

        1. Collapses *support* to counts via :class:`collections.Counter`.
        2. Retains only tokens with odd multiplicity (parity 1).
        3. Maps each such token ``t`` to an index
           ``idx = _h64(('pf', t), seed=seed) % bits``.
        4. Sets the corresponding bits, yielding a 0/1 vector.

        :param support: Iterable of integer tokens.
        :type support: Iterable[int]
        :returns: Binary sketch of length ``bits`` (dtype ``uint8``).
        :rtype: numpy.ndarray
        """
        counts = self._as_counter(support)
        if not counts:
            return np.zeros(self.bits, dtype=np.uint8)

        indices = [
            _h64(("pf", int(t)), seed=self.seed) % self.bits
            for t, c in counts.items()
            if c & 1  # only odd parity contributes
        ]
        if not indices:
            return np.zeros(self.bits, dtype=np.uint8)

        v = np.bincount(np.fromiter(indices, int), minlength=self.bits) & 1
        return v.astype(np.uint8, copy=False)

    @staticmethod
    def describe() -> str:
        """
        Return a brief usage example for :class:`ParityFold`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.parity_fold import ParityFold\n"
            "sk = ParityFold(bits=2048, seed=0)\n"
            "fp = sk.build(tokens)  # -> np.uint8[bits]\n"
        )
