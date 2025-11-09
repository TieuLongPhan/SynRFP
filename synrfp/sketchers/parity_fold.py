# synrfp/sketchers/parity_fold.py
from __future__ import annotations

from typing import Iterable
import numpy as np

from synrfp.sketchers.base import BaseSketch


class ParityFold(BaseSketch):
    """
    **ParityFold** — XOR-fold tokens into a compact binary sketch.

    Each token ``t`` toggles bit ``idx = hash((seed, t)) % bits``. The result is
    a 0/1 vector. Parity helps mitigate heavy-collision bias vs simple counts.

    :param bits: Length of the bit array, > 0.
    :type bits: int
    :param seed: Random seed influencing the hashing of indices.
    :type seed: int
    :raises ValueError: If ``bits`` <= 0.

    Example
    -------
    >>> pf = ParityFold(bits=1024, seed=42)
    >>> sk = pf.build([1,2,2,3]); (sk.sum() >= 0) and (len(sk) == 1024)
    True
    """

    def __init__(self, bits: int = 1024, seed: int = 1):
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits must be a positive integer")
        super().__init__(seed=seed)
        self.bits = int(bits)

    def __repr__(self) -> str:
        return f"ParityFold(bits={self.bits}, seed={self.seed})"

    def build(self, support: Iterable[int]) -> np.ndarray:
        idx = [hash((self.seed, int(h))) % self.bits for h in support]
        if not idx:
            return np.zeros(self.bits, dtype=np.uint8)
        v = np.bincount(np.fromiter(idx, int), minlength=self.bits) & 1
        return v.astype(np.uint8, copy=False)

    def describe(self) -> str:  # noqa: D401
        """Return a brief usage example."""
        return "ParityFold(bits=2048, seed=0).build(tokens)  # -> np.uint8[bits]\n"
