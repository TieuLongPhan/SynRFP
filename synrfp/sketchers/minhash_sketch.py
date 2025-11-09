# synrfp/sketchers/minhash_sketch.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from synrfp.sketchers.base import BaseSketch

try:
    from datasketch import MinHash

    _HAVE_DS = True
except Exception:
    _HAVE_DS = False


class MinHashSketch(BaseSketch):
    """
    Classical MinHash for Jaccard estimation with a dependency-free fallback.

    :param m: Number of permutations (hash functions), > 0.
    :type m: int
    :param seed: Random seed.
    :type seed: int
    :raises ValueError: If ``m`` not positive.

    Example
    -------
    >>> mh = MinHashSketch(m=128, seed=0)
    >>> hv = mh.build([1,2,3,4])
    >>> len(hv) == 128
    True
    """

    _PRIME64: int = (1 << 61) - 1

    def __init__(self, m: int = 256, seed: int = 1):
        if not isinstance(m, int) or m <= 0:
            raise ValueError("m must be a positive integer")
        super().__init__(seed=seed)
        self.m = int(m)
        rng = np.random.default_rng(seed)
        self._ab: List[Tuple[np.uint64, np.uint64]] = [
            (
                np.uint64(rng.integers(1, self._PRIME64)),
                np.uint64(rng.integers(0, self._PRIME64)),
            )
            for _ in range(self.m)
        ]

    def __repr__(self) -> str:
        return f"MinHashSketch(m={self.m}, seed={self.seed})"

    def _fallback(self, tokens: Iterable[int]) -> List[int]:
        toks = np.fromiter((int(t) for t in set(tokens)), dtype=np.uint64)
        if toks.size == 0:
            return [0] * self.m
        P = np.uint64(self._PRIME64)
        mins = np.full(self.m, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        for i, (a, b) in enumerate(self._ab):
            vals = (a * toks + b) % P
            mins[i] = np.min(vals)
        return [int(x) for x in mins]

    def build(self, support: Iterable[int]) -> List[int]:
        if _HAVE_DS:
            mh = MinHash(num_perm=self.m, seed=self.seed)
            for h in set(int(x) for x in support):
                mh.update(str(h).encode("utf-8"))
            return list(mh.hashvalues)
        return self._fallback(support)

    def describe(self) -> str:  # noqa: D401
        """Return a brief usage example."""
        return "MinHashSketch(m=256, seed=0).build(tokens)\n"
