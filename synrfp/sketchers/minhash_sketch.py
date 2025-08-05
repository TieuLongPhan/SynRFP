# -----------------------------------------------------------------------------
# synrfp/sketchers/minhash_sketch.py
from typing import Iterable, List
from synrfp.sketchers.base import BaseSketch

try:
    from datasketch import MinHash

    _HAVE_DATASKETCH = True
except ImportError:
    _HAVE_DATASKETCH = False


class MinHashSketch(BaseSketch):
    """
    **MinHashSketch** — classical MinHash for (weighted) Jaccard estimation.

    :param m: Number of permutations (hash functions).
    :type m: int
    :param seed: Random seed.
    :type seed: int

    Requires the *datasketch* library.
    """

    def __init__(self, m: int = 256, seed: int = 1):
        if not _HAVE_DATASKETCH:
            raise RuntimeError("`datasketch` must be installed for MinHashSketch")
        if m <= 0:
            raise ValueError("m must be positive")
        super().__init__(seed)
        self.m = int(m)

    def __repr__(self) -> str:  # noqa: D401
        return f"MinHashSketch(m={self.m}, seed={self.seed})"

    def build(self, support: Iterable[int]) -> List[int]:  # noqa: D401
        mh = MinHash(num_perm=self.m, seed=self.seed)
        for h in support:
            mh.update(str(int(h)).encode("utf-8"))
        return list(mh.hashvalues)

    def describe(self) -> str:  # noqa: D401
        return "mh = MinHashSketch(m=256, seed=0)\n" "sk = mh.build(tokens)\n"
