# /mnt/data/srp_sketch.py
from __future__ import annotations

from typing import Mapping
import numpy as np

from synrfp.sketchers.base import WeightedSketch, ArrayLike


class SRPSketch(WeightedSketch):
    """
    Signed Random Projection (SRP) sketch for cosine similarity.

    Projects the signed dense vector into an m-dimensional random
    subspace and returns the sign pattern (+1/-1) as a compact sketch.

    :param m: Number of random projections (sketch length), > 0.
    :type m: int
    :param seed: RNG seed for projection matrix.
    :type seed: int
    :param normalize: If True, the dense helper L1-normalizes inputs before
        projection. For pure cosine similarity, scale does not matter, but
        normalization can improve numerical stability.
    :type normalize: bool
    :param memory_safe_threshold: If the product (m * n) of projection
        dimension times input dimension exceeds this threshold, SRP will
        compute projections row-by-row to avoid allocating the full
        (m, n) projection matrix. Set to 0 to always use the row-by-row
        strategy.
    :type memory_safe_threshold: int
    """

    def __init__(
        self,
        m: int = 256,
        seed: int = 0,
        normalize: bool = True,
        memory_safe_threshold: int = 50_000_000,
    ):
        super().__init__(m=m, seed=seed, normalize=normalize)
        # memory_safe_threshold is (m * n) threshold; if m*n > threshold,
        #  use row-wise projection
        self._memory_safe_threshold = int(memory_safe_threshold)

    def __repr__(self) -> str:
        return (
            f"SRPSketch(m={self._m}, seed={self._seed}, "
            f"normalize={self._normalize}, memory_safe_threshold={self._memory_safe_threshold})"
        )

    def build(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> ArrayLike:
        """
        Build an SRP sketch for the signed multiset (pos - neg).

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts.
        :type neg: Mapping[int,int]
        :returns: Sketch array of shape (m,) with entries in {-1,+1} (dtype np.int8).
        :rtype: numpy.ndarray
        """
        # signed dense vector; length n depends on union of pos,neg keys
        vec, index_map = self.dicts_to_dense(pos, neg, ensure_signed=True)
        n = int(vec.size)
        if n == 0:
            return np.zeros(self._m, dtype=np.int8)

        # ensure vec is float dtype expected by projections
        vec = np.asarray(vec, dtype=self._dtype, copy=False)

        # decide whether to allocate full R matrix or stream row-by-row
        use_rowwise = (self._memory_safe_threshold > 0) and (
            self._m * n > self._memory_safe_threshold
        )

        rng = np.random.default_rng(self._seed)

        if not use_rowwise:
            # allocate full random projection matrix R ~ N(0,1)
            R = rng.standard_normal((self._m, n), dtype=self._dtype)
            proj = R @ vec  # shape (m,)
        else:
            # memory-safe: compute projection one row at a time
            proj = np.empty(self._m, dtype=self._dtype)
            for i in range(self._m):
                r = rng.standard_normal(n, dtype=self._dtype)
                proj[i] = float(np.dot(r, vec))

        # sign -> +1 / -1, return as int8 for compactness
        sketch = np.where(proj >= 0.0, 1, -1).astype(np.int8, copy=False)
        return sketch

    @staticmethod
    def describe() -> str:
        """
        Return usage example for SRPSketch.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.srp_sketch import SRPSketch\n"
            "sk = SRPSketch(m=256, seed=0)\n"
            "sig = sk.build({10:2, 24:1}, {5:1})  # -> np.int8 array of +/-1\n"
        )

    @staticmethod
    def approx_cosine_from_sketch(a: np.ndarray, b: np.ndarray) -> float:
        """
        Approximate cosine similarity from two SRP sketches.

        For SRP sign sketches s_a, s_b in {-1,+1}:

            E[ sign(Rx) == sign(Ry) ] = 1 - arccos(sim(x,y))/pi
        and a simple unbiased estimator for cosine(sim) is:

            cos_est ≈ 2 * mean(s_a == s_b) - 1

        :param a: SRP sketch (np.ndarray of int8, +/-1).
        :param b: SRP sketch (same shape as a).
        :returns: Estimated cosine similarity in [-1, 1].
        :rtype: float
        """
        if a.shape != b.shape:
            raise ValueError("Sketch shapes must match")
        # fraction of equal signs
        frac = float(np.mean(a == b))
        return 2.0 * frac - 1.0
