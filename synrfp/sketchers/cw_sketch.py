# synrfp/sketchers/cw_sketch.py
from __future__ import annotations

from typing import Mapping
import numpy as np

from synrfp.sketchers.base import WeightedSketch

try:
    from datasketch import WeightedMinHashGenerator

    _HAVE_DS = True
except Exception:
    _HAVE_DS = False


class CWSketch(WeightedSketch):
    """
    Consistent Weighted Sampling (Ioffe, 2010) with deterministic fallback.

    Supports signed inputs by splitting the signed vector into two non-negative
    channels (pos, neg) and sketching their concatenation.

    :param m: Number of samples (permutations), > 0.
    :type m: int
    :param seed: Random seed.
    :type seed: int
    :param normalize: If True, dense helpers L1-normalize inputs.
    :type normalize: bool
    :raises ValueError: If arguments invalid.

    Example
    -------
    >>> cw = CWSketch(m=64, seed=1)
    >>> sk = cw.build({10:2, 20:1}, {30:1})
    >>> len(sk) == 64
    True
    """

    def __init__(self, m: int = 256, seed: int = 0, normalize: bool = True):
        super().__init__(m=m, seed=seed, normalize=normalize)

    def __repr__(self) -> str:
        return f"CWSketch(m={self._m}, seed={self._seed}, normalize={self._normalize})"

    # --------------------------- public API -------------------------------
    def build(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> np.ndarray:
        """
        Build a length-``m`` weighted hash signature.

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts.
        :type neg: Mapping[int,int]
        :returns: Array of sampled indices (hash values) of length ``m``.
        :rtype: numpy.ndarray
        """
        # Convert to signed dense, then split to two nonnegative channels.
        signed, _ = self.dicts_to_dense(pos, neg, ensure_signed=True)
        w_pos, w_neg = self.signed_to_pos_neg_arrays(signed)
        weights = np.concatenate([w_pos, w_neg], axis=0)
        if weights.size == 0:
            return np.zeros(self._m, dtype=np.uint64)

        if _HAVE_DS:
            gen = WeightedMinHashGenerator(
                len(weights), sample_size=self._m, seed=self._seed
            )
            mh = gen.minhash(weights)
            return mh.hashvalues.copy()

        return self._fallback_cws(weights)

    # --------------------------- deterministic fallback -------------------
    def _fallback_cws(self, weights: np.ndarray) -> np.ndarray:
        """
        Deterministic CWS implementation (vectorized).

        :param weights: Non-negative weights (1D array).
        :type weights: numpy.ndarray
        :returns: Hashvalues array (uint64) length ``m``.
        :rtype: numpy.ndarray
        """
        w = np.asarray(weights, dtype=float)
        n = w.size
        if n == 0:
            return np.zeros(self._m, dtype=np.uint64)

        # log(w); zero-weights -> -inf so never selected
        with np.errstate(divide="ignore"):
            logw = np.log(w)

        out = np.empty(self._m, dtype=np.uint64)

        # For each permutation, sample r,c ~ Gamma(2,1), beta ~ U(0,1) per feature
        # Using a deterministic RNG seeded by (seed + perm)
        for i in range(self._m):
            rng = np.random.default_rng(self._seed + i)
            U = rng.random(
                (3, 2, n), dtype=np.float64
            )  # for r,c: sum of two Exp(1); for beta: later
            # r, c via sum of two Exponential(1): -log U1 - log U2
            r = -np.log(U[0, 0]) - np.log(U[0, 1])
            c = -np.log(U[1, 0]) - np.log(U[1, 1])
            beta = rng.random(n)

            t = np.floor(logw / r + beta)
            y = np.exp(r * (t - beta))
            a = c / (y * np.exp(r))
            a[~np.isfinite(logw)] = np.inf

            out[i] = np.uint64(np.argmin(a))

        return out
