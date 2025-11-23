# synrfp/sketchers/cw_sketch.py
from __future__ import annotations

from typing import Mapping

import numpy as np

from synrfp.sketchers.base import WeightedSketch, ArrayLike

try:
    from datasketch import WeightedMinHashGenerator  # type: ignore

    _HAVE_DS = True
except Exception:
    _HAVE_DS = False


class CWSketch(WeightedSketch):
    """
    Consistent Weighted Sampling (CWS) sketch for weighted Jaccard.

    This sketch operates on *signed* sparse multisets (``pos``, ``neg``). It
    converts them into a signed dense vector via
    :meth:`~synrfp.sketchers.base.WeightedSketch.dicts_to_dense`, splits into
    separate non-negative positive/negative channels, concatenates them, and
    applies Consistent Weighted Sampling.

    If :mod:`datasketch` is available, it delegates to
    :class:`datasketch.WeightedMinHashGenerator`. Otherwise it uses a
    deterministic ICWS-like fallback implementation.

    :param m: Number of CWS samples (length of the sketch).
    :type m: int
    :param seed: Random seed for the sampler.
    :type seed: int
    :param normalize: If True, dense helpers L1-normalize the signed vector
        prior to splitting. Scaling does not change the weighted Jaccard but
        can improve numerical stability.
    :type normalize: bool
    :raises ValueError: If arguments are invalid.
    """

    def __init__(self, m: int = 256, seed: int = 0, normalize: bool = True) -> None:
        super().__init__(m=m, seed=seed, normalize=normalize)

    def __repr__(self) -> str:
        return f"CWSketch(m={self._m}, seed={self._seed}, normalize={self._normalize})"

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def build(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> ArrayLike:
        """
        Build a length-``m`` CWS hash signature for a signed multiset.

        Internally this:

        1. Uses :meth:`dicts_to_dense` with ``ensure_signed=True`` to obtain
           a 1D signed dense vector.
        2. Splits it into non-negative positive/negative arrays using
           :meth:`WeightedSketch.signed_to_pos_neg_arrays`.
        3. Concatenates these into a non-negative weight vector.
        4. Applies either :mod:`datasketch` or a deterministic fallback
           to draw ``m`` CWS samples.

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts.
        :type neg: Mapping[int,int]
        :returns: Array of sampled indices (hash values) of length ``m``.
        :rtype: numpy.ndarray
        """
        # signed dense vector
        vec, index_map = self.dicts_to_dense(pos, neg, ensure_signed=True)
        if vec.size == 0:
            return np.zeros(self._m, dtype=np.uint64)

        # split into non-negative channels and concatenate
        w_pos, w_neg = self.signed_to_pos_neg_arrays(vec)
        w_pos = w_pos.astype(float, copy=False)
        w_neg = w_neg.astype(float, copy=False)
        weights = np.concatenate([w_pos, w_neg], axis=0)
        if weights.size == 0:
            return np.zeros(self._m, dtype=np.uint64)

        # datasketch backend
        if _HAVE_DS:
            gen = WeightedMinHashGenerator(
                len(weights), sample_size=self._m, seed=self._seed
            )
            mh = gen.minhash(weights)
            return mh.hashvalues.copy()

        # deterministic CWS fallback
        return self._fallback_cws(weights)

    # ------------------------------------------------------------------ #
    # deterministic fallback (ICWS-style)                                #
    # ------------------------------------------------------------------ #
    def _fallback_cws(self, weights: ArrayLike) -> ArrayLike:
        """
        Deterministic CWS implementation (vectorized over permutations).

        This follows the ICWS scheme (Ioffe, 2010) for non-negative weights.
        Zero weights are never selected.

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

        rng = np.random.default_rng(self._seed)
        out = np.empty(self._m, dtype=np.uint64)

        for i in range(self._m):
            # Gamma(2,1) via sum of two Exp(1): -log U1 - log U2
            U = rng.random((2, 2, n))
            r = -np.log(U[0, 0]) - np.log(U[0, 1])
            c = -np.log(U[1, 0]) - np.log(U[1, 1])
            beta = rng.random(n)

            t = np.floor(logw / r + beta)
            y = np.exp(r * (t - beta))
            a = c / (y * np.exp(r))

            # coordinates with w <= 0 (logw = -inf or nan) must never win
            a[~np.isfinite(logw)] = np.inf

            out[i] = np.uint64(np.argmin(a))

        return out

    @staticmethod
    def describe() -> str:
        """
        Return a brief usage example for :class:`CWSketch`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.cw_sketch import CWSketch\n"
            "cw = CWSketch(m=64, seed=1)\n"
            "sig = cw.build({10: 2, 20: 1}, {30: 1})  # -> np.uint64[m]\n"
        )
