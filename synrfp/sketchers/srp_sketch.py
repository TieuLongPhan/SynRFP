# synrfp/sketchers/srp_sketch.py
from __future__ import annotations

from typing import Mapping
import numpy as np

from synrfp.sketchers.base import WeightedSketch, ArrayLike
from synrfp.sketchers._hashing import gaussian, mix, to_uint64_keys


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

        The Gaussian projection entry for token ``t`` in row ``i`` is a pure
        function of ``(seed, i, t)``, so the same token is projected
        identically in every reaction and the sign agreement between two
        sketches estimates their angular (cosine) similarity.

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts (magnitudes).
        :type neg: Mapping[int,int]
        :return: ``int8`` array of +1/-1 of length ``m``; all +1 when empty.
        :rtype: numpy.ndarray
        """
        self.validate_pos_neg(pos, neg)
        net = {}
        for t, c in pos.items():
            net[int(t)] = net.get(int(t), 0.0) + float(c)
        for t, c in neg.items():
            net[int(t)] = net.get(int(t), 0.0) - float(c)
        items = [(t, c) for t, c in net.items() if c != 0.0]
        if not items:
            return np.ones(self._m, dtype=np.int8)
        keys = to_uint64_keys(t for t, _ in items)
        vec = np.fromiter((c for _, c in items), dtype=np.float64)
        if self._normalize:
            vec = vec / np.abs(vec).sum()

        seed = np.uint64(self._seed)
        proj = np.empty(self._m, dtype=np.float64)
        # stream rows in blocks to bound memory for large supports
        block = max(1, self._memory_safe_threshold // max(1, keys.size))
        for start in range(0, self._m, block):
            rows = np.arange(start, min(self._m, start + block), dtype=np.uint64)
            i = rows[:, None]
            g = gaussian(
                mix(seed, i, keys[None, :], np.uint64(1)),
                mix(seed, i, keys[None, :], np.uint64(2)),
            )
            stop = start + rows.size
            proj[start:stop] = g @ vec
        return np.where(proj >= 0.0, 1, -1).astype(np.int8, copy=False)

    @staticmethod
    def describe() -> str:
        """
        Return usage example for SRPSketch.

        :return: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.srp_sketch import SRPSketch\n"
            "sk = SRPSketch(m=256, seed=0)\n"
            "sig = sk.build({10:2, 24:1}, {5:1})  # -> np.int8 array of +/-1\n"
        )

    @staticmethod
    def approx_cosine_from_sketch(a: np.ndarray, b: np.ndarray) -> float:
        r"""
        Approximate cosine similarity from two SRP sketches.

        For SRP sign sketches ``s_a``, ``s_b`` in ``{-1, +1}``:

        .. math::

           \mathbb{E}[\operatorname{sign}(Rx) == \operatorname{sign}(Ry)]
           = 1 - \frac{\arccos(\operatorname{sim}(x, y))}{\pi}

        An estimator for cosine similarity is:

        .. math::

           \cos_{\text{est}} \approx 2 \cdot \operatorname{mean}(s_a == s_b) - 1

        :param a: SRP sketch (np.ndarray of int8, +/-1).
        :param b: SRP sketch (same shape as a).
        :return: Estimated cosine similarity in [-1, 1].
        :rtype: float
        """
        if a.shape != b.shape:
            raise ValueError("Sketch shapes must match")
        # fraction of equal signs
        frac = float(np.mean(a == b))
        return 2.0 * frac - 1.0
