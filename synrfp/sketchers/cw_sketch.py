from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from synrfp.sketchers.base import WeightedSketch, ArrayLike
from synrfp.sketchers._hashing import mix, to_uint64_keys, uniform_open01

# channel tags keep production (+) and consumption (-) of a token distinct
_POS = np.uint64(0x2B)
_NEG = np.uint64(0x2D)


class CWSketch(WeightedSketch):
    """
    Consistent Weighted Sampling (CWS) sketch for weighted Jaccard similarity.

    The signed multiset (``pos``, ``neg``) is sign-lifted into two
    non-negative channels, so the same token produced and consumed yields two
    different keys. Improved CWS (Ioffe, 2010) is then applied with random
    variables that are a pure function of ``(seed, sample, key)``. The same
    token therefore receives the same randomness in every reaction, which is
    what makes

    .. math::
        \\Pr[s_i(w) = s_i(w')] = J_w(w, w')

    hold across reactions.

    Each sample is returned as one 64-bit hash of the selected
    ``(key, t)`` pair, so two signatures can be compared elementwise.

    :param m: Number of CWS samples (length of the signature).
    :type m: int
    :param seed: Non-negative seed.
    :type seed: int
    :param normalize: If True, L1-normalise the weights before sampling. This
        changes weighted Jaccard between reactions of different total weight,
        so it is off by default.
    :type normalize: bool
    """

    def __init__(self, m: int = 256, seed: int = 0, normalize: bool = False) -> None:
        super().__init__(m=m, seed=seed, normalize=normalize)

    def __repr__(self) -> str:
        return f"CWSketch(m={self._m}, seed={self._seed}, normalize={self._normalize})"

    def _lifted_weights(
        self, pos: Mapping[int, int], neg: Mapping[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(keys, weights)`` of the sign-lifted, non-negative channels."""
        self.validate_pos_neg(pos, neg)
        net = {}
        for t, c in pos.items():
            net[int(t)] = net.get(int(t), 0.0) + float(c)
        for t, c in neg.items():
            net[int(t)] = net.get(int(t), 0.0) - float(c)
        items = [(t, c) for t, c in net.items() if c != 0.0]
        if not items:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64)
        toks = to_uint64_keys(t for t, _ in items)
        vals = np.fromiter((c for _, c in items), dtype=np.float64)
        keys = mix(toks, np.where(vals > 0, _POS, _NEG))
        weights = np.abs(vals)
        if self._normalize:
            weights = weights / weights.sum()
        return keys, weights

    def build(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> ArrayLike:
        """
        Build a length-``m`` CWS signature for the signed multiset pos - neg.

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts (magnitudes).
        :type neg: Mapping[int,int]
        :return: ``uint64`` array of length ``m``; all zeros for an empty input.
        :rtype: numpy.ndarray
        """
        keys, w = self._lifted_weights(pos, neg)
        if keys.size == 0:
            return np.zeros(self._m, dtype=np.uint64)

        seed = np.uint64(self._seed)
        i = np.arange(self._m, dtype=np.uint64)[:, None]  # (m, 1)
        k = keys[None, :]  # (1, n)

        def u(stream: int) -> np.ndarray:
            return uniform_open01(mix(seed, i, k, np.uint64(stream)))

        r = -np.log(u(1)) - np.log(u(2))  # Gamma(2, 1)
        c = -np.log(u(3)) - np.log(u(4))  # Gamma(2, 1)
        beta = u(5)
        logw = np.log(w)[None, :]
        t = np.floor(logw / r + beta)
        y = np.exp(r * (t - beta))
        a = c / (y * np.exp(r))

        j = np.argmin(a, axis=1)
        rows = np.arange(self._m)
        t_sel = t[rows, j].astype(np.int64).view(np.uint64)
        return mix(keys[j], t_sel)

    @staticmethod
    def describe() -> str:
        """
        Return a brief usage example for :class:`CWSketch`.

        :return: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.cw_sketch import CWSketch\n"
            "cw = CWSketch(m=64, seed=1)\n"
            "sig = cw.build({10: 2, 20: 1}, {30: 1})  # -> np.uint64[m]\n"
        )
