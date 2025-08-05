# synrfp/sketchers/cw_sketch.py

from typing import Dict, Tuple
import numpy as np

from synrfp.sketchers.base import WeightedSketch

try:
    from datasketch import WeightedMinHashGenerator

    _HAVE_DATASKETCH = True
except ImportError:
    _HAVE_DATASKETCH = False


class CWSketch(WeightedSketch):
    """
    Consistent Weighted Sampling (Weighted MinHash).

    Approximates weighted Jaccard similarity over positive/negative namespaces
    using a WeightedMinHashGenerator.

    :param m: Number of hash samples (permutation count).
    :type m: int
    :param seed: Random seed for reproducibility.
    :type seed: int

    :raises RuntimeError: If `datasketch` is not installed.
    :raises ValueError: If `m` or `seed` are invalid.

    :example:
        >>> from synrfp.sketchers.cw_sketch import CWSketch
        >>> cw = CWSketch(m=128, seed=42)
        >>> pos = {100: 2, 200: 1}
        >>> neg = {150: 1}
        >>> sketch = cw.build(pos, neg)
        >>> len(sketch)  # = 128
    """

    def __init__(self, m: int = 256, seed: int = 1):
        if not _HAVE_DATASKETCH:
            raise RuntimeError(
                "`datasketch` is not installed; install it to use CWSketch."
            )
        if not isinstance(m, int) or m <= 0:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

        self.m = m
        self.seed = seed

    def __repr__(self) -> str:
        return f"CWSketch(m={self.m}, seed={self.seed})"

    @staticmethod
    def describe() -> str:
        """
        Return a usage example for CWSketch.

        :returns: Example instantiation and usage code.
        :rtype: str
        """
        return (
            ">>> from synrfp.sketchers.cw_sketch import CWSketch\n"
            ">>> cw = CWSketch(m=256, seed=1)\n"
            ">>> pos = {h1: 2, h2: 1}\n"
            ">>> neg = {h3: 1}\n"
            ">>> sketch = cw.build(pos, neg)\n"
        )

    def build(self, pos: Dict[int, int], neg: Dict[int, int]) -> np.ndarray:
        """
        Build a weighted sketch from positive and negative token counts.

        :param pos: Mapping from token to positive count.
        :type pos: Dict[int, int]
        :param neg: Mapping from token to negative count.
        :type neg: Dict[int, int]
        :returns: NumPy array of hash values of length `m`.
        :rtype: numpy.ndarray

        :raises TypeError: If `pos` or `neg` are not dicts.
        """
        if not isinstance(pos, dict) or not isinstance(neg, dict):
            raise TypeError("pos and neg must be dicts mapping int to int")

        # Empty‐input shortcut: return a zero‐array
        if not pos and not neg:
            import numpy as _np

            return _np.zeros(self.m, dtype="uint64")

        # Flatten features into weights list with namespace keys
        index: Dict[Tuple[str, int], int] = {}
        weights: list[float] = []

        def _add(namespace: str, token: int, count: int):
            if count <= 0:
                return
            key = (namespace, int(token))
            idx = index.get(key)
            if idx is None:
                idx = len(index)
                index[key] = idx
                weights.append(0.0)
            weights[idx] = float(count)

        for token, count in pos.items():
            _add("p", token, count)
        for token, count in neg.items():
            _add("n", token, count)

        # Build the Weighted MinHash sketch
        wmg = WeightedMinHashGenerator(len(weights), sample_size=self.m, seed=self.seed)
        mh = wmg.minhash(weights)
        return mh.hashvalues.copy()
