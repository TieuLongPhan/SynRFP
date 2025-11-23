# synrfp/sketchers/minhash_sketch.py
from __future__ import annotations

from typing import Iterable, List, Set

from synrfp.sketchers.base import BaseSketch
from synrfp.tokenizers.utils import _h64

try:
    from datasketch import MinHash  # type: ignore

    _HAVE_DS = True
except Exception:
    _HAVE_DS = False


class MinHashSketch(BaseSketch):
    """
    Set-based MinHash sketch for approximating Jaccard similarity.

    This sketch treats the input as a *set* of tokens (multiplicities are
    ignored). It computes a fixed-length signature such that the fraction of
    shared components approximates the Jaccard index between two sets.

    If :mod:`datasketch` is available and ``use_datasketch`` is True, the
    implementation delegates to :class:`datasketch.MinHash`. Otherwise a
    deterministic fallback based on repeated 64-bit hashing is used.

    :param m: Number of hash permutations (length of the sketch).
    :type m: int
    :param seed: Non-negative integer seed for all hash permutations.
    :type seed: int
    :param use_datasketch: Whether to use :mod:`datasketch` if available.
    :type use_datasketch: bool
    :raises ValueError: If ``m`` is not positive or ``seed`` is negative.
    """

    def __init__(
        self, m: int = 256, seed: int = 0, use_datasketch: bool = True
    ) -> None:
        if not isinstance(m, int) or m <= 0:
            raise ValueError("m must be a positive integer")
        super().__init__(seed=seed)
        self.m: int = int(m)
        self.use_datasketch: bool = bool(use_datasketch)

    def __repr__(self) -> str:
        return (
            f"MinHashSketch(m={self.m}, seed={self.seed}, "
            f"use_datasketch={self.use_datasketch})"
        )

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _fallback(self, tokens: Set[int]) -> List[int]:
        """
        Deterministic MinHash fallback using 64-bit hashing.

        For each permutation index ``i = 0..m-1``, this computes

        .. math::

            h_i(S) = \\min_{t \\in S} H(i, t)

        where :math:`H` is a 64-bit hash seeded with ``self.seed``.

        :param tokens: Set of distinct tokens.
        :type tokens: set[int]
        :returns: List of length ``m`` of integer hash values.
        :rtype: list[int]
        """
        if not tokens:
            return [0] * self.m

        tokens_list = [int(t) for t in tokens]
        sig: List[int] = []
        for i in range(self.m):
            current_min = None
            for t in tokens_list:
                h = _h64(("mh", i, t), seed=self.seed)
                if current_min is None or h < current_min:
                    current_min = h
            sig.append(int(current_min if current_min is not None else 0))
        return sig

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def build(self, support: Iterable[int]) -> List[int]:
        """
        Build a MinHash signature from an unweighted token stream.

        Multiplicities in *support* are ignored; only distinct tokens contribute
        to the sketch.

        :param support: Iterable of integer tokens.
        :type support: Iterable[int]
        :returns: MinHash signature as a list of length ``m``.
        :rtype: list[int]
        """
        tokens = set(int(x) for x in support)
        if not tokens:
            return [0] * self.m

        if _HAVE_DS and self.use_datasketch:
            mh = MinHash(num_perm=self.m, seed=self.seed)
            for t in tokens:
                mh.update(str(int(t)).encode("utf-8"))
            # datasketch returns np.ndarray[uint64]; convert to Python ints
            return [int(x) for x in mh.hashvalues]

        return self._fallback(tokens)

    @staticmethod
    def describe() -> str:
        """
        Return a brief usage example for :class:`MinHashSketch`.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            "from synrfp.sketchers.minhash_sketch import MinHashSketch\n"
            "sk = MinHashSketch(m=256, seed=0)\n"
            "sig = sk.build(tokens)  # -> list[int] of length m\n"
        )
