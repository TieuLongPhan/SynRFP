# synrfp/sketchers/base.py
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Mapping, Optional, Iterable, Any
from collections import Counter


class BaseSketch(ABC):
    """
    Abstract base class for set / multiset sketchers.

    Subclasses must implement :meth:`build` and may override :meth:`describe`.

    :param seed: Non-negative integer seed for reproducibility.
    :type seed: int
    :raises ValueError: If ``seed`` is negative or not an integer.

    Example
    -------
    >>> class Dummy(BaseSketch):
    ...     def build(self, support): return Counter(support)
    ...
    >>> sk = Dummy(seed=1)
    >>> C = sk.build([1, 2, 2, 3]); C[2]
    2
    """

    def __init__(self, seed: int = 1):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed

    @abstractmethod
    def build(self, support: Iterable[int]) -> Any:
        """
        Build a sketch from an *unweighted* iterable of integer tokens.

        :param support: Iterable of integer-encoded features (can repeat).
        :type support: Iterable[int]
        :returns: Sketch object (type depends on subclass).
        :rtype: Any
        """
        raise NotImplementedError

    def describe(self) -> str:  # noqa: D401
        """Return a short usage example."""
        return (
            f"sketcher = {self.__class__.__name__}(seed=1)\n"
            "sketch = sketcher.build(tokens)\n"
        )

    def _as_counter(self, support: Iterable[int]) -> Counter:
        """
        Convert *support* to a :class:`collections.Counter`.

        :param support: Iterable of tokens.
        :type support: Iterable[int]
        :returns: Token multiplicities.
        :rtype: Counter
        """
        return Counter(int(x) for x in support)


ArrayLike = np.ndarray


class WeightedSketch(ABC):
    """
    Abstract base for *weighted* (signed) sketchers.

    Utilities provided:
      - input validation for pos/neg sparse multisets,
      - deterministic sparse→dense conversion,
      - signed / two-channel (pos,neg) representations,
      - exact reference similarities (weighted-Jaccard, cosine),
      - fluent config for normalization and dtype.

    Concrete subclasses must implement :meth:`build`.

    :param m: (Optional) number of sketch samples (backend may use it).
    :type m: int
    :param seed: RNG seed for deterministic behavior.
    :type seed: int
    :param normalize: If True, helpers can L1-normalize outputs.
    :type normalize: bool
    :raises ValueError: If parameters are invalid.

    Example
    -------
    >>> class Echo(WeightedSketch):
    ...     def build(self, pos, neg): return self.dicts_to_dense(pos, neg)[0]
    ...
    >>> es = Echo(m=4, seed=0)
    >>> vec, _ = es.dicts_to_dense({1:2},{2:1})
    >>> vec.sum() != 0
    True
    """

    def __init__(self, m: int = 256, seed: int = 0, normalize: bool = True):
        if not isinstance(m, int) or m <= 0:
            raise ValueError("m must be a positive integer")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean")
        self._m = int(m)
        self._seed = int(seed)
        self._normalize = bool(normalize)
        self._dtype = np.float64
        self._last_index_map: Optional[Dict[int, int]] = None

    # -------------------------- fluent setters --------------------------
    def set_normalize(self, normalize: bool) -> "WeightedSketch":
        """
        Set whether helpers produce L1-normalized arrays.

        :param normalize: True to enable L1 normalization.
        :type normalize: bool
        :returns: self
        :rtype: WeightedSketch
        """
        self._normalize = bool(normalize)
        return self

    def set_dtype(self, dtype: np.dtype) -> "WeightedSketch":
        """
        Configure dtype for dense arrays.

        :param dtype: NumPy dtype (e.g., np.float32, np.float64).
        :type dtype: numpy.dtype
        :returns: self
        :rtype: WeightedSketch
        """
        self._dtype = np.dtype(dtype)
        return self

    # ------------------------------ core API ----------------------------
    @abstractmethod
    def build(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> Any:
        """
        Build a sketch for the signed multiset (pos - neg).

        :param pos: Mapping token -> non-negative count.
        :type pos: Mapping[int, int]
        :param neg: Mapping token -> non-negative count.
        :type neg: Mapping[int, int]
        :returns: Implementation-defined sketch object.
        :rtype: Any
        """
        raise NotImplementedError

    # --------------------- validation & sparse→dense ---------------------
    def validate_pos_neg(self, pos: Mapping[int, int], neg: Mapping[int, int]) -> None:
        """
        Validate pos/neg dictionaries.

        :param pos: Positive token counts.
        :type pos: Mapping[int,int]
        :param neg: Negative token counts.
        :type neg: Mapping[int,int]
        :raises TypeError: If types invalid.
        :raises ValueError: If keys not int or counts negative.
        """
        if not isinstance(pos, Mapping) or not isinstance(neg, Mapping):
            raise TypeError("pos and neg must be mapping-like (dict[int,int])")
        for name, d in (("pos", pos), ("neg", neg)):
            for k, v in d.items():
                if not isinstance(k, int):
                    raise ValueError(f"{name} keys must be int; got {type(k)}")
                if not (isinstance(v, int) or isinstance(v, float)):
                    raise ValueError(f"{name}[{k!r}] must be numeric")
                if v < 0:
                    raise ValueError(f"{name}[{k!r}] must be non-negative")

    def _union_index(self, *maps: Mapping[int, int]) -> Dict[int, int]:
        """
        Deterministically map tokens to indices.

        :param maps: One or more maps token->count.
        :type maps: Mapping[int,int]
        :returns: token->index mapping sorted by token id.
        :rtype: Dict[int,int]
        """
        vocab = set()
        for m in maps:
            vocab.update(int(k) for k in m.keys())
        return {tok: i for i, tok in enumerate(sorted(vocab))}

    def dicts_to_dense(
        self,
        pos: Mapping[int, int],
        neg: Mapping[int, int],
        index_map: Optional[Dict[int, int]] = None,
        *,
        ensure_signed: bool = True,
    ) -> Tuple[ArrayLike, Dict[int, int]]:
        """
        Convert sparse pos/neg dicts into a dense array.

        :param pos: Positive counts.
        :type pos: Mapping[int,int]
        :param neg: Negative counts.
        :type neg: Mapping[int,int]
        :param index_map: Optional precomputed token->index map.
        :type index_map: Optional[Dict[int,int]]
        :param ensure_signed: If True return (n,) signed array (pos-neg); else
                              return (2,n) with [pos, neg] channels.
        :type ensure_signed: bool
        :returns: (array, index_map).
        :rtype: Tuple[numpy.ndarray, Dict[int,int]]
        """
        self.validate_pos_neg(pos, neg)

        if index_map is None:
            index_map = self._union_index(pos, neg)

        n = len(index_map)
        arr = np.zeros(n if ensure_signed else (2, n), dtype=self._dtype)

        if ensure_signed:
            for t, c in pos.items():
                arr[index_map[t]] += float(c)
            for t, c in neg.items():
                arr[index_map[t]] -= float(c)
            if self._normalize:
                s = float(np.sum(np.abs(arr)))
                if s > 0:
                    arr = arr / s
        else:
            for t, c in pos.items():
                arr[0, index_map[t]] += float(c)
            for t, c in neg.items():
                arr[1, index_map[t]] += float(c)
            if self._normalize:
                s = float(np.sum(arr))
                if s > 0:
                    arr = arr / s

        self._last_index_map = dict(index_map)
        return arr, dict(index_map)

    # ---------------------------- signed helpers -------------------------
    @staticmethod
    def signed_to_pos_neg_arrays(vec: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Split a signed vector into non-negative positive/negative arrays.

        :param vec: Signed vector.
        :type vec: numpy.ndarray
        :returns: (pos, neg) arrays, both >= 0.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        v = np.asarray(vec)
        pos = np.maximum(v, 0.0).astype(v.dtype, copy=False)
        neg = np.maximum(-v, 0.0).astype(v.dtype, copy=False)
        return pos, neg

    # --------------------------- exact similarities ----------------------
    @staticmethod
    def weighted_jaccard_signed(vec_a: ArrayLike, vec_b: ArrayLike) -> float:
        """
        Weighted-Jaccard for signed vectors via two-channel decomposition.

        :param vec_a: Signed vector A.
        :type vec_a: numpy.ndarray
        :param vec_b: Signed vector B.
        :type vec_b: numpy.ndarray
        :returns: Similarity in [0,1].
        :rtype: float
        """
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        if a.shape != b.shape:
            raise ValueError("Shape mismatch for weighted_jaccard_signed")
        ap, an = WeightedSketch.signed_to_pos_neg_arrays(a)
        bp, bn = WeightedSketch.signed_to_pos_neg_arrays(b)
        num = np.sum(np.minimum(ap, bp)) + np.sum(np.minimum(an, bn))
        den = np.sum(np.maximum(ap, bp)) + np.sum(np.maximum(an, bn))
        return float(num / den) if den > 0 else 0.0

    @staticmethod
    def cosine_similarity(vec_a: ArrayLike, vec_b: ArrayLike) -> float:
        """
        Cosine similarity (safe with zeros).

        :param vec_a: Vector A.
        :type vec_a: numpy.ndarray
        :param vec_b: Vector B.
        :type vec_b: numpy.ndarray
        :returns: Cosine in [-1,1].
        :rtype: float
        """
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def exact_similarities_from_dicts(
        self,
        pos_a: Mapping[int, int],
        neg_a: Mapping[int, int],
        pos_b: Mapping[int, int],
        neg_b: Mapping[int, int],
        *,
        index_map: Optional[Dict[int, int]] = None,
    ) -> Dict[str, float]:
        """
        Compute exact similarities for two signed multisets.

        :param pos_a: Positive counts for A.
        :type pos_a: Mapping[int,int]
        :param neg_a: Negative counts for A.
        :type neg_a: Mapping[int,int]
        :param pos_b: Positive counts for B.
        :type pos_b: Mapping[int,int]
        :param neg_b: Negative counts for B.
        :type neg_b: Mapping[int,int]
        :param index_map: Optional shared index map.
        :type index_map: Optional[Dict[int,int]]
        :returns: {"weighted_jaccard": float, "cosine": float}
        :rtype: Dict[str,float]
        """
        if index_map is None:
            index_map = self._union_index(pos_a, neg_a, pos_b, neg_b)
        va, _ = self.dicts_to_dense(
            pos_a, neg_a, index_map=index_map, ensure_signed=True
        )
        vb, _ = self.dicts_to_dense(
            pos_b, neg_b, index_map=index_map, ensure_signed=True
        )
        return {
            "weighted_jaccard": self.weighted_jaccard_signed(va, vb),
            "cosine": self.cosine_similarity(va, vb),
        }

    # ------------------------------ introspection -------------------------
    def last_index_map(self) -> Optional[Dict[int, int]]:
        """
        Return the last token→index map computed by :meth:`dicts_to_dense`.

        :returns: Shallow copy of index map or None.
        :rtype: Optional[Dict[int,int]]
        """
        return dict(self._last_index_map) if self._last_index_map is not None else None

    def describe(self) -> str:
        """
        Short usage snippet for subclasses.

        :returns: Example text.
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}(m={self._m}, seed={self._seed})\\\n"
            ".set_normalize(True).set_dtype(np.float32)\n"
            "sketch = _.build({10:2,20:1}, {30:1})\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(m={self._m}, seed={self._seed}, "
            f"normalize={self._normalize}, dtype={self._dtype})"
        )
