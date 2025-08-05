# synrfp/sketchers/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any
from collections import Counter


class BaseSketch(ABC):
    """
    Abstract base class for set / multiset sketchers.

    Subclasses must implement :meth:`build` and optionally override :meth:`describe`.
    """

    def __init__(self, seed: int = 1):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed

    # ---------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------
    @abstractmethod
    def build(self, support: Iterable[int]):
        """
        Build a sketch object from an *unweighted* iterable of integers.

        :param support: Iterable of hashable integers representing the feature set.
        :type support: Iterable[int]
        :returns: Sketch object (type depends on subclass).
        """
        raise NotImplementedError

    def describe(self) -> str:  # noqa: D401 – imperative description is fine
        """Return a short usage example."""
        return (
            f"sketcher = {self.__class__.__name__}(...);"
            + " sketch = sketcher.build(tokens)"
        )

    # ---------------------------------------------------------------------
    # Helper utilities shared by subclasses
    # ---------------------------------------------------------------------
    def _as_counter(self, support: Iterable[int]) -> Counter:
        """Convert *support* to a :class:`collections.Counter`."""
        return Counter(support)


class WeightedSketch(ABC):
    """
    Abstract interface for weighted (signed) sketchers: build(pos, neg) -> sketch object.

    Weighted sketches take separate positive and negative multisets and produce
    a fixed-size sketch.
    """

    @abstractmethod
    def build(self, pos: Dict[int, int], neg: Dict[int, int]) -> Any:
        """
        Build a weighted sketch from positive and negative token counts.

        :param pos: Mapping from token → positive count.
        :type pos: Dict[int, int]
        :param neg: Mapping from token → negative count.
        :type neg: Dict[int, int]
        :returns: Sketch object (e.g. array of hash values).
        :rtype: Any

        :raises ValueError: If both `pos` and `neg` are empty.
        """
        raise NotImplementedError

    @staticmethod
    def describe() -> str:
        """
        Return usage example for a weighted sketcher.

        :returns: Example code snippet.
        :rtype: str
        """
        return (
            ">>> from synrfp.sketchers.cw_sketch import CWSketch\n"
            ">>> sketcher = CWSketch(m=256, seed=1)\n"
            ">>> pos = {h1: 2, h2: 1}\n"
            ">>> neg = {h3: 1}\n"
            ">>> sketch = sketcher.build(pos, neg)\n"
        )
