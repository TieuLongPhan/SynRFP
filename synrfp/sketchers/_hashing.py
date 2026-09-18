# synrfp/sketchers/_hashing.py
"""
Counter-based, vectorised hashing used by the weighted sketchers.

Weighted sketches (CWS, SRP) must draw the *same* random variables for the
same token in every reaction, otherwise signatures of different reactions are
not comparable. Seeding a generator per reaction and indexing it by a local
dense position breaks this. Here every random quantity is a pure function of
``(seed, sample index, token key, stream)``, computed with SplitMix64.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

_MASK = (1 << 64) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_TWO_M53 = 2.0**-53


def splitmix64(x: np.ndarray) -> np.ndarray:
    """
    SplitMix64 finaliser applied elementwise to a ``uint64`` array.

    :param x: Input array (converted to ``uint64``).
    :type x: numpy.ndarray
    :return: Mixed ``uint64`` array of the same shape.
    :rtype: numpy.ndarray
    """
    z = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
    return z ^ (z >> np.uint64(31))


def mix(*keys: np.ndarray) -> np.ndarray:
    """
    Combine several ``uint64`` arrays (broadcast together) into one hash.

    :param keys: Arrays or scalars interpreted as ``uint64``.
    :return: Broadcast ``uint64`` hash array.
    :rtype: numpy.ndarray
    """
    h = np.uint64(0x243F6A8885A308D3)
    for k in keys:
        h = splitmix64(np.asarray(k, dtype=np.uint64) ^ h)
    return h


def to_uint64_keys(tokens: Iterable[int]) -> np.ndarray:
    """
    Reduce arbitrary Python integers (possibly > 64 bit) to ``uint64`` keys.

    :param tokens: Integer token identifiers.
    :return: ``uint64`` array.
    :rtype: numpy.ndarray
    """
    return np.fromiter(
        ((int(t) ^ (int(t) >> 64)) & _MASK for t in tokens), dtype=np.uint64
    )


def uniform_open01(h: np.ndarray) -> np.ndarray:
    """
    Map ``uint64`` hashes to floats strictly inside ``(0, 1)``.

    :param h: ``uint64`` hash array.
    :return: Float array in the open unit interval.
    :rtype: numpy.ndarray
    """
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * _TWO_M53


def gaussian(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Standard normal variates from two independent hash streams (Box-Muller).

    :param h1: First ``uint64`` hash array.
    :param h2: Second ``uint64`` hash array.
    :return: Float array of N(0, 1) variates.
    :rtype: numpy.ndarray
    """
    u1 = uniform_open01(h1)
    u2 = uniform_open01(h2)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
