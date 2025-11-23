from typing import Dict, List, Union
import numpy as np

from synrfp.graph.molecule import Molecule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def signature_to_bits(sig, n_bits: int) -> List[int]:
    """
    Convert an integer signature (e.g. MinHash or CWSketch) into a binary
    bit-vector of length ``n_bits``.

    Each integer h in the signature is mapped to an index ``h % n_bits`` and
    that position is set to 1 (collisions are allowed).

    :param sig: 1D signature (list/array of ints).
    :type sig: Sequence[int] or numpy.ndarray
    :param n_bits: Length of the output fingerprint.
    :type n_bits: int
    :returns: List of 0/1 bits of length ``n_bits``.
    :rtype: List[int]
    """
    arr = np.asarray(sig, dtype=np.int64).ravel()
    bits = np.zeros(n_bits, dtype=np.uint8)
    if arr.size == 0:
        return bits.tolist()
    idx = arr % n_bits
    bits[idx] = 1
    return bits.tolist()


def build_graph_from_printout(
    nodes: Dict[int, Dict],
    edges: Dict[tuple[int, int], Dict],
) -> Molecule:
    """
    Helper to convert “printout” dicts directly into a :class:`Molecule`.

    :param nodes: Mapping from node ID to attribute dict.
    :type nodes: Dict[int, Dict]
    :param edges: Mapping from (u, v) edges (with u < v) to attribute dict.
    :type edges: Dict[tuple[int, int], Dict]
    :returns: A fresh :class:`Molecule` instance.
    :rtype: Molecule

    :example:
        >>> nodes = {0: {'element': 'C'}, 1: {'element': 'O'}}
        >>> edges = {(0, 1): {'order': 1.5}}
        >>> G = build_graph_from_printout(nodes, edges)
    """
    return Molecule.from_dicts(nodes, edges)


def tanimoto_bits(
    b1: Union[bytearray, List[int], np.ndarray],
    b2: Union[bytearray, List[int], np.ndarray],
) -> float:
    """
    Compute the Tanimoto (Jaccard) similarity between two binary‐bit sketches.

    Accepts ``bytearray``, ``list[int]``, or ``numpy.ndarray`` of 0/1.

    :param b1: First bit array.
    :type b1: bytearray or List[int] or numpy.ndarray
    :param b2: Second bit array.
    :type b2: bytearray or List[int] or numpy.ndarray
    :returns: Intersection size divided by union size, or 0.0 if union is zero.
    :rtype: float
    """
    a1 = np.asarray(b1, dtype=int)
    a2 = np.asarray(b2, dtype=int)
    if a1.shape != a2.shape:
        raise ValueError("bit sketches must have identical shape")
    inter = int(np.sum((a1 & 1) & (a2 & 1)))
    union = int(np.sum(((a1 & 1) | (a2 & 1))))
    return 0.0 if union == 0 else inter / union


def jaccard_minhash(h1: Union[list, tuple], h2: Union[list, tuple]) -> float:
    """
    Estimate Jaccard similarity from two MinHash signature arrays.

    :param h1: First MinHash hash‐value sequence.
    :type h1: list or tuple
    :param h2: Second MinHash sequence (must be same length).
    :type h2: list or tuple
    :returns: Fraction of positions where ``h1[i] == h2[i]``.
    :rtype: float
    """
    a1 = np.asarray(h1)
    a2 = np.asarray(h2)
    if a1.shape != a2.shape:
        raise ValueError("MinHash signatures must be the same length")
    return float((a1 == a2).mean())


def sketch_to_binary(sketch: Union[bytearray, List[int], np.ndarray]) -> List[int]:
    """
    Convert a binary sketch (e.g. a ParityFold result) into a list of 0/1 bits.

    Supports ``bytearray``, ``list[int]``/``list[bool]``, or numpy arrays with
    integer/boolean dtype.

    :param sketch: Sketch to convert.
    :type sketch: bytearray or List[int] or numpy.ndarray
    :returns: Binary bit-vector.
    :rtype: List[int]
    :raises TypeError: If ``sketch`` is not a supported type or contains
                       non-binary values.
    """
    # numpy array path
    if isinstance(sketch, np.ndarray):
        if sketch.ndim != 1:
            raise TypeError("numpy sketch must be a 1D array of bits")
        arr = sketch.astype(int).tolist()
        if not all((x == 0 or x == 1) for x in arr):
            raise TypeError("numpy sketch contains values other than 0/1")
        return arr

    # bytearray path
    if isinstance(sketch, bytearray):
        return list(sketch)

    # list-of-int/bool path
    if isinstance(sketch, list) and all(isinstance(x, (int, bool)) for x in sketch):
        if not all((int(x) == 0 or int(x) == 1) for x in sketch):
            raise TypeError("list sketch contains values other than 0/1")
        return [0 if int(x) == 0 else 1 for x in sketch]

    raise TypeError(
        "sketch_to_binary expects a bytearray, list[int]/list[bool], or 1D numpy array"
    )


def sketch_to_array(sketch: object) -> np.ndarray:
    """
    Generic helper: coerce any sketch-like object into a 1D numpy int array.

    Used for non-binary sketchers (MinHash, CWSketch) when exposing signatures.

    Behaviour
    ---------
    - If the incoming array is 1D, it is returned as-is (converted to int).
    - If the incoming array is 2D (e.g. shape (m, 2) from some CW
      implementations), it is flattened to 1D with ``arr.ravel()``.
    - Higher-dimensional arrays are rejected.

    :param sketch: Sketch object (list/array/bytearray).
    :type sketch: object
    :returns: 1D numpy array of ints.
    :rtype: numpy.ndarray
    :raises TypeError: If ``sketch`` cannot be coerced to 1D.
    """
    arr = np.asarray(sketch)

    if arr.ndim == 1:
        return arr.astype(int, copy=False)

    if arr.ndim == 2:
        # e.g. CW sketch returning (m, 2): flatten to 1D before hashing
        return arr.reshape(-1).astype(int, copy=False)

    raise TypeError(f"Expected 1D or 2D sketch array, got shape {arr.shape!r}")
