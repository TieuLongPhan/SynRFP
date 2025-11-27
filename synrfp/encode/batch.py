# synrfp/encode/batch.py
from __future__ import annotations

from typing import List, Sequence, Optional
import numpy as np

try:
    from joblib import Parallel, delayed  # optional dependency

    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover - optional dependency
    Parallel = None
    delayed = None
    _HAVE_JOBLIB = False

from synrfp.encode.core import synrfp as _synrfp_call


class BatchEncoder:
    """
    Batch encoder for mapping reaction SMILES (RSMI) to SynRFP fingerprints.

    The encoder wraps the top-level ``synrfp`` function and provides:
    - configuration stored on the instance (tokenizer/sketch/seed/etc.),
    - single-item and batched encoding,
    - optional parallelization via ``joblib.Parallel``,
    - optional chunked processing via ``batch_size`` for memory control.

    :param tokenizer: Tokenizer name ('wl', 'nauty', 'morgan', 'path', ...).
    :type tokenizer: str
    :param radius: Neighborhood / iteration radius for tokenizers that use it.
    :type radius: int
    :param sketch: Sketcher name ('parity', 'minhash', 'cw', 'srp', ...).
    :type sketch: str
    :param bits: Length of the final bit-vector for bit-based sketches.
    :type bits: int
    :param m: Sketch size parameter (e.g. number of hash samples/projections).
    :type m: int
    :param seed: RNG seed forwarded to tokenizers/sketchers.
    :type seed: int
    :param mode: Token aggregation mode: 'delta' or 'union'.
    :type mode: str
    :param node_attrs: Optional node attribute names passed to tokenizers.
    :type node_attrs: Sequence[str] | None
    :param edge_attrs: Optional edge attribute names passed to tokenizers.
    :type edge_attrs: Sequence[str] | None
    :param n_jobs: Number of jobs for parallel encoding (1 = serial).
    :type n_jobs: int
    :param batch_size: Maximum number of reactions per batch. If None, no chunking.
    :type batch_size: int | None
    :param verbose: Verbosity forwarded to joblib (int). If None, 0 is used.
    :type verbose: int | None
    :param backend: joblib backend (e.g. 'loky', 'threading').
    :type backend: str

    :example:

    >>> from synrfp.encode.batch import BatchEncoder
    >>> enc = BatchEncoder(tokenizer='wl', sketch='parity', bits=1024, n_jobs=1)
    >>> single_fp = enc.encode_one('CCO>>CCO')           # numpy array shape (1024,)
    >>> many = enc.encode_many(['CCO>>CCO', 'CCO>>CCO'])  # shape (2, 1024)

    A small-batch example (useful for constrained memory / large inputs):

    >>> enc = BatchEncoder(batch_size=128, n_jobs=2)
    >>> X = enc.encode_many(list_of_rsmi)  # processes list_of_rsmi in chunks of 128
    """

    def __init__(
        self,
        tokenizer: str = "wl",
        radius: int = 2,
        sketch: str = "parity",
        bits: int = 1024,
        m: int = 256,
        seed: int = 1,
        mode: str = "delta",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        *,
        n_jobs: int = 1,
        batch_size: Optional[int] = None,
        verbose: Optional[int] = None,
        backend: str = "loky",
    ) -> None:
        # validation
        if not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer; use 1 for serial.")
        if bits <= 0:
            raise ValueError("bits must be a positive integer.")
        if m <= 0:
            raise ValueError("m must be a positive integer.")
        if mode not in ("delta", "union"):
            raise ValueError("mode must be either 'delta' or 'union'.")
        if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None.")

        # store config
        self.tokenizer = str(tokenizer)
        self.radius = int(radius)
        self.sketch = str(sketch)
        self.bits = int(bits)
        self.m = int(m)
        self.seed = int(seed)
        self.mode = mode
        self.node_attrs = list(node_attrs) if node_attrs is not None else None
        self.edge_attrs = list(edge_attrs) if edge_attrs is not None else None

        self.n_jobs = int(n_jobs)
        self.batch_size = batch_size
        self.verbose = verbose
        self.backend = backend

    def __repr__(self) -> str:
        return (
            "BatchEncoder("
            f"tokenizer={self.tokenizer!r}, radius={self.radius}, "
            f"sketch={self.sketch!r}, bits={self.bits}, m={self.m}, "
            f"seed={self.seed}, mode={self.mode!r}, "
            f"n_jobs={self.n_jobs}, batch_size={self.batch_size}, "
            f"backend={self.backend!r})"
        )

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _encode_one(self, rsmi: str) -> List[int]:
        """
        Encode a single reaction SMILES string by delegating to the
        top-level :func:`synrfp.encode.api.synrfp` function.

        :param rsmi: Reaction SMILES string (RSMI).
        :type rsmi: str
        :returns: Fingerprint as a Python list of ints (0/1 for bit-based sketches).
        :rtype: list[int]
        """
        return _synrfp_call(
            rsmi,
            tokenizer=self.tokenizer,
            radius=self.radius,
            sketch=self.sketch,
            bits=self.bits,
            m=self.m,
            seed=self.seed,
            mode=self.mode,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
        )

    def _encode_chunk(self, rsmi_list: Sequence[str]) -> List[List[int]]:
        """
        Encode a chunk (sequence) of RSMIs. This supports parallel execution.

        :param rsmi_list: Sequence of reaction SMILES strings.
        :type rsmi_list: Sequence[str]
        :returns: List of fingerprint lists in input order.
        :rtype: list[list[int]]
        :raises RuntimeError: If ``n_jobs != 1`` but joblib is not available.
        """
        if self.n_jobs == 1:
            return [self._encode_one(s) for s in rsmi_list]

        if not _HAVE_JOBLIB:
            raise RuntimeError(
                "joblib is required for parallel encoding (n_jobs != 1). "
                "Install joblib or set n_jobs=1."
            )

        # ensure we pass an int to joblib.verbose (some joblib versions crash on None)
        joblib_verbose = self.verbose if self.verbose is not None else 0

        return Parallel(
            n_jobs=self.n_jobs,
            verbose=joblib_verbose,
            backend=self.backend,
        )(delayed(self._encode_one)(s) for s in rsmi_list)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def encode_one(self, rsmi: str) -> np.ndarray:
        """
        Encode a single reaction SMILES into a 1D numpy array.

        :param rsmi: Reaction SMILES string.
        :type rsmi: str
        :returns: 1D numpy array of ints (length == bits).
        :rtype: numpy.ndarray
        """
        fp = self._encode_one(rsmi)
        return np.asarray(fp, dtype=int)

    def encode_many(self, rsmi_list: Sequence[str]) -> np.ndarray:
        """
        Encode a sequence of reaction SMILES into a 2D numpy array.

        If ``batch_size`` is set and smaller than the input length, encoding
        proceeds in chunks of at most ``batch_size`` and results are concatenated.

        :param rsmi_list: Sequence of reaction SMILES strings.
        :type rsmi_list: Sequence[str]
        :returns: 2D numpy array with shape (N, L) where N = len(rsmi_list)
        and L = fingerprint length.
        :rtype: numpy.ndarray
        :raises ValueError: If fingerprint lengths are inconsistent across batches.
        """
        n = len(rsmi_list)
        if n == 0:
            return np.empty((0, 0), dtype=int)

        if self.batch_size is None or self.batch_size >= n:
            results = self._encode_chunk(rsmi_list)
        else:
            results: List[List[int]] = []
            for start in range(0, n, self.batch_size):
                chunk = rsmi_list[start : start + self.batch_size]  # noqa
                chunk_res = self._encode_chunk(chunk)
                results.extend(chunk_res)

        lengths = {len(fp) for fp in results}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent fingerprint lengths encountered: {lengths}")

        return np.asarray(results, dtype=int)

    @classmethod
    def encode(
        cls,
        rsmi_list: Sequence[str],
        *,
        tokenizer: str = "wl",
        radius: int = 2,
        sketch: str = "parity",
        bits: int = 1024,
        m: int = 256,
        seed: int = 1,
        mode: str = "delta",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convenience classmethod to encode many reactions (serial execution).

        :param rsmi_list: Sequence of reaction SMILES.
        :type rsmi_list: Sequence[str]
        :param tokenizer: Tokenizer name.
        :type tokenizer: str
        :param radius: Neighborhood radius.
        :type radius: int
        :param sketch: Sketcher name.
        :type sketch: str
        :param bits: Fingerprint length for bit-based sketches.
        :type bits: int
        :param m: Sketch size parameter.
        :type m: int
        :param seed: RNG seed.
        :type seed: int
        :param mode: Token aggregation mode ('delta'|'union').
        :type mode: str
        :param node_attrs: Node attribute names for tokenizer.
        :type node_attrs: Sequence[str] | None
        :param edge_attrs: Edge attribute names for tokenizer.
        :type edge_attrs: Sequence[str] | None
        :param batch_size: Maximum chunk size for encode_many.
        :type batch_size: int | None
        :returns: 2D numpy array of fingerprints.
        :rtype: numpy.ndarray
        :example:
        >>> BatchEncoder.encode(['CCO>>CCO'], tokenizer='wl', sketch='parity', bits=64)
        array([[...]], dtype=int)
        """
        enc = cls(
            tokenizer=tokenizer,
            radius=radius,
            sketch=sketch,
            bits=bits,
            m=m,
            seed=seed,
            mode=mode,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            n_jobs=1,
            batch_size=batch_size,
        )
        return enc.encode_many(rsmi_list)
