# encoder.py
from typing import List, Optional
import numpy as np

try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except Exception:
    Parallel = None
    delayed = None
    _HAVE_JOBLIB = False

from synrfp.synrfp import synrfp


class SynRFPEncoder:
    """
    Batch-encode reaction SMILES into binary SynRFP fingerprints,
    with optional parallelism.

    :example (single-threaded):
        >>> from synrfp.encoder import SynRFPEncoder
        >>> fps = SynRFPEncoder.encode(["CCO>>C=C.O"], bits=16, seed=0)
        >>> fps.shape
        (1, 16)
    """

    def __init__(
        self, n_jobs: int = 1, verbose: Optional[int] = None, backend: str = "loky"
    ) -> None:
        """
        Initialize an encoder.

        :param n_jobs: The maximum number of concurrently running jobs.
        :type  n_jobs: int
        :param verbose: The verbosity level.
        :type  verbose: Optional[int]
        :param backend: Parallelization backend to use (e.g. 'loky', 'threading').
        :type  backend: str
        """
        self._n_jobs = int(n_jobs)
        self._verbose = verbose
        self._backend = backend

    def __repr__(self) -> str:
        return f"<SynRFPEncoder(n_jobs={self._n_jobs}, backend='{self._backend}')>"

    @staticmethod
    def describe() -> None:
        help_text = (
            "SynRFPEncoder(n_jobs=1, backend='loky')\n"
            "Methods:\n"
            "    encode(rxn_smiles, *, tokenizer, radius, sketch, bits, m,"
            " seed, require_pynauty) -> numpy.ndarray\n"
        )
        print(help_text)

    def _encode_instance(
        self,
        rxn_smiles: List[str],
        tokenizer: str = "wl",
        radius: int = 2,
        sketch: str = "parity",
        bits: int = 1024,
        m: int = 256,
        seed: int = 1,
        require_pynauty: bool = False,
    ) -> np.ndarray:
        """
        Instance method: encode SMILES with parallel options.
        Uses direct calls when n_jobs == 1 to preserve exception behaviour.
        """
        if not rxn_smiles:
            return np.empty((0, 0), dtype=int)

        def _worker(smi: str):
            # forward arguments to library synrfp (may raise)
            return synrfp(
                smi,
                tokenizer=tokenizer,
                radius=radius,
                sketch=sketch,
                bits=bits,
                m=m,
                seed=seed,
                require_pynauty=require_pynauty,
            )

        # If single-threaded requested, avoid joblib and call directly so
        # exceptions propagate cleanly (and monkeypatches on this module work).
        if self._n_jobs == 1:
            fps_list = [_worker(smi) for smi in rxn_smiles]
        else:
            if not _HAVE_JOBLIB:
                raise RuntimeError("joblib is required for multi-job encoding")
            try:
                fps_list = Parallel(
                    n_jobs=self._n_jobs, verbose=self._verbose, backend=self._backend
                )(delayed(_worker)(smi) for smi in rxn_smiles)
            except Exception as e:
                # Rewrap with a clearer message
                raise RuntimeError(f"Parallel execution failed: {e}") from e

        # ensure we received a sequence of equal-length bit vectors
        try:
            lengths = {len(v) for v in fps_list}
        except TypeError:
            raise ValueError(
                "synrfp returned non-sequence results; check implementation"
            )
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent fingerprint lengths: {lengths}")

        return np.asarray(fps_list, dtype=int)

    @classmethod
    def encode(
        cls,
        rxn_smiles: List[str],
        *,
        tokenizer: str = "wl",
        radius: int = 2,
        sketch: str = "parity",
        bits: int = 1024,
        m: int = 256,
        seed: int = 1,
        require_pynauty: bool = False,
    ) -> np.ndarray:
        encoder = cls()
        return encoder._encode_instance(
            rxn_smiles,
            tokenizer=tokenizer,
            radius=radius,
            sketch=sketch,
            bits=bits,
            m=m,
            seed=seed,
            require_pynauty=require_pynauty,
        )
