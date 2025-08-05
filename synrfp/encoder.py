from typing import List, Optional
import numpy as np
from joblib import Parallel, delayed

from synrfp.synrfp import rsmi_to_fingerprint


class SynRFPEncoder:
    """
    Batch-encode reaction SMILES into binary SynRFP fingerprints,
    with optional parallelism.

    :example (single-threaded):
        >>> from synrfp.encoder import SynRFPEncoder
        >>> fps = SynRFPEncoder.encode(["CCO>>C=C.O"], bits=16, seed=0)
        >>> fps.shape
        (1, 16)

    :example (multi-threaded):
        >>> from synrfp.encoder import SynRFPEncoder
        >>> encoder = SynRFPEncoder(n_jobs=4, backend='loky')
        >>> rxns = ["CCO>>C=C.O", "CC>>C.C"]
        >>> fps = encoder._encode_instance(rxns, bits=32, seed=1)
        >>> fps.shape
        (2, 32)
    """

    def __init__(
        self, n_jobs: int = 1, verbose: Optional[int] = 0, backend: str = "loky"
    ) -> None:
        """
        Initialize an encoder for reaction SMILES fingerprints.

        :param n_jobs: The maximum number of concurrently running jobs.
        :type  n_jobs: int
        :param verbose: The verbosity level.
        :type  verbose: Optional[int]
        :param backend: Parallelization backend to use (e.g. 'loky', 'threading').
        :type  backend: str
        """
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._backend = backend

    def __repr__(self) -> str:
        """
        Return a string representation of the encoder.

        :returns: Representation including parallel settings.
        :rtype: str
        """
        return f"<SynRFPEncoder(n_jobs={self._n_jobs}, backend='{self._backend}')>"

    @staticmethod
    def describe() -> None:
        """
        Print usage examples and available options.

        :returns: None
        """
        help_text = (
            "SynRFPEncoder(n_jobs=1, backend='loky')\n"
            "Methods:\n"
            "    encode(rxn_smiles, *, tokenizer, radius, sketch, bits, m,"
            " seed, require_pynauty) -> numpy.ndarray\n"
            "Parameters:\n"
            "    tokenizer: 'wl' or 'nauty'\n"
            "    sketch: 'parity', 'minhash', or 'cw'\n"
            "    radius, bits, m, seed: integer options\n"
            "Example:\n"
            "    fps = SynRFPEncoder.encode([rxn_smiles1, rxn_smiles2], tokenizer='wl',"
            " radius=2, sketch='parity', bits=1024, seed=0)"
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
        """
        if not rxn_smiles:
            return np.empty((0, 0), dtype=int)

        def _worker(smi: str) -> List[int]:
            return rsmi_to_fingerprint(
                smi,
                tokenizer=tokenizer,
                radius=radius,
                sketch=sketch,
                bits=bits,
                m=m,
                seed=seed,
                require_pynauty=require_pynauty,
            )

        try:
            fps_list = Parallel(
                n_jobs=self._n_jobs, verbose=self._verbose, backend=self._backend
            )(delayed(_worker)(smi) for smi in rxn_smiles)
        except ImportError as e:
            raise RuntimeError(f"Parallel execution dependencies are missing: {e}")

        lengths = {len(v) for v in fps_list}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent fingerprint lengths: {lengths}")

        return np.array(fps_list, dtype=int)

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
        """
        Encode a list of reaction SMILES into a 2D array of 0/1 fingerprints,
        using default single-job settings.

        :param rxn_smiles: List of reaction SMILES strings.
        :type  rxn_smiles: List[str]
        :param tokenizer: 'wl' or 'nauty'.
        :type  tokenizer: str
        :param radius: Neighborhood radius.
        :type  radius: int
        :param sketch: 'parity', 'minhash', or 'cw'.
        :type  sketch: str
        :param bits: Number of bits (for parity-fold).
        :type  bits: int
        :param m: Number of hash samples (for minhash or cw).
        :type  m: int
        :param seed: Random seed.
        :type  seed: int
        :param require_pynauty: Enforce `pynauty` for nauty tokenizer.
        :type  require_pynauty: bool
        :returns: 2D NumPy array of shape `(len(rxn_smiles), L)`.
        :rtype: numpy.ndarray
        :raises ValueError: On invalid tokenizer/sketch names or mismatched lengths.
        :raises RuntimeError: If dependencies are missing (`pynauty`, `datasketch`).
        """
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
