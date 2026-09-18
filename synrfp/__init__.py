"""SynRFP: mapping-free reaction fingerprints from graph sketches."""

from importlib.metadata import PackageNotFoundError, version as _version

from .synrfp import (
    SynRFP,
    SynRFPResult,
    synrfp,
)
from .utils import build_graph_from_printout, tanimoto_bits, jaccard_minhash
from .encode import BatchEncoder

try:
    __version__ = _version("synrfp")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "build_graph_from_printout",
    "tanimoto_bits",
    "jaccard_minhash",
    "SynRFP",
    "SynRFPResult",
    "BatchEncoder",
    "synrfp",
]
