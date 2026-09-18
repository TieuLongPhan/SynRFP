# synrfp/encode/__init__.py
"""
Encoding utilities for mapping reaction SMILES to SynRFP fingerprints.

Public API
----------
- SynRFPResult  : result container (tokens, support, sketch)
- SynRFP        : engine for Molecule -> SynRFPResult
- synrfp        : top-level RSMI -> bit-vector API
- fingerprint_from_tokens : stages 2-3 from precomputed token counts
- SynRFPEncoder : batch encoder with optional parallelism
"""

from .result import SynRFPResult
from .engine import SynRFP
from .core import synrfp, fingerprint_from_tokens
from .batch import BatchEncoder

__all__ = [
    "SynRFPResult",
    "SynRFP",
    "synrfp",
    "fingerprint_from_tokens",
    "BatchEncoder",
]
