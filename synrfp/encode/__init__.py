# synrfp/encode/__init__.py
"""
Encoding utilities for mapping reaction SMILES to SynRFP fingerprints.

Public API
----------
- SynRFPResult  : result container (tokens, support, sketch)
- SynRFP        : engine for Molecule -> SynRFPResult
- synrfp        : top-level RSMI -> bit-vector API
- SynRFPEncoder : batch encoder with optional parallelism
"""

from .result import SynRFPResult
from .engine import SynRFP
from .core import synrfp
from .batch import BatchEncoder

__all__ = ["SynRFPResult", "SynRFP", "synrfp", "BatchEncoder"]
