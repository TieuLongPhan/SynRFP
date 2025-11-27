.. _api-reference:

API Reference
=============

This page provides an overview of the public Python API exposed by
:mod:`synrfp`. It focuses on:

- core fingerprint builders in :mod:`synrfp`,
- graph containers and reaction helpers in :mod:`synrfp.graph`,
- tokenization backends in :mod:`synrfp.tokenizers`,
- sketching backends in :mod:`synrfp.sketchers`, and
- batch encoding utilities in :mod:`synrfp.encoder`.

If you are looking for high-level usage examples, see the README
on GitHub or the :doc:`Getting Started <getting_started>` page.

-------------------------
Top-level API: :mod:`synrfp`
-------------------------

The :mod:`synrfp` module collects the main user-facing entry points:
fingerprint engines, convenience wrappers, and similarity utilities.

.. automodule:: synrfp
   :members:
   :undoc-members:
   :show-inheritance:

Key classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrfp.SynRFP
   synrfp.SynRFPResult
   synrfp.synrfp
   synrfp.build_graph_from_printout
   synrfp.tanimoto_bits
   synrfp.jaccard_minhash
   synrfp.SynRFPEncoder

Typical responsibilities of the top-level API include:

- converting reaction SMILES (RSMI) directly into fixed-length fingerprints,
- configuring a :class:`~synrfp.SynRFP` engine with a tokenizer and sketcher,
- exposing similarity utilities for binary and MinHash sketches, and
- providing a simple batch encoder for lists of reaction SMILES.

------------------------------
Graph utilities: :mod:`synrfp.graph`
------------------------------

The :mod:`synrfp.graph` subpackage defines light-weight graph containers
and helpers for representing reactions.

.. automodule:: synrfp.graph.graph_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.graph.reaction
   :members:
   :undoc-members:
   :show-inheritance:

Key classes
~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrfp.graph.graph_data.GraphData
   synrfp.graph.reaction.Reaction

Typical responsibilities include:

- representing labeled molecular graphs via :class:`~synrfp.graph.graph_data.GraphData`,
- constructing :class:`~synrfp.graph.reaction.Reaction` objects from RSMI or
  from existing :mod:`networkx` graphs, and
- providing small convenience methods for introspection and tabular summaries.

--------------------------------
Tokenizers: :mod:`synrfp.tokenizers`
--------------------------------

Tokenizers map molecular graphs to multisets of integer tokens
(e.g. WL subtree hashes). They are responsible for the graph → tokens stage.

.. automodule:: synrfp.tokenizers.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.tokenizers.wl
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.tokenizers.nauty
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.tokenizers.utils
   :members:
   :undoc-members:
   :show-inheritance:

Key classes and helpers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrfp.tokenizers.base.BaseTokenizer
   synrfp.tokenizers.wl.WLTokenizer
   synrfp.tokenizers.nauty.NautyTokenizer
   synrfp.tokenizers.utils._h64
   synrfp.tokenizers.utils.atom_label_tuple
   synrfp.tokenizers.utils.bond_label_tuple

Typical responsibilities include:

- defining the abstract :class:`~synrfp.tokenizers.base.BaseTokenizer` interface,
- providing concrete implementations such as
  :class:`~synrfp.tokenizers.wl.WLTokenizer` and
  :class:`~synrfp.tokenizers.nauty.NautyTokenizer`,
- specifying which node/edge attributes participate in labels, and
- offering hashing helpers used to turn label tuples into stable 64-bit tokens.

--------------------------------
Sketchers: :mod:`synrfp.sketchers`
--------------------------------

Sketchers compress (signed) token multisets into fixed-size
fingerprints. They implement the Δ/U → sketch stage.

.. automodule:: synrfp.sketchers.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.sketchers.parity_fold
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.sketchers.minhash_sketch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synrfp.sketchers.cw_sketch
   :members:
   :undoc-members:
   :show-inheritance:

Key classes
~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrfp.sketchers.base.BaseSketch
   synrfp.sketchers.base.WeightedSketch
   synrfp.sketchers.parity_fold.ParityFold
   synrfp.sketchers.minhash_sketch.MinHashSketch
   synrfp.sketchers.cw_sketch.CWSketch

Typical responsibilities include:

- defining unweighted and weighted sketcher interfaces
  (:class:`~synrfp.sketchers.base.BaseSketch`,
  :class:`~synrfp.sketchers.base.WeightedSketch`),
- implementing binary parity-fold sketches
  (:class:`~synrfp.sketchers.parity_fold.ParityFold`),
- implementing MinHash-based sketches for Jaccard estimation
  (:class:`~synrfp.sketchers.minhash_sketch.MinHashSketch`), and
- implementing consistent weighted sampling for signed deltas
  (:class:`~synrfp.sketchers.cw_sketch.CWSketch`).

------------------------------------
Batch encoding: :mod:`synrfp.encoder`
------------------------------------

The :mod:`synrfp.encoder` module provides a convenience wrapper for batch
encoding lists of reaction SMILES into SynRFP fingerprints, with optional
parallelization via :mod:`joblib`.

.. automodule:: synrfp.encoder
   :members:
   :undoc-members:
   :show-inheritance:

Key class
~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrfp.encoder.SynRFPEncoder

Typical responsibilities of :class:`~synrfp.encoder.SynRFPEncoder` include:

- turning a list of RSMI strings into a 2D NumPy array of fingerprints,
- exposing the same configuration knobs as :func:`synrfp.synrfp`
  (tokenizer, radius, sketch type, bit length, seed), and
- handling multi-process or multi-threaded execution transparently when
  ``n_jobs > 1``.
