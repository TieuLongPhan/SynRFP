.. _api-reference:

API Reference
=============

This page provides an overview of the public Python API exposed by
:mod:`synrfp`. It is structured according to the core framework architecture:

- **Top-level API** in :mod:`synrfp` (convenience functions and common classes)
- **Encoding & Pipeline Engine** in :mod:`synrfp.encode` (:class:`~synrfp.encode.engine.SynRFP`, :class:`~synrfp.encode.batch.BatchEncoder`, :func:`~synrfp.encode.core.fingerprint_from_tokens`)
- **Graph & Reaction Containers** in :mod:`synrfp.graph` (:class:`~synrfp.graph.molecule.Molecule`, :class:`~synrfp.graph.reaction.Reaction`)
- **Subgraph Tokenizers** in :mod:`synrfp.tokenizers` (WL, Morgan, Path, Nauty, MAP)
- **Sketchers** in :mod:`synrfp.sketchers` (ParityFold, MinHash, CWS, SRP)
- **Utilities & Metrics** in :mod:`synrfp.utils` (Tanimoto, MinHash Jaccard, bit conversions)

----------------------------
Top-level API: :mod:`synrfp`
----------------------------

The top-level :mod:`synrfp` module collects primary user-facing entry points:

.. automodule:: synrfp
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   synrfp
   SynRFP
   SynRFPResult
   BatchEncoder
   tanimoto_bits
   jaccard_minhash
   build_graph_from_printout

---------------------------------------
Encoding & Engine: :mod:`synrfp.encode`
---------------------------------------

The :mod:`synrfp.encode` subpackage coordinates the 3-stage pipeline:
token extraction, algebraic combination, and sketching.

Core Wrapper & Token Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.encode.core
   :members:
   :undoc-members:
   :show-inheritance:

Batch Processing
~~~~~~~~~~~~~~~~

.. automodule:: synrfp.encode.batch
   :members:
   :undoc-members:
   :show-inheritance:

Engine Composition
~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.encode.engine
   :members:
   :undoc-members:
   :show-inheritance:

Result Container
~~~~~~~~~~~~~~~~

.. automodule:: synrfp.encode.result
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

-------------------------------------
Graph Containers: :mod:`synrfp.graph`
-------------------------------------

The :mod:`synrfp.graph` subpackage defines lightweight graph representations
for molecules and chemical transformations.

Molecules & Graph Data
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.graph.molecule
   :members:
   :undoc-members:
   :show-inheritance:

Chemical Reactions
~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.graph.reaction
   :members:
   :undoc-members:
   :show-inheritance:

------------------------------------
Tokenizers: :mod:`synrfp.tokenizers`
------------------------------------

Tokenizers decompose molecular graphs into multisets of integer tokens representing
local chemical environments.

Base Tokenizer Interface
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.base
   :members:
   :undoc-members:
   :show-inheritance:

Weisfeiler–Lehman (WL) Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.wl
   :members:
   :undoc-members:
   :show-inheritance:

Morgan Circular Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.morgan
   :members:
   :undoc-members:
   :show-inheritance:

Path Tokenizer
~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.path
   :members:
   :undoc-members:
   :show-inheritance:

Nauty Canonical Subgraph Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.nauty
   :members:
   :undoc-members:
   :show-inheritance:

Tokenizer Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.tokenizers.utils
   :members:
   :undoc-members:
   :show-inheritance:

----------------------------------
Sketchers: :mod:`synrfp.sketchers`
----------------------------------

Sketchers compress (signed) token summaries into compact, fixed-dimensional vectors.

Base Sketcher Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.sketchers.base
   :members:
   :undoc-members:
   :show-inheritance:

Parity Folding (XOR Hashing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.sketchers.parity_fold
   :members:
   :undoc-members:
   :show-inheritance:

MinHash Sketching
~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.sketchers.minhash_sketch
   :members:
   :undoc-members:
   :show-inheritance:

Consistent Weighted Sampling (CWS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.sketchers.cw_sketch
   :members:
   :undoc-members:
   :show-inheritance:

Signed Random Projection (SRP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synrfp.sketchers.srp_sketch
   :members:
   :undoc-members:
   :show-inheritance:

------------------------------
Utilities: :mod:`synrfp.utils`
------------------------------

Helper functions for signature folding, binary bit conversions, and distance/similarity metrics.

.. automodule:: synrfp.utils
   :members:
   :undoc-members:
   :show-inheritance:
