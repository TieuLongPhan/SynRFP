.. SynRFP documentation master file

========================================================================
SynRFP: Mapping-Free Reaction Fingerprints from Graph Sketches
========================================================================

**SynRFP** is a generic, **mapping-free**, permutation-invariant framework for representing chemical transformations as fixed-length fingerprints :cite:`phan2025synrfp`.

It formalizes reaction fingerprinting as three modular, mathematically explicit operators:

.. math::

   \mathbf{f}_{\Theta,\omega,\Xi}(G \to H) = \mathcal{S}_{\Xi}\Bigl(\operatorname{Comb}_{\omega}\bigl(\Psi_{\Theta}(G),\, \Psi_{\Theta}(H)\bigr)\Bigr)

- **$\Psi_\Theta$ (Subgraph Extractor)**: Decomposes molecular graphs into discrete structural motifs (1D Weisfeiler–Lehman, Morgan circular, simple paths, canonical ego-graphs, or topological atom pairs).
- **$\operatorname{Comb}_\omega$ (Algebraic Combiner)**: Compares reactant and product multisets to compute signed net change ($\Delta = C_H - C_G$) or total motif abundance ($U = C_H + C_G$).
- **$\mathcal{S}_\Xi$ (Randomized Sketcher)**: Compresses token summaries into fixed-length binary or real-valued vectors (ParityFold, MinHash, Consistent Weighted Sampling, Signed Random Projection).

.. figure:: figure/synrfp.png
   :alt: SynRFP Architecture
   :align: center
   :width: 90%

   Figure: The three-stage SynRFP pipeline: subgraph extraction ($\Psi$), algebraic combination ($\operatorname{Comb}_\omega$), and sketching ($\mathcal{S}$).

-----------------
Key Features
-----------------

- **Mapping-Free & Robust**: Requires no atom-to-atom mapping (AAM) or imaginary transition state heuristics.
- **Strict Invariance**: Invariant to atom relabeling, component ordering, and spectator placement.
- **Directional & Antisymmetric**: Signed net change $\Delta$ cleanly distinguishes formed from cleaved bonds ($\Delta(H \to G) = -\Delta(G \to H)$).
- **State-of-the-Art Accuracy**: Mean MCC of **0.953** across 12 benchmarks, outperforming fine-tuned RXNFP (0.926) and DRFP (0.918).
- **High Throughput**: Pure graph algorithms and bitwise hashing encode thousands of reactions per second on standard CPUs without GPU requirements.

-----------------
Documentation
-----------------

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   Getting Started <getting_started>
   Tutorials and Examples <tutorials_and_examples>

.. toctree::
   :caption: Reference
   :maxdepth: 2

   API Reference <api>
   References <reference>

------------------
Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
