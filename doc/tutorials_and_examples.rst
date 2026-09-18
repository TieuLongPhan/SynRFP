.. _tutorials-and-examples:

======================
Tutorials and Examples
======================

This page provides practical, copy-and-paste Python examples demonstrating the full capabilities of **SynRFP**:

1. :ref:`single-reaction-and-inspection`
2. :ref:`reaction-directionality-and-antisymmetry`
3. :ref:`comparing-sketchers-and-similarities`
4. :ref:`fast-parameter-sweeps-from-tokens`
5. :ref:`high-throughput-batch-encoding`
6. :ref:`persisting-fingerprints`
7. :ref:`supervised-classification-with-mlp`
8. :ref:`unsupervised-reaction-clustering`

.. _single-reaction-and-inspection:

1. Single Reaction Encoding & Token Inspection
==============================================

Decompose a chemical transformation into reactant and product token multisets,
compute the signed net change $\Delta = C_H - C_G$, and generate a binary fingerprint:

.. code-block:: python

   from synrfp import SynRFP
   from synrfp.graph.reaction import Reaction
   from synrfp.tokenizers.wl import WLTokenizer
   from synrfp.sketchers.parity_fold import ParityFold

   # 1) Parse reaction SMILES
   rxn = Reaction.from_rsmi("CCO>>C=C.O")

   # 2) Set up tokenizer (radius 2 Weisfeiler-Lehman) and sketcher (1024-bit ParityFold)
   tokenizer = WLTokenizer()
   sketcher = ParityFold(bits=1024, seed=42)
   engine = SynRFP(tokenizer=tokenizer, radius=2, sketch=sketcher)

   # 3) Compute fingerprint result
   result = engine.fingerprint(rxn.reactant, rxn.product, mode="delta")

   # 4) Inspect extracted chemical features
   print(f"Reactant motifs: {len(result.tokens_R)} unique tokens")
   print(f"Product motifs:  {len(result.tokens_P)} unique tokens")
   print(f"Non-zero delta motifs: {len(result.delta)} (formed + consumed)")

   # Show the first 5 signed delta counts: positive = formed, negative = consumed
   for motif_id, change in list(result.delta.items())[:5]:
       role = "formed" if change > 0 else "consumed"
       print(f"  Motif {motif_id}: {role} ({change:+d})")

   # 5) Convert to binary bit-vector (list of 0/1 integers)
   bits = result.to_binary()
   print(f"Fingerprint bits: {len(bits)}, set bits: {sum(bits)}")

.. _reaction-directionality-and-antisymmetry:

2. Reaction Directionality & Antisymmetry
=========================================

The signed delta $\Delta = C_H - C_G$ is strictly antisymmetric under reaction reversal:
$\Delta(H \to G) = -\Delta(G \to H)$. Conversely, the total abundance $U = C_H + C_G$ is symmetric.

.. code-block:: python

   from synrfp import synrfp, tanimoto_bits

   forward_rxn = "CCO>>C=C.O"
   reverse_rxn = "C=C.O>>CCO"

   # In delta mode, forward and reverse reactions have distinct fingerprints
   fp_fwd_delta = synrfp(forward_rxn, mode="delta", bits=1024)
   fp_rev_delta = synrfp(reverse_rxn, mode="delta", bits=1024)
   print("Tanimoto (forward vs reverse, delta):", tanimoto_bits(fp_fwd_delta, fp_rev_delta))

   # In union mode, fragment identities are pooled regardless of side
   fp_fwd_union = synrfp(forward_rxn, mode="union", bits=1024)
   fp_rev_union = synrfp(reverse_rxn, mode="union", bits=1024)
   print("Tanimoto (forward vs reverse, union):", tanimoto_bits(fp_fwd_union, fp_rev_union))
   # In union mode, the two representations are identical (Tanimoto = 1.0)

.. _comparing-sketchers-and-similarities:

3. Comparing Sketchers & Computing Similarities
===============================================

SynRFP provides four sketch cores targeting distinct similarity semantics:

- **ParityFold**: Fast XOR parity compression into compact bitvectors.
- **MinHashSketch**: Minwise hashing estimating unweighted Jaccard set similarity.
- **CWSketch**: Consistent Weighted Sampling estimating count-aware weighted Jaccard similarity.
- **SRPSketch**: Signed Random Projection estimating cosine similarity.

.. code-block:: python

   from synrfp import synrfp, tanimoto_bits

   rxn_a = "CCO>>C=C.O"
   rxn_b = "CCCO>>CC=C.O"

   # 1) ParityFold (binary XOR hashing)
   fp_p_a = synrfp(rxn_a, sketch="parity", bits=1024)
   fp_p_b = synrfp(rxn_b, sketch="parity", bits=1024)
   print("ParityFold Tanimoto:", tanimoto_bits(fp_p_a, fp_p_b))

   # 2) b-bit MinHash (unweighted Jaccard approximation)
   fp_mh_a = synrfp(rxn_a, sketch="minhash", bits=1024)
   fp_mh_b = synrfp(rxn_b, sketch="minhash", bits=1024)
   print("MinHash Tanimoto:   ", tanimoto_bits(fp_mh_a, fp_mh_b))

   # 3) Consistent Weighted Sampling (weighted Jaccard, count-aware)
   fp_cw_a = synrfp(rxn_a, sketch="cw", bits=1024)
   fp_cw_b = synrfp(rxn_b, sketch="cw", bits=1024)
   print("CWS Tanimoto:       ", tanimoto_bits(fp_cw_a, fp_cw_b))

.. _fast-parameter-sweeps-from-tokens:

4. Rapid Parameter Sweeps with Precomputed Tokens
=================================================

When tuning sketch parameters (sketch type, bit length, seed), re-tokenizing molecular graphs
is redundant. Use :func:`~synrfp.encode.core.fingerprint_from_tokens` to bypass graph parsing:

.. code-block:: python

   from synrfp.graph.reaction import Reaction
   from synrfp.tokenizers.wl import WLTokenizer
   from synrfp.encode.core import fingerprint_from_tokens

   # 1) Tokenize once
   rxn = Reaction.from_rsmi("CCO>>C=C.O")
   tok = WLTokenizer()
   tokens_R = tok.tokens_graph(rxn.reactant, radius=2)
   tokens_P = tok.tokens_graph(rxn.product, radius=2)

   # 2) Sweep sketchers or bit lengths instantly
   fp_512  = fingerprint_from_tokens(tokens_R, tokens_P, sketch="parity", bits=512)
   fp_1024 = fingerprint_from_tokens(tokens_R, tokens_P, sketch="parity", bits=1024)
   fp_cw   = fingerprint_from_tokens(tokens_R, tokens_P, sketch="cw", bits=1024)

   print(f"fp_512 shape: {fp_512.shape}, fp_1024 shape: {fp_1024.shape}")

.. _high-throughput-batch-encoding:

5. High-Throughput Batch Encoding
=================================

Use :class:`~synrfp.encode.batch.BatchEncoder` for processing large reaction collections
with multi-core CPU parallelization:

.. code-block:: python

   from synrfp import BatchEncoder

   reactions = [
       "CCO>>C=C.O",
       "CC(=O)O.OCC>>CC(=O)OCC.O",
       "CO.O[C@@H]1CCNC1.[C-]#[N+]CC(=O)OC>>[C-]#[N+]CC(=O)N1CC[C@@H](O)C1",
       "CCOC(=O)C(CC)c1cccnc1.Cl.O>>CCC(C(=O)O)c1cccnc1",
   ]

   # Configure batch encoder with joblib parallelization
   encoder = BatchEncoder(
       tokenizer="wl",
       radius=2,
       sketch="parity",
       bits=1024,
       mode="delta",
       n_jobs=4,          # number of parallel workers
       batch_size=2,      # chunk size for streaming execution
   )

   fps = encoder.encode_many(reactions)
   print("Encoded matrix shape:", fps.shape)  # (4, 1024)

.. _persisting-fingerprints:

6. Persisting Fingerprints to Disk
==================================

Save generated fingerprints for reproducible ML workflows using NumPy or Pandas/Parquet:

.. code-block:: python

   import numpy as np
   import pandas as pd

   # 1) Fast binary NumPy dump
   np.save("reaction_fps.npy", fps)
   np.savez_compressed("reaction_fps.npz", fps=fps)

   # 2) Tabular format with metadata via Parquet
   metadata = pd.DataFrame({"reaction_smiles": reactions})
   df_fps = pd.DataFrame(fps, columns=[f"bit_{i}" for i in range(fps.shape[1])])
   df_combined = pd.concat([metadata, df_fps], axis=1)
   df_combined.to_parquet("reactions_with_fps.parquet", index=False)

.. _supervised-classification-with-mlp:

7. Supervised Classification (MLP Baseline)
===========================================

In the SynRFP publication :cite:`phan2025synrfp`, a lightweight Multilayer Perceptron (MLP)
with one hidden layer of 512 units achieved state-of-the-art results (mean MCC = 0.953)
across 12 benchmarks. Here is a minimal implementation:

.. code-block:: python

   import numpy as np
   from sklearn.neural_network import MLPClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import matthews_corrcoef, classification_report
   from synrfp import BatchEncoder

   # Sample reaction dataset (replace with actual SMILES and labels)
   rxn_data = [
       "CCO>>C=C.O",
       "CCCO>>CC=C.O",
       "CC(=O)O.OCC>>CC(=O)OCC.O",
       "CCC(=O)O.OCC>>CCC(=O)OCC.O",
   ]
   labels = [0, 0, 1, 1]

   # 1) Encode reactions to SynRFP bitvectors
   X = BatchEncoder.encode(rxn_data, tokenizer="wl", radius=2, bits=1024)
   y = np.array(labels)

   # 2) Train / test split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=42
   )

   # 3) Initialize and fit MLP classifier
   clf = MLPClassifier(
       hidden_layer_sizes=(512,),
       activation="relu",
       max_iter=200,
       random_state=42,
       early_stopping=True,
   )
   clf.fit(X_train, y_train)

   # 4) Evaluate
   y_pred = clf.predict(X_test)
   print("Validation MCC:", matthews_corrcoef(y_test, y_pred))

.. _unsupervised-reaction-clustering:

8. Unsupervised Reaction Clustering ($k$-Means)
===============================================

SynRFP fingerprints induce an effective geometry for unsupervised clustering
without requiring pre-trained transformers or supervised labels:

.. code-block:: python

   from sklearn.cluster import KMeans
   from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

   # Cluster fingerprints into k groups
   kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
   cluster_labels = kmeans.fit_predict(X)

   # Evaluate cluster agreement against ground truth
   ari = adjusted_rand_score(labels, cluster_labels)
   nmi = normalized_mutual_info_score(labels, cluster_labels)
   print(f"Adjusted Rand Index (ARI): {ari:.3f}")
   print(f"Normalized Mutual Info (NMI): {nmi:.3f}")
