.. _tutorials-and-examples:

Tutorials and Examples
======================

This page collects practical, copy-and-paste examples showing how to:

- compute and inspect **SynRFP** fingerprints for single reactions,
- explore tokenizer tokens and Δ (delta / support) statistics,
- encode reaction lists in batch (CPU-friendly),
- persist fingerprints to disk (NumPy / Parquet) for ML,
- run a tiny supervised baseline (scikit-learn) using SynRFP bits,
- compare sketchers (ParityFold, MinHash, CWSketch) and compute similarities.

If you have not installed **SynRFP** yet, see :doc:`Getting Started <getting_started>`.


Single reaction — token + fingerprint
------------------------------------

This example shows the common, low-level flow: parse reaction SMILES, create graph objects,
tokenize, aggregate Δ (net change), and sketch.

.. code-block:: python

   from synrfp.graph.reaction import Reaction
   from synrfp.tokenizers.wl import WLTokenizer
   from synrfp.sketchers.parity_fold import ParityFold
   from synrfp.synrfp import SynRFP

   # 1) parse RSMI -> GraphData pair (reactant_graph, product_graph)
   reactant_G, product_G = Reaction.from_rsmi("CCO>>C=C.O")

   # 2) instantiate tokenizer + sketcher
   tokenizer = WLTokenizer()                 # WL-style subtree tokens
   sketcher = ParityFold(bits=1024, seed=42) # XOR parity fold to 1024 bits

   # 3) build engine (composes tokenizer + sketcher)
   engine = SynRFP(tokenizer=tokenizer, radius=1, sketch=sketcher)

   # 4) compute fingerprint result (object contains tokens, Δ, and sketch)
   result = engine.fingerprint(reactant_G, product_G)

   # Print a human-friendly summary
   print(result)          # e.g., tokens_R=..., tokens_P=..., support=..., sketch_type=...
   bits = result.to_binary()   # list/array of 0/1 for downstream use
   print("bits length:", len(bits), "sample:", bits[:16])

Inspect tokenizer tokens and Δ (deltas)
--------------------------------------

Often you want to inspect local tokens and the signed delta U (counts) to validate the representation.

.. code-block:: python

   # Token extraction (tokenizer API: tokens_graph or equivalent)
   # NOTE: API name may vary by version; common pattern: tokenizer.tokens_graph(graph)
   toks_R = tokenizer.tokens_graph(reactant_G, radius=1)
   toks_P = tokenizer.tokens_graph(product_G, radius=1)
   print("Reactant tokens (sample):", list(toks_R)[:20])
   print("Product tokens (sample):",  list(toks_P)[:20])

   # If your SynRFP engine exposes a delta / support view:
   # The exact attribute names differ by release; typical pattern:
   print("Delta (signed counts):", result.delta)   # token -> integer (neg/pos)
   print("Total support (unique tokens):", result.support)

   # If method names differ, inspect result.__dict__ or repr(result) to find fields.

Batch encoding
--------------

Encode many reactions in one call. Use the BatchEncoder convenience helper for throughput and batching.

.. code-block:: python

   from synrfp import BatchEncoder

   rxn_smiles = [
       "CCO>>C=C.O",
       "CO.O[C@@H]1CCNC1.[C-]#[N+]CC(=O)OC>>[C-]#[N+]CC(=O)N1CC[C@@H](O)C1",
       # ...
   ]

   fps = BatchEncoder.encode(
       rxn_smiles,
       tokenizer="wl",      # accepts string aliases: "wl", "nauty", ...
       radius=1,
       sketch="parity",     # "parity", "minhash", "cw"
       bits=1024,
       seed=42,
       batch_size=64,       # tune to your RAM/CPU
       n_jobs=4,            # if BatchEncoder supports parallel backends
   )

   # fps is a (N, B) NumPy array or list-of-lists depending on version
   import numpy as np
   fps = np.asarray(fps)
   print("fingerprints shape:", fps.shape)

Persisting fingerprints
-----------------------

Save fingerprints and metadata for reuse (NumPy + Pandas / Parquet recommended).

.. code-block:: python

   import numpy as np
   import pandas as pd

   # fps: (N, B) binary array, meta: list of rxn smiles
   np.save("fps.npy", fps)  # fast binary dump
   np.savez_compressed("fps_compressed.npz", fps=fps)

   # for ML: save in Parquet with metadata (index aligned)
   meta = pd.DataFrame({"rxn": rxn_smiles, "label": labels})  # if supervised
   df_bits = pd.DataFrame(fps.astype(int), index=meta.index)
   output = pd.concat([meta, df_bits.add_prefix("b_")], axis=1)
   output.to_parquet("fps_table.parquet")

Quick ML baseline: RandomForest classifier
-----------------------------------------

A minimal pipeline: encode, train/test split, train a RandomForest and evaluate.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import matthews_corrcoef, f1_score

   # assume fps (N,B) and labels (N,)
   X_train, X_test, y_train, y_test = train_test_split(fps, labels, test_size=0.2, random_state=0, stratify=labels)

   clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
   clf.fit(X_train, y_train)

   y_pred = clf.predict(X_test)
   print("MCC:", matthews_corrcoef(y_test, y_pred))
   print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))

Comparing sketchers and computing similarity
-------------------------------------------

You may want to compare different sketchers (binary parity vs MinHash vs weighted CWSketch) or compute similarity (Tanimoto/Jaccard for bits).

.. code-block:: python

   import numpy as np

   def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
       # expects binary 0/1 arrays
       a = a.astype(bool)
       b = b.astype(bool)
       inter = np.logical_and(a, b).sum()
       union = np.logical_or(a, b).sum()
       return float(inter) / union if union > 0 else 0.0

   # Example: compute pairwise similarity for first 10 fps
   sims = np.zeros((10, 10))
   for i in range(10):
       for j in range(10):
           sims[i, j] = tanimoto(fps[i], fps[j])
   print(sims[:3, :3])

   # To compare sketchers: create engines with different sketch classes and encode the same reactions;
   # then compute pairwise correlations between resulting representations (e.g. cosine on MinHash int vectors).
