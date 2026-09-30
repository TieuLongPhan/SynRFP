# SynRFP: A Generic Graph-Based Framework for Mapping-Free Reaction Fingerprints

[![PyPI version](https://img.shields.io/pypi/v/synrfp.svg)](https://pypi.org/project/synrfp/)
[![Python Version](https://img.shields.io/pypi/pyversions/synrfp.svg)](https://pypi.org/project/synrfp/)
[![Release](https://img.shields.io/github/v/release/tieulongphan/synrfp.svg)](https://github.com/tieulongphan/synrfp/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tieulongphan/synrfp/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synrfp/actions/workflows/test-and-lint.yml)
[![Documentation Status](https://readthedocs.org/projects/synrfp/badge/?version=latest)](https://synrfp.readthedocs.io/en/latest/)
[![ZENODO](https://zenodo.org/badge/1032455620.svg)](https://doi.org/10.5281/zenodo.17563778)
[![Stars](https://img.shields.io/github/stars/tieulongphan/synrfp.svg?style=social&label=Star)](https://github.com/tieulongphan/synrfp/stargazers)

**SynRFP** is a generic, **mapping-free**, permutation-invariant framework for representing chemical transformations as fixed-length fingerprints.

Unlike existing reaction fingerprints that conflate motif extraction, reactant–product combination, and sketch compression, SynRFP decomposes the featurization pipeline into three mathematically explicit, modular operators:
$$\mathbf{f}_{\Theta,\omega,\Xi}(G \to H) = \mathcal{S}_{\Xi}\Bigl(\operatorname{Comb}_{\omega}\bigl(\Psi_{\Theta}(G),\, \Psi_{\Theta}(H)\bigr)\Bigr)$$

![SynRFP Workflow](https://raw.githubusercontent.com/TieuLongPhan/SynRFP/main/doc/figure/synrfp.png)

---

## 🌟 Key Highlights

- **Mapping-Free & Robust**: Operates directly on molecular graphs without requiring atom-to-atom mapping (AAM), condensed graph of reaction (CGR) heuristics, or transition-state calculations.
- **Strict Invariance Guarantees**: Mathematically invariant to atom renumbering, component permutations (reactants, products, spectator reagents), and SMILES serialization artifacts.
- **Directional & Antisymmetric**: Signed net change encoding distinguishes formed from cleaved motifs ($\Delta(H \to G) = -\Delta(G \to H)$) while automatically canceling unchanged spectator molecules.
- **Audited representation evidence**: The current evaluation package reports
  signed-delta, reversal, and unchanged-spectator controls, feature-validity
  rates, paired clustering stability, and protocol metadata. See
  [`Experiments/11_interpretation/`](Experiments/11_interpretation/).
- **Classification and clustering benchmarks**: The current results are kept
  split-aware and model-specific; the XGBoost and MLP queues are still being
  completed, so no single pooled MCC headline is advertised here.
- **Unsupervised reaction discovery**: SynRFP leads DRFP on all seven small
  K-means endpoints by both AMI and ARI in the completed ten-seed comparison,
  while the large-data exceptions remain visible in the audit tables.
- **CPU-compatible implementation**: SynRFP can be encoded without a GPU;
  measured throughput depends on tokenizer, dataset, and hardware, so this
  README does not claim an unbenchmarked rate.

---

## 🔬 Framework Architecture

SynRFP separates reaction featurization into three independent, pluggable stages:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                    Reaction G ➔ H                      │
                  └───────────────┬────────────────────────┬───────────────┘
                                  │                        │
                          Reactants G                  Products H
                                  │                        │
                    ┌─────────────▼────────────┐ ┌─────────▼────────────┐
 1. EXTRACTION      │     Ψ_Θ(G) = C_G         │ │     Ψ_Θ(H) = C_H     │  Extract isomorphism-invariant
    (Subgraph)      │     Token Multiset       │ │     Token Multiset   │  neighborhoods / paths
                    └─────────────┬────────────┘ └─────────┬────────────┘
                                  │                        │
                                  └───────────┬────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
 2. COMBINATION     │   Comb_ω(C_G, C_H):                               │  Algebraic comparison
    (Pointwise)     │     • Δ = C_H - C_G  (Signed Net Change)          │  • Δ: preserves direction & cancels agents
                    │     • U = C_H + C_G  (Total Motif Abundance)      │  • U: encodes total structural context
                    └─────────────────────────┬─────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
 3. SKETCHING       │   S_Ξ: Compression ➔ f ∈ {0, 1}^B                 │  Randomized compression
    (Locality-Hash) │     • ParityFold (XOR hashing)                    │  • b-bit MinHash (unweighted Jaccard)
                    │     • CWSketch (count-aware weighted Jaccard)     │  • SRP (Signed Random Projection)
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                                 Fingerprint Vector f ∈ F
```

### 1. Subgraph Extractor ($\Psi_\Theta$)
Decomposes molecular graphs into discrete structural motifs:
- **`wl` (Weisfeiler–Lehman)**: Atom-centered radius-$r$ neighborhoods refined via 1D-WL color refinement (*recommended default*).
- **`morgan` (Morgan circular)**: Atom-centered circular environments across radii $0 \dots R$, mirroring extended-connectivity fingerprints.
- **`path` (Simple paths)**: Linear atom-bond sequences of lengths $m \le \ell \le L$ with lexicographical canonical orientation.
- **`nauty` (Canonical subgraphs)**: Exact canonical certificates for ego-subgraphs.
- **`map` (MinHashed Atom Pairs)**: Atom-pair style topological distance tokens.

### 2. Algebraic Combiner ($\operatorname{Comb}_\omega$)
Combines reactant ($C_G$) and product ($C_H$) token multisets:
- **Signed Delta ($\Delta = C_H - C_G$, `mode="delta"`)**: Encodes the net chemical change. Formed motifs have $\Delta > 0$, consumed motifs have $\Delta < 0$, and unchanged spectators have $\Delta = 0$. Sign-lifting ($\mathcal{C} \times \{+, -\}$) preserves directionality.
- **Total Count ($U = C_H + C_G$, `mode="union"`)**: Retains coarse molecular context across both reaction sides.

### 3. Sketcher ($\mathcal{S}_\Xi$)
Compresses sparse token summaries into fixed-length vectors:
- **`parity` (`ParityFold`)**: Deterministic XOR parity compression into compact $B$-bit fingerprints.
- **`minhash` (`MinHashSketch`)**: $m$-coordinate minwise hashing with $b$-bit encoding targeting unweighted Jaccard similarity.
- **`cw` (`CWSketch`)**: Consistent Weighted Sampling estimating weighted Jaccard similarity while preserving motif counts/multiplicities.
- **`srp` (`SRPSketch`)**: Signed Random Projection estimating cosine similarity.

---

## 📊 Benchmark Performance

The repository's current comparison is generated from the split-aware
artifacts under [`Experiments/11_interpretation/results/`](Experiments/11_interpretation/results/).
The primary benchmark keeps scaffold-CV, fixed reference splits, and factorial
random holdouts separate. XGBoost and MLP factorial runs are still in flight;
the generated readiness and progress files identify which cells are complete.

> **Historical snapshot.** The table below comes from an earlier protocol and
> must not be read as the current audited result. Use the generated audit
> tables linked above for current comparisons.

| Benchmark Category | Dataset | Size | Classes | DRFP (2048b) | RXNFP (fine-tuned) | **SynRFP (1024b)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Chemical / Patent** | `Schneider_UB` | 50,000 | 50 | 0.956 | **0.985** | 0.966 |
| | `Schneider_B` | 50,000 | 50 | 0.925 | **0.956** | 0.944 |
| | `USPTO_50k_UB` | 50,016 | 10 | 0.898 | **0.993** | 0.973 |
| | `USPTO_50k_B` | 50,016 | 10 | 0.910 | **0.989** | 0.978 |
| | `USPTO_TPL_UB` | 445,115 | 1000 | 0.977 | **0.988** | 0.952 |
| | `USPTO_TPL_B` | 445,115 | 1000 | 0.962 | **0.971** | 0.949 |
| | `SynTemp_R0` | 43,441 | 143 | 0.971 | **0.994** | 0.992 |
| **Enzymatic (ECREACT)** | `ECREACT_1st` | 185,734 | 7 | 0.915 | 0.857 | **0.921** |
| | `ECREACT_2nd` | 185,734 | 63 | 0.896 | 0.855 | **0.910** |
| | `ECREACT_3rd` | 185,734 | 175 | 0.863 | 0.825 | **0.865** |
| **Mechanistic** | `Mech_1st` | 31,673 | 9 | 0.860 | 0.844 | **0.990** |
| | `Mech_2nd` | 31,673 | 63 | 0.889 | 0.859 | **0.993** |
| **Overall Average** | *(Mean ± SD across 12 datasets)* | | | 0.918 ± 0.039 | 0.926 ± 0.067 | **0.953 ± 0.037** |

> **Historical factorial note.** The current completed kNN factorial confirms a
> strong delta-versus-union contrast, but model-specific XGBoost and MLP
> factorial effects are withheld until their full matrices are complete.

---

## ⚙️ Installation

### Requirements
- **Python**: `>= 3.11`

### Install from PyPI
```bash
# Standard installation
pip install synrfp

# With parallel batch encoding (joblib)
pip install "synrfp[parallel]"
```

### Install from Source (Development)
```bash
git clone https://github.com/TieuLongPhan/SynRFP.git
cd SynRFP
pip install -e ".[dev,parallel]"
```

Verify the installation:
```bash
python -c "import synrfp; print(synrfp.__version__)"
pytest
```

---

## 🚀 Quick Start & Usage Examples

### 1. One-Line Fingerprint Generation
Compute a fixed-length binary fingerprint directly from a reaction SMILES:

```python
from synrfp import synrfp

# Default: WL radius 2, signed delta, 1024-bit ParityFold
bits = synrfp(
    "CCO>>C=C.O",
    tokenizer="wl",    # "wl", "morgan", "path", "nauty", "map"
    radius=2,          # neighborhood radius
    sketch="parity",   # "parity", "minhash", "cw", "srp"
    bits=1024,         # fingerprint bit length
    mode="delta",      # "delta" (net change) or "union" (total abundance)
    seed=42,
)

print(f"Fingerprint: {len(bits)} bits, {sum(bits)} active bits")
print("First 16 bits:", bits[:16])
```

### 2. High-Throughput Batch Encoding
Encode thousands of reactions efficiently using serial or parallel execution:

```python
from synrfp import BatchEncoder

reactions = [
    "CCO>>C=C.O",
    "CC(=O)O.OCC>>CC(=O)OCC.O",
    "CO.O[C@@H]1CCNC1.[C-]#[N+]CC(=O)OC>>[C-]#[N+]CC(=O)N1CC[C@@H](O)C1",
]

# Parallel encoding across CPU cores with memory chunking
encoder = BatchEncoder(
    tokenizer="wl",
    radius=2,
    sketch="parity",
    bits=1024,
    mode="delta",
    n_jobs=4,          # number of parallel workers
    batch_size=64,     # chunk size for streaming execution
)

fps = encoder.encode_many(reactions)
print("Encoded array shape:", fps.shape)  # (3, 1024)
```

### 3. Object-Oriented Engine & Interpretability
Inspect reactant/product token multisets, reaction directionality, and signed deltas:

```python
from synrfp import SynRFP
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.sketchers.parity_fold import ParityFold

# 1. Parse reaction SMILES into graph structures
rxn = Reaction.from_rsmi("CCO>>C=C.O")

# 2. Build engine with explicit components
engine = SynRFP(
    tokenizer=WLTokenizer(),
    radius=2,
    sketch=ParityFold(bits=1024, seed=42),
)

# 3. Compute fingerprint result
res = engine.fingerprint(rxn.reactant, rxn.product, mode="delta")

# 4. Inspect extracted chemical information
print(f"Reactant motifs extracted: {len(res.tokens_R)}")
print(f"Product motifs extracted:  {len(res.tokens_P)}")
print(f"Net change motifs (Δ ≠ 0): {len(res.delta)}")

# Non-zero delta counts show formed (+) and cleaved (-) environments
sample_deltas = list(res.delta.items())[:5]
print("Sample signed tokens (motif_id -> change):", sample_deltas)

# Export to binary format
bits = res.to_binary()
```

### 4. Comparing Sketchers & Calculating Similarities
SynRFP provides specialized sketchers tailored to different similarity metrics:

```python
from synrfp import synrfp, tanimoto_bits

rxn1 = "CCO>>C=C.O"
rxn2 = "CCCO>>CC=C.O"

# ParityFold (compact XOR hashing)
fp_p1 = synrfp(rxn1, sketch="parity", bits=1024)
fp_p2 = synrfp(rxn2, sketch="parity", bits=1024)
print("ParityFold Tanimoto:", tanimoto_bits(fp_p1, fp_p2))

# MinHash (approximates unweighted Jaccard similarity between token sets)
fp_mh1 = synrfp(rxn1, sketch="minhash", bits=1024)
fp_mh2 = synrfp(rxn2, sketch="minhash", bits=1024)
print("MinHash Tanimoto:   ", tanimoto_bits(fp_mh1, fp_mh2))

# Consistent Weighted Sampling (CWS) (preserves motif counts and multiplicities)
fp_cw1 = synrfp(rxn1, sketch="cw", bits=1024)
fp_cw2 = synrfp(rxn2, sketch="cw", bits=1024)
print("CWS Tanimoto:       ", tanimoto_bits(fp_cw1, fp_cw2))
```

### 5. Rapid Hyperparameter Sweeps with Precomputed Tokens
Reuse extracted tokens across multiple sketchers and bit lengths without re-tokenizing:

```python
from collections import Counter
from synrfp.encode.core import fingerprint_from_tokens

# Extract or supply token multisets
tokens_R = Counter({1001: 2, 1002: 1, 1005: 3})
tokens_P = Counter({1001: 1, 1003: 1, 1005: 3})

# Sweep representations instantly
fp_512  = fingerprint_from_tokens(tokens_R, tokens_P, bits=512,  sketch="parity")
fp_1024 = fingerprint_from_tokens(tokens_R, tokens_P, bits=1024, sketch="parity")
fp_cw   = fingerprint_from_tokens(tokens_R, tokens_P, bits=1024, sketch="cw")
```

### 6. Machine Learning Integration (scikit-learn)
Train a fast classifier directly on SynRFP bit vectors:

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report
from synrfp import BatchEncoder

# Encode reactions
X = BatchEncoder.encode(reactions, tokenizer="wl", radius=2, bits=1024)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a lightweight MLP (matching paper default)
clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Test MCC:", matthews_corrcoef(y_test, y_pred))
```

---

## 📖 API Parameters Summary

The top-level `synrfp(...)` function and `BatchEncoder(...)` class accept the following parameters:

| Parameter | Type | Default | Options / Description |
| :--- | :--- | :--- | :--- |
| `tokenizer` | `str` | `"wl"` | Subgraph extractor: `"wl"`, `"morgan"`, `"path"`, `"nauty"`, `"map"`. |
| `radius` | `int` | `2` | Context radius or refinement iterations (used by WL, Morgan, Nauty). |
| `sketch` | `str` | `"parity"` | Compression scheme: `"parity"` (ParityFold), `"minhash"` (MinHash), `"cw"` (CWSketch), `"srp"` (Signed Random Projection). |
| `bits` | `int` | `1024` | Total bit length of the resulting fingerprint vector. |
| `mode` | `str` | `"delta"` | Combiner mode: `"delta"` (signed net change $C_H - C_G$) or `"union"` (total count $C_H + C_G$). |
| `seed` | `int` | `1` | Hash randomization seed for reproducible sketching. |
| `agents` | `str` | `"both"` | Treatment of spectator agents written in SMILES: `"both"` (cancels in delta), `"reactants"`, or `"drop"`. |
| `bbit` | `int` | `2` | Bits per sample for $b$-bit minwise hashing in MinHash/CWS. |
| `n_jobs` | `int` | `1` | Number of parallel worker threads/processes (`BatchEncoder` only). |
| `batch_size`| `int` | `None` | Chunk size for streaming memory management (`BatchEncoder` only). |

---

## 📚 Reproducibility & Experiments

All experiments, figures, and tables reported in the paper can be reproduced using the scripts in the [`Experiments/`](./Experiments) directory:

- `Experiments/01_data`: Benchmark curation, rare-class filtering, and data audits.
- `Experiments/02_featurize`: High-throughput reaction encoding across configurations.
- `Experiments/03_classification`: 5-NN, XGBoost, and MLP benchmarks under fixed and repeated scaffold-grouped splits; MLP runs through the GPU queue.
- `Experiments/04_factorial`: Factorial analysis across 162 design configurations.
- `Experiments/05_clustering`: $k$-means, BitBIRCH, and HDBSCAN unsupervised clustering.
- `Experiments/06_retrieval`: Metric-free k-NN similarity search across hierarchical taxonomies.
- `Experiments/07_robustness`: Invariance checks (atom/component permutations, balanced vs. unbalanced data).
- `Experiments/08_reaction_center`: Mechanistic localization and signed feature enrichment.

---

## 📜 Citation

If you use **SynRFP** in your research, please cite our paper:

```bibtex
@article{phan2025synrfp,
  title={{SynRFP}: A generic graph-based framework for mapping-free construction of reaction fingerprints},
  author={Phan, Tieu-Long and Golnik, Richard and Nguyen, Phuoc-Chung Van and Gatter, Thomas and Stadler, Peter F.},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  note={Under review}
}
```

A permanent archive of the release is available on Zenodo:
[![DOI](https://zenodo.org/badge/1032455620.svg)](https://doi.org/10.5281/zenodo.17563778)

---

## 👥 Authors & Acknowledgments

- **Tieu-Long Phan** ([GitHub](https://github.com/tieulongphan), [Website](https://tieulongphan.github.io/)) — Leipzig University & University of Southern Denmark
- **Richard Golnik** — Leipzig University
- **Phuoc-Chung Van Nguyen** — University of Medicine and Pharmacy at Ho Chi Minh City
- **Thomas Gatter** — Leipzig University
- **Peter F. Stadler** — Leipzig University, MPI MIS, University of Vienna, Santa Fe Institute

This project has received funding from the European Union's Horizon Europe Doctoral Network programme under the Marie Skłodowska-Curie grant agreement No **101072930** ([TACsy](https://tacsy.eu/) — Training Alliance for Computational systems chemistry).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
