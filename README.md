## SynRFP — Graph-based, mapping-free reaction fingerprints

[![PyPI version](https://img.shields.io/pypi/v/synrfp.svg)](https://pypi.org/project/synrfp/)
[![Release](https://img.shields.io/github/v/release/tieulongphan/synrfp.svg)](https://github.com/tieulongphan/synrfp/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tieulongphan/synrfp.svg)](https://github.com/tieulongphan/synrfp/commits)
[![ZENODO](https://zenodo.org/badge/1032455620.svg)](https://doi.org/10.5281/zenodo.17563778)
[![CI](https://github.com/tieulongphan/synrfp/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synrfp/actions/workflows/test-and-lint.yml)
[![Stars](https://img.shields.io/github/stars/tieulongphan/synrfp.svg?style=social&label=Star)](https://github.com/tieulongphan/synrfp/stargazers)

**SynRFP** (Synthesis Reaction FingerPrint) is a **mapping-free**, permutation-invariant framework for representing chemical transformations as fixed-length fingerprints.  
It explicitly factorises the fingerprint pipeline into three modular operators:

- **Ψ (extractor)** — isomorphism-invariant subgraph/token extraction from each reaction side;  
- **Φ (combination)** — algebraic reparameterisation producing the signed net change Δ and total counts U per token;  
- **𝒮 (sketcher)** — randomized compression of (Δ,U) into a fixed-dimensional fingerprint **f ∈ 𝓕**.


![SynRFP Workflow](https://raw.githubusercontent.com/TieuLongPhan/SynRFP/main/doc/figure/synrfp.png)

---

## ⚙️ Installation

```bash
# 1) Clone the repository
git clone https://github.com/TieuLongPhan/synrfp.git
cd synrfp

# 2) Install the package (with optional extras)
pip install .                  # core functionality
pip install .[all]             # with datasketch and pynauty support
```
or can install via pip
```bash
pip install synrfp
```

## 🔧 Quick Start

### 1. Single‐reaction fingerprint

```python
from synrfp.graph.reaction import Reaction
from synrfp import SynRFP
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.sketchers.parity_fold import ParityFold

# Parse RSMI into GraphData
reactant_G, product_G = Reaction.from_rsmi("CCO>>C=C.O")

# Build engine: WL at radius 1 + 1024-bit parity-fold
fp_engine = SynRFP(
    tokenizer=WLTokenizer(),
    radius=1,
    sketch=ParityFold(bits=1024, seed=42),
)

# Compute fingerprint
res = fp_engine.fingerprint(reactant_G, product_G)
print(res)               # SynRFPResult(tokens_R=3 tokens, tokens_P=3 tokens, support=0, sketch_type=bytearray)
bits = res.to_binary()   # [0,1,0,0, …]
```

### 2. One‐line wrapper

```python
from synrfp import synrfp

# Generate a 1024-bit binary fingerprint in one call
bits = synrfp(
    "CCO>>C=C.O",
    tokenizer="wl",
    radius=1,
    sketch="parity",
    bits=1024,
    seed=42,
)
print(len(bits), bits[:16])  # e.g. 1024 [0, 1, 0, 0, …]
```

### 3.  Batch encoding

```python
from synrfp import BatchEncoder

rxn_smiles = [
    "CO.O[C@@H]1CCNC1.[C-]#[N+]CC(=O)OC>>[C-]#[N+]CC(=O)N1CC[C@@H](O)C1",
    "CCOC(=O)C(CC)c1cccnc1.Cl.O>>CCC(C(=O)O)c1cccnc1",
]

# Encode two reactions into a 2×1024 array of bits
fps = BatchEncoder.encode(
    rxn_smiles,
    tokenizer="wl",
    radius=1,
    sketch="parity",
    bits=1024,
    seed=42,
    batch_size=2
)

print(fps.shape)    # (2, 1024)
print(fps[0][:16])  # first 16 bits of the first fingerprint
```

## Contributing
- [Tieu-Long Phan](https://tieulongphan.github.io/)

## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)