"""Fingerprint computation and caching.

Features are cached per *scope* (a reaction table, or a subsample of one) as
``work/features/<scope>/<fingerprint>.npz`` with

* ``X``: packed bits (``np.packbits``) for bit fingerprints, ``float32`` otherwise;
* ``n_bits``: number of bits (0 for real-valued features);
* ``ok``: boolean mask of reactions that could be featurised.

SynRFP configurations sharing a tokenizer and radius are computed from one
tokenization per reaction (``fingerprint_from_tokens``), so the 162-point grid
costs nine tokenizations, not 162.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from .config import EXP_DIR, SynRFPConfig, log

REPO_ROOT = EXP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# loky workers do not inherit sys.path; make the local package importable there
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO_ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
)

EXTERNAL = ("rxnfp_pretrained", "rxnfp_ft")
CHUNK = 500


class Features:
    """Feature matrix with its kind (``bits`` or ``dense``)."""

    def __init__(self, X: np.ndarray, n_bits: int, ok: np.ndarray):
        self.X, self.n_bits, self.ok = X, int(n_bits), ok

    @property
    def is_bits(self) -> bool:
        return self.n_bits > 0

    def dense(self, rows: np.ndarray | None = None) -> np.ndarray:
        X = self.X if rows is None else self.X[rows]
        if self.is_bits:
            return np.unpackbits(X, axis=1, count=self.n_bits).astype(np.float32)
        return X.astype(np.float32, copy=False)

    def packed(self, rows: np.ndarray | None = None) -> np.ndarray:
        if not self.is_bits:
            raise TypeError("real-valued features have no packed form")
        return self.X if rows is None else self.X[rows]


def _path(cfg: Dict[str, Any], scope: str, fp: str) -> Path:
    return cfg["paths"]["work"] / "features" / scope / f"{fp}.npz"


def load(cfg: Dict[str, Any], scope: str, fp: str) -> Features:
    p = _path(cfg, scope, fp)
    if not p.exists():
        raise FileNotFoundError(f"missing features {p}; run 02_featurize.py")
    z = np.load(p)
    return Features(z["X"], int(z["n_bits"]), z["ok"])


def _save(cfg: Dict[str, Any], scope: str, fp: str, X, n_bits, ok) -> None:
    p = _path(cfg, scope, fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, X=X, n_bits=np.int64(n_bits), ok=ok)
    tmp.replace(p)


# ---------------------------------------------------------------------------
# SynRFP
# ---------------------------------------------------------------------------
def _tokenizer(name: str, node_attrs, edge_attrs):
    from synrfp.tokenizers.morgan import MorganTokenizer
    from synrfp.tokenizers.path import PathTokenizer
    from synrfp.tokenizers.wl import WLTokenizer

    cls = {"wl": WLTokenizer, "morgan": MorganTokenizer, "path": PathTokenizer}[name]
    return cls(node_attrs=node_attrs, edge_attrs=edge_attrs)


def _synrfp_chunk(
    rsmis: Sequence[str],
    tok_name: str,
    radius: int,
    configs: List[SynRFPConfig],
    scfg: Dict[str, Any],
    fp_seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    from synrfp.encode.core import fingerprint_from_tokens
    from synrfp.graph.reaction import Reaction

    tok = _tokenizer(tok_name, scfg["node_attrs"], scfg["edge_attrs"])
    out = {c.name: np.zeros((len(rsmis), c.bits // 8), dtype=np.uint8) for c in configs}
    ok = np.zeros(len(rsmis), dtype=bool)
    for i, rsmi in enumerate(rsmis):
        try:
            rxn = Reaction.from_rsmi(rsmi, agents=scfg["agents"])
            tr = tok.tokens_graph(rxn.reactant, radius)
            tp = tok.tokens_graph(rxn.product, radius)
        except Exception:  # unparsable record: all-zero row, flagged in `ok`
            continue
        for c in configs:
            fp = fingerprint_from_tokens(
                tr, tp, sketch=c.sketch, bits=c.bits, mode=c.mode,
                seed=fp_seed, bbit=scfg["bbit"],
            )
            out[c.name][i] = np.packbits(fp)
        ok[i] = True
    return out, ok


def compute_synrfp(
    cfg: Dict[str, Any], scope: str, rsmis: Sequence[str], configs: List[SynRFPConfig]
) -> None:
    todo = [c for c in configs if not _path(cfg, scope, c.name).exists()]
    groups: Dict[Tuple[str, int], List[SynRFPConfig]] = defaultdict(list)
    for c in todo:
        groups[(c.tokenizer, c.radius)].append(c)
    for (tok, r), cs in groups.items():
        log.info("%s: SynRFP %s r=%d -> %d configs, %d reactions", scope, tok, r,
                 len(cs), len(rsmis))
        chunks = [rsmis[i:i + CHUNK] for i in range(0, len(rsmis), CHUNK)]
        res = Parallel(n_jobs=cfg["n_jobs"])(
            delayed(_synrfp_chunk)(ch, tok, r, cs, cfg["synrfp"], cfg["fp_seed"])
            for ch in chunks
        )
        ok = np.concatenate([o for _, o in res])
        for c in cs:
            X = np.concatenate([d[c.name] for d, _ in res])
            _save(cfg, scope, c.name, X, c.bits, ok)
        if (~ok).any():
            log.warning("%s: %d unparsable reactions", scope, int((~ok).sum()))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _drfp_chunk(rsmis, params) -> Tuple[np.ndarray, np.ndarray]:
    from drfp import DrfpEncoder

    n = params["n_folded_length"]
    X = np.zeros((len(rsmis), n // 8), dtype=np.uint8)
    ok = np.zeros(len(rsmis), dtype=bool)
    for i, s in enumerate(rsmis):
        try:
            fp = DrfpEncoder.encode([s], n_folded_length=n, radius=params["radius"],
                                    rings=params["rings"])[0]
            X[i] = np.packbits(np.asarray(fp, dtype=np.uint8))
            ok[i] = True
        except Exception:
            pass
    return X, ok


def _rdkit_chunk(rsmis, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    from rdkit import DataStructs, RDLogger
    from rdkit.Chem import rdChemReactions

    RDLogger.DisableLog("rdApp.*")
    if kind == "rdkit_struct":
        X = np.zeros((len(rsmis), 4096 // 8), dtype=np.uint8)
    else:
        X = np.zeros((len(rsmis), 2048), dtype=np.float32)
    ok = np.zeros(len(rsmis), dtype=bool)
    for i, s in enumerate(rsmis):
        try:
            rxn = rdChemReactions.ReactionFromSmarts(s, useSmiles=True)
            if kind == "rdkit_struct":
                fp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn)
                arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                X[i] = np.packbits(arr)
            else:
                fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
                for k, v in fp.GetNonzeroElements().items():
                    X[i, k] = v
            ok[i] = True
        except Exception:
            pass
    return X, ok


def compute_baseline(cfg: Dict[str, Any], scope: str, rsmis: Sequence[str], fp: str) -> None:
    if _path(cfg, scope, fp).exists():
        return
    chunks = [rsmis[i:i + CHUNK] for i in range(0, len(rsmis), CHUNK)]
    log.info("%s: %s, %d reactions", scope, fp, len(rsmis))
    if fp == "drfp":
        params = cfg["baselines"]["drfp"]
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_drfp_chunk)(c, params) for c in chunks)
        n_bits = params["n_folded_length"]
    elif fp in ("rdkit_struct", "rdkit_diff"):
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_rdkit_chunk)(c, fp) for c in chunks)
        n_bits = 4096 if fp == "rdkit_struct" else 0
    else:
        raise ValueError(f"unknown baseline {fp!r}")
    X = np.concatenate([x for x, _ in res])
    ok = np.concatenate([o for _, o in res])
    _save(cfg, scope, fp, X, n_bits, ok)


def compute(cfg: Dict[str, Any], scope: str, rsmis: Sequence[str], fps: List[str]) -> None:
    """Compute every missing fingerprint in ``fps`` for the reactions of ``scope``."""
    syn = [SynRFPConfig.from_name(f) for f in fps if f.startswith("synrfp-")]
    if syn:
        compute_synrfp(cfg, scope, list(rsmis), syn)
    for f in fps:
        if f.startswith("synrfp-"):
            continue
        if f in EXTERNAL:
            if not _path(cfg, scope, f).exists() and f == "rxnfp_pretrained":
                log.warning("%s: %s missing; run external/rxnfp_embed.py", scope, f)
            continue
        compute_baseline(cfg, scope, list(rsmis), f)


def external_ft_path(cfg: Dict[str, Any], dataset: str, scheme: str, seed: int) -> Path:
    """Fine-tuned RXNFP embeddings are split specific (trained on that split's train set)."""
    return cfg["paths"]["work"] / "features" / "rxnfp_ft" / dataset / f"{scheme}_s{seed}.npz"


def load_for_run(
    cfg: Dict[str, Any], scope: str, fp: str, dataset: str, scheme: str, seed: int
) -> Features:
    if fp == "rxnfp_ft":
        p = external_ft_path(cfg, dataset, scheme, seed)
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run external/rxnfp_finetune.py")
        z = np.load(p)
        return Features(z["X"], 0, z["ok"])
    return load(cfg, scope, fp)


def compute_in_memory(cfg: Dict[str, Any], rsmis: Sequence[str], fp: str) -> Features:
    """Fingerprints for an ad-hoc list of reactions, without caching."""
    rsmis = list(rsmis)
    chunks = [rsmis[i:i + CHUNK] for i in range(0, len(rsmis), CHUNK)]
    if fp.startswith("synrfp-"):
        c = SynRFPConfig.from_name(fp)
        res = Parallel(n_jobs=cfg["n_jobs"])(
            delayed(_synrfp_chunk)(ch, c.tokenizer, c.radius, [c], cfg["synrfp"],
                                   cfg["fp_seed"]) for ch in chunks)
        X = np.concatenate([d[c.name] for d, _ in res])
        ok = np.concatenate([o for _, o in res])
        return Features(X, c.bits, ok)
    if fp == "drfp":
        params = cfg["baselines"]["drfp"]
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_drfp_chunk)(c, params) for c in chunks)
        n_bits = params["n_folded_length"]
    elif fp in ("rdkit_struct", "rdkit_diff"):
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_rdkit_chunk)(c, fp) for c in chunks)
        n_bits = 4096 if fp == "rdkit_struct" else 0
    else:
        raise ValueError(f"cannot compute {fp!r} in memory")
    return Features(np.concatenate([x for x, _ in res]), n_bits,
                    np.concatenate([o for _, o in res]))


def rowwise_similarity(a: Features, b: Features) -> np.ndarray:
    """Similarity of row i of ``a`` with row i of ``b`` (Tanimoto or cosine)."""
    A, B = a.dense(), b.dense()
    if a.is_bits:
        inter = (A * B).sum(1)
        union = A.sum(1) + B.sum(1) - inter
        return np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    dot = (A * B).sum(1)
    both_zero = (na == 0) & (nb == 0)
    sim = np.divide(dot, na * nb, out=np.zeros_like(dot), where=(na * nb) > 0)
    sim[both_zero] = 1.0
    return sim


def rowwise_identical(a: Features, b: Features) -> np.ndarray:
    """Exact equality of row i of ``a`` and row i of ``b``."""
    return np.all(a.X == b.X, axis=1)
