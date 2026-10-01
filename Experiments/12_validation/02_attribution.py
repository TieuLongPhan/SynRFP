#!/usr/bin/env python
"""Step 2. Attribution: cancellation, multiplicity, direction, and folding.

Everything except the payload is held fixed within a contrast: input policy,
vocabulary, bit budget, hash seed, learner, and split. With side counts
c_R, c_P of a token t, the payload keys are

  set_unsigned        t with (c_R > 0) xor (c_P > 0)       symmetric set difference
  set_signed          (t, side) for the same t
  multiset_unsigned   t with c_P != c_R
  multiset_signed     (t, sign(c_P - c_R))                 SynRFP default
  pooled              t with c_R + c_P > 0                 SynRFP union
  pooled_side_tagged  (t, R) and (t, P) for every present t

Two vocabularies: `wl` (SynRFP tokens, archived radius and attributes) and
`drfp_shingle` (DRFP substructure SMILES; the count of a shingle on a side is
the number of molecules that contain it, because DRFP shingles are sets per
molecule). Every distinct key is hashed once per seed; `parity` XORs
colliding keys and `or` keeps the bit set. `exact` is Tanimoto on the
unhashed key sets (5-NN only).

Learners: archived 5-NN for every arm; MLP (archived sklearn settings) for one
setting per payload (study.yaml: attribution.mlp). Splits are the benchmark
grouped-CV codes restricted to the study rows.

One CSV per (dataset, vocabulary) under runs/attribution/; finished files are
skipped, so the script can be restarted or split with --datasets/--vocabularies.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import vlib as L
from synrfp_exp.config import log, setup_logging
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features
from synrfp_exp.models import _vote, fit_predict, similarity_topk
from synrfp_exp.splits import TEST, make_split, subsample_rows

PAYLOADS = ["set_unsigned", "set_signed", "multiset_unsigned", "multiset_signed",
            "pooled", "pooled_side_tagged"]


def payload_keys(cR, cP):
    """Integer keys; signed keys use the library encoding 2t + (0 created, 1 consumed)."""
    out = {p: [] for p in PAYLOADS}
    for t in cR.keys() | cP.keys():
        r, p = cR.get(t, 0), cP.get(t, 0)
        out["pooled"].append(t)
        if r:
            out["pooled_side_tagged"].append(2 * t + 1)
        if p:
            out["pooled_side_tagged"].append(2 * t)
        if (r > 0) != (p > 0):
            out["set_unsigned"].append(t)
            out["set_signed"].append(2 * t + (0 if p else 1))
        if r != p:
            out["multiset_unsigned"].append(t)
            out["multiset_signed"].append(2 * t + (0 if p > r else 1))
    return out


def side_counts(rsmi, vocab, scfg, dcfg):
    if vocab == "wl":
        from synrfp.graph.reaction import Reaction
        from synrfp.tokenizers.wl import WLTokenizer

        tok = WLTokenizer(node_attrs=scfg["node_attrs"], edge_attrs=scfg["edge_attrs"])
        rxn = Reaction.from_rsmi(rsmi, agents=scfg["agents"])
        r = scfg["default"]["radius"]
        return tok.tokens_graph(rxn.reactant, r), tok.tokens_graph(rxn.product, r)
    from drfp import DrfpEncoder
    from rdkit import Chem
    from synrfp.tokenizers.utils import _h64

    reac, agent, prod = rsmi.strip().split(" ")[0].split(">")
    out = []
    for field in (".".join(x for x in (reac, agent) if x), prod):  # DRFP joins agents left
        c = Counter()
        for s in (x for x in field.split(".") if x):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(s)
            for sh in DrfpEncoder.shingling_from_mol(mol, radius=dcfg["radius"],
                                                     rings=dcfg["rings"]):
                c[_h64(("drfp", sh))] += 1
        out.append(c)
    return out[0], out[1]


def chunk(rsmis, vocab, scfg, dcfg, seeds):
    """Per reaction: {payload: uint64 array (seeds x keys)}, or None on failure."""
    from rdkit import RDLogger
    from synrfp.tokenizers.utils import _h64

    RDLogger.DisableLog("rdApp.*")
    res = []
    for s in rsmis:
        try:
            keys = payload_keys(*side_counts(s, vocab, scfg, dcfg))
        except Exception:
            res.append(None)
            continue
        res.append({p: np.array([[_h64(("pf", int(k)), seed=sd) for k in ks] for sd in seeds],
                                dtype=np.uint64).reshape(len(seeds), len(ks))
                    for p, ks in keys.items()})
    return res


def fold(hashes, bits, kind):
    X = np.zeros((len(hashes), bits), dtype=np.uint8)
    for i, h in enumerate(hashes):
        if h is None or not len(h):
            continue
        idx = (h % np.uint64(bits)).astype(np.int64)
        if kind == "or":
            X[i, idx] = 1
        else:
            X[i] = (np.bincount(idx, minlength=bits) & 1).astype(np.uint8)
    return X


def mcc(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def knn_mcc(X, y, code, k):
    tr, te = np.flatnonzero(code != TEST), np.flatnonzero(code == TEST)
    idx, _ = similarity_topk(X[te].astype(np.float32), X[tr].astype(np.float32), k,
                             tanimoto=True)
    return mcc(y[te], _vote(y[tr][idx]))


def mlp_mcc(X, y, code, params, seed):
    tr, te = np.flatnonzero(code != TEST), np.flatnonzero(code == TEST)
    feats = Features(np.packbits(X, axis=1), X.shape[1], np.ones(len(X), dtype=bool))
    pred = fit_predict("mlp", params, feats, tr, y[tr], te[:0], y[:0], te, seed=seed, n_jobs=1)
    return mcc(y[te], pred)


def exact_mcc(hashes, y, code, k):
    """Exact Tanimoto on key sets through a sparse incidence matrix."""
    from scipy import sparse

    lens = np.array([0 if h is None else len(h) for h in hashes])
    flat = np.concatenate([h for h in hashes if h is not None and len(h)])
    _, cols = np.unique(flat, return_inverse=True)
    rows = np.repeat(np.arange(len(hashes)), lens)
    M = sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)),
                          shape=(len(hashes), int(cols.max()) + 1))
    tr, te = np.flatnonzero(code != TEST), np.flatnonzero(code == TEST)
    D, Q = M[tr], M[te]
    Dt = D.T.tocsr()
    dn = np.asarray(D.sum(1)).ravel()
    pred = np.empty(len(te), dtype=y.dtype)
    for s in range(0, len(te), 512):
        q = Q[s:s + 512]
        S = (q @ Dt).toarray()
        union = np.asarray(q.sum(1)).ravel()[:, None] + dn[None, :] - S
        S = np.divide(S, union, out=np.zeros_like(S), where=union > 0)
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        idx = np.take_along_axis(part, np.argsort(-ps, axis=1, kind="stable"), axis=1)
        pred[s:s + 512] = _vote(y[tr][idx])
    return mcc(y[te], pred)


def counterexamples(rsmis, vocab, scfg, dcfg):
    """Share of reactions whose key set is unchanged by reversal, by an added
    spectator on both sides, and by doubling every component (stoichiometry)."""
    rows = []
    for s in rsmis:
        try:
            a, m, b = s.split(" ")[0].split(">")
            if m:
                continue
            base = payload_keys(*side_counts(s, vocab, scfg, dcfg))
            other = {
                "reversal": payload_keys(*side_counts(f"{b}>>{a}", vocab, scfg, dcfg)),
                "spectator": payload_keys(*side_counts(f"{a}.CCOC(C)=O>>{b}.CCOC(C)=O",
                                                       vocab, scfg, dcfg)),
                "doubled": payload_keys(*side_counts(f"{a}.{a}>>{b}.{b}", vocab, scfg, dcfg)),
            }
        except Exception:
            continue
        rows.append({f"{p}__{n}": set(base[p]) == set(o[p]) for p in PAYLOADS
                     for n, o in other.items()})
    d = pd.DataFrame(rows).mean().rename("identical_fraction").reset_index()
    d[["payload", "perturbation"]] = d["index"].str.split("__", expand=True)
    d["n"] = len(rows)
    return d.drop(columns="index")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--vocabularies", nargs="*")
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args()
    setup_logging()
    study = L.study_settings()
    a = study["attribution"]
    policy = args.policy or study["primary_policy"]
    cfg = L.load_policy(policy)
    out = L.RUNS / "attribution" / policy
    out.mkdir(parents=True, exist_ok=True)
    seeds, k = a["hash_seeds"], cfg["models"]["knn"]["k"]
    nj = cfg["n_jobs"]
    datasets = [d for d in (args.datasets or a["datasets"])
                if (cfg["paths"]["work"] / "tables" / f"ds_{d}.pkl").exists()]
    written = []
    for ds in datasets:
        table = dataset_table(cfg, ds)
        pos = subsample_rows(table["y"].to_numpy(), a["max_records"], seed=0)
        y = table["y"].to_numpy()[pos]
        rs = table["rxn"].to_numpy()[pos].tolist()
        codes = {s: make_split(cfg, ds, "cv", s)[pos] for s in a["cv_seeds"]}
        for vocab in args.vocabularies or a["vocabularies"]:
            target = out / f"{ds}__{vocab}.csv"
            written.append(target)
            if target.exists():
                log.info("%s exists, skipped", target.name)
                continue
            log.info("%s %s: tokenizing %d reactions", ds, vocab, len(rs))
            res = Parallel(n_jobs=nj)(
                delayed(chunk)(rs[i:i + 500], vocab, cfg["synrfp"], cfg["baselines"]["drfp"],
                               seeds) for i in range(0, len(rs), 500))
            res = [r for ch in res for r in ch]
            n_failed = sum(r is None for r in res)
            if vocab == "wl":   # the default payload must reproduce the library exactly
                from synrfp_exp.features import compute_in_memory
                from synrfp_exp.config import default_config

                ref = compute_in_memory(cfg, rs[:2000], default_config(cfg).name)
                sd = seeds.index(cfg["fp_seed"])
                mine = fold([None if r is None else r["multiset_signed"][sd] for r in res[:2000]],
                            default_config(cfg).bits, "parity")
                assert np.array_equal(ref.dense().astype(np.uint8)[ref.ok], mine[ref.ok]), \
                    "multiset_signed/parity does not reproduce the SynRFP default"
                log.info("%s: default payload reproduces library fingerprints", ds)
            runs = []
            for p in a["payloads"]:
                n_keys = float(np.mean([r[p].shape[1] for r in res if r is not None]))
                jobs = []
                for j, sd in enumerate(seeds):
                    hs = [None if r is None else r[p][j] for r in res]
                    for bits in a["bits"]:
                        for kind in a["folds"]:
                            X = fold(hs, bits, kind)
                            meta = dict(dataset=ds, vocabulary=vocab, payload=p, fold=kind,
                                        bits=bits, hash_seed=sd, mean_keys=n_keys,
                                        mean_bits_set=float(X.sum(1).mean()),
                                        n_failed=n_failed, n=len(rs))
                            jobs += [(dict(meta, model="knn", seed=s),
                                      delayed(knn_mcc)(X, y, c, k)) for s, c in codes.items()]
                            m = a["mlp"]
                            if (not args.no_mlp and bits == m["bits"] and kind == m["fold"]
                                    and sd == m["hash_seed"]):
                                jobs += [(dict(meta, model="mlp", seed=s),
                                          delayed(mlp_mcc)(X, y, c, cfg["models"]["mlp"], s))
                                         for s, c in codes.items()]
                if a["exact_reference"]:
                    hs = [None if r is None else r[p][0] for r in res]
                    jobs += [(dict(dataset=ds, vocabulary=vocab, payload=p, fold="exact", bits=0,
                                   hash_seed=0, mean_keys=n_keys, mean_bits_set=n_keys,
                                   n_failed=n_failed, n=len(rs), model="knn", seed=s),
                              delayed(exact_mcc)(hs, y, c, k)) for s, c in codes.items()]
                # 5-NN in threads (BLAS releases the GIL, matrices are shared);
                # MLP fits in separate processes (pure-Python training loop)
                fast = [(m, j) for m, j in jobs if m["model"] == "knn"]
                slow = [(m, j) for m, j in jobs if m["model"] != "knn"]
                scores = Parallel(n_jobs=max(1, nj // 4), prefer="threads")(j for _, j in fast)
                scores += Parallel(n_jobs=nj)(j for _, j in slow) if slow else []
                runs += [dict(meta, mcc=s) for (meta, _), s in zip(fast + slow, scores)]
                d = pd.DataFrame(runs)
                d = d[d.payload.eq(p) & d.fold.eq("parity") & d.model.eq("knn")]
                log.info("%s %s %-20s 5-NN parity MCC=%.4f", ds, vocab, p, d.mcc.mean())
            pd.DataFrame(runs).to_csv(target, index=False)
            ce = out / f"counterexamples__{ds}__{vocab}.csv"
            rng = np.random.default_rng(0)
            sample = [rs[i] for i in rng.choice(len(rs), min(a["counterexample_sample"],
                                                             len(rs)), replace=False)]
            counterexamples(sample, vocab, cfg["synrfp"], cfg["baselines"]["drfp"]).to_csv(
                ce, index=False)
            written.append(ce)
    L.write_manifest(f"02_attribution_{policy}", written)


if __name__ == "__main__":
    main()
