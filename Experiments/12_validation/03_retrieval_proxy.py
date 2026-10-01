#!/usr/bin/env python
"""Step 3. Cross-database retrieval proxy with automatic relevance labels.

Queries are ECReact reactions that come from BRENDA, PathBank, or MetaNetX.
The reference collection is a frozen Rhea release. A reference reaction is
relevant to a query when their EC numbers agree to `label_level` digits.
This is an automatic proxy: it shows whether representations separate at all
before any expert adjudication is paid for. EC agreement is not proof of
chemical usefulness, enzyme activity, or mechanism.

Design choices that keep the comparison fair:
  * Every string passes the primary input policy.
  * A Rhea master reaction is one candidate. Its score is the larger of the
    similarities to its two written directions, so direction-blind and
    direction-aware fingerprints face the same candidate list.
  * Candidates that are the same reaction as the query (either direction,
    after canonicalization) are removed for that query.
  * Queries are split by EC number into development and final parts
    (study.yaml: retrieval.development_fraction). Look at development only
    until the comparison is frozen; `--part final` is run once.
  * A failed query encoding scores zero and stays in the denominator.

Stages (run in order; each is restartable):
  --stage fetch      download Rhea tables and the ECReact source table, with checksums
  --stage build      queries.csv, corpus.csv, overlap exclusions, coverage report
  --stage featurize  label-free fingerprints -> features/<name>.npz
  --stage score      P@k per query, strata, paired cluster bootstrap

Learned encoders: run 03b_retrieval_rxnfp.py in the RXNFP environment after
`build`; `score` picks up every features/*.npz it finds.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

import vlib as L
from synrfp_exp.config import expand_fingerprints, log, setup_logging
from synrfp_exp.features import compute_in_memory
from synrfp_exp.models import similarity_topk

EXT = L.HERE / "external"


def out_dir():
    d = L.RUNS / "retrieval_proxy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(r):
    EXT.mkdir(exist_ok=True)
    files = {"rhea-release.properties": f"{r['rhea_base']}/rhea-release.properties",
             "rhea-reaction-smiles.tsv": f"{r['rhea_base']}/tsv/rhea-reaction-smiles.tsv",
             "rhea-directions.tsv": f"{r['rhea_base']}/tsv/rhea-directions.tsv",
             "rhea2ec.tsv": f"{r['rhea_base']}/tsv/rhea2ec.tsv",
             "ecreact-1.0.csv": r["ecreact_url"]}
    stamp = {}
    for name, url in files.items():
        path = EXT / name
        if not path.exists():
            log.info("downloading %s", url)
            urllib.request.urlretrieve(url, path)
        stamp[name] = {"url": url, "sha256": L.sha256(path), "bytes": path.stat().st_size}
    stamp["rhea_release"] = (EXT / "rhea-release.properties").read_text().strip()
    (EXT / "SOURCES.json").write_text(json.dumps(stamp, indent=1))
    log.info("sources frozen: %s", stamp["rhea_release"].replace("\n", ", "))


def undirected_key(s):
    a, _, b = s.partition(">>") if ">>" in s else (s.split(">")[0], "", s.split(">")[-1])
    return "|".join(sorted((a, b)))


def ec_prefix(ec, level):
    parts = str(ec).split(".")
    if len(parts) < level or any(p in ("-", "") or not p.isdigit() for p in parts[:level]):
        return None
    return ".".join(parts[:level])


def build(r, policy):
    out = out_dir()
    from joblib import Parallel, delayed

    def canon(strings):
        res = Parallel(n_jobs=-1, batch_size=1024)(
            delayed(L.apply_policy)(s, policy) for s in strings)
        return [x[0] for x in res], [x[1] for x in res]

    # ---- corpus: Rhea master reactions with an EC number, both directions
    smi = pd.read_csv(EXT / "rhea-reaction-smiles.tsv", sep="\t", header=None,
                      names=["rhea_id", "rxn_source"])
    dirs = pd.read_csv(EXT / "rhea-directions.tsv", sep="\t")
    ec = pd.read_csv(EXT / "rhea2ec.tsv", sep="\t").rename(columns={"ID": "ec"})
    ec = ec.groupby("MASTER_ID").ec.apply(lambda x: sorted(set(x))).rename("ecs")
    d = dirs.melt(id_vars="RHEA_ID_MASTER", value_vars=["RHEA_ID_LR", "RHEA_ID_RL"],
                  var_name="direction", value_name="rhea_id")
    corpus = d.merge(smi, on="rhea_id").merge(ec, left_on="RHEA_ID_MASTER", right_index=True)
    corpus = corpus.rename(columns={"RHEA_ID_MASTER": "master"})
    corpus["direction"] = corpus["direction"].str[-2:]
    corpus["rxn"], corpus["policy_status"] = canon(corpus.rxn_source.tolist())
    corpus["key"] = corpus.rxn.map(undirected_key)
    corpus["ec_list"] = corpus.ecs.map(";".join)
    corpus = corpus.drop(columns="ecs").sort_values(["master", "direction"]).reset_index(drop=True)

    # ---- queries: ECReact reactions from the other source databases
    q = pd.read_csv(EXT / "ecreact-1.0.csv")
    n_all = len(q)
    q = q[q.source.isin(r["query_sources"])].copy()
    q["rxn_source"] = q.rxn_smiles.str.replace(r"\|[^>]*>>", ">>", regex=True)
    q["rxn"], q["policy_status"] = canon(q.rxn_source.tolist())
    q["key"] = q.rxn.map(undirected_key)
    q["label"] = q.ec.map(lambda e: ec_prefix(e, r["label_level"]))
    steps = {"ecreact_rows": n_all, "from_query_sources": len(q)}
    q = q[q.label.notna()]
    steps["with_complete_ec"] = len(q)
    q = q.drop_duplicates(["key", "ec"])
    steps["after_duplicate_queries_removed"] = len(q)
    in_rhea = q.key.isin(set(corpus.key))
    q["same_reaction_in_corpus"] = in_rhea.values
    steps["queries_identical_to_a_corpus_reaction"] = int(in_rhea.sum())
    labels = {m: {ec_prefix(e, r["label_level"]) for e in s.split(";")} - {None}
              for m, s in corpus.drop_duplicates("master").set_index("master").ec_list.items()}
    by_label = {}
    for m, ls in labels.items():
        for lab in ls:
            by_label.setdefault(lab, set()).add(m)
    own = corpus.groupby("key").master.apply(set).to_dict()
    q["n_relevant"] = [len(by_label.get(lab, set()) - own.get(k, set()))
                       for lab, k in zip(q.label, q.key)]
    steps["queries_with_a_relevant_candidate"] = int((q.n_relevant > 0).sum())
    steps["queries_without_known_analogue"] = int((q.n_relevant == 0).sum())
    # development / final split by EC number (source group), stratified by EC class
    rng = np.random.default_rng(r["seed"])
    part = {}
    for cls, g in q.groupby(q.ec.str.split(".").str[0]):
        ecs = sorted(g.ec.unique())
        dev = set(rng.choice(ecs, max(1, int(round(len(ecs) * r["development_fraction"]))),
                             replace=False))
        part.update({e: "development" if e in dev else "final" for e in ecs})
    q["part"] = q.ec.map(part)
    q["stratum"] = "EC " + q.ec.str.split(".").str[0]
    q = q.reset_index(drop=True)
    q.insert(0, "query_id", ["q%06d" % i for i in range(len(q))])
    steps["development_queries"] = int((q.part == "development").sum())
    steps["final_queries"] = int((q.part == "final").sum())
    steps["corpus_masters"] = int(corpus.master.nunique())
    steps["corpus_fallback_raw_strings"] = int((corpus.policy_status != "ok").sum())
    steps["query_fallback_raw_strings"] = int((q.policy_status != "ok").sum())
    corpus.to_csv(out / "corpus.csv", index=False)
    q.drop(columns=["rxn_smiles"]).to_csv(out / "queries.csv", index=False)
    (out / "coverage.json").write_text(json.dumps(
        {"policy": policy, "steps": steps,
         "queries_by_source": q.source.value_counts().to_dict(),
         "queries_by_stratum": q.stratum.value_counts().sort_index().to_dict()}, indent=1))
    log.info("built %d queries and %d corpus masters: %s", len(q), steps["corpus_masters"], steps)


def featurize(r, cfg):
    out = out_dir() / "features"
    out.mkdir(exist_ok=True)
    q = pd.read_csv(out_dir() / "queries.csv")
    c = pd.read_csv(out_dir() / "corpus.csv")
    for fp in expand_fingerprints(cfg, r["fingerprints"]):
        path = out / f"{fp}.npz"
        if path.exists():
            continue
        fq = compute_in_memory(cfg, q.rxn.tolist(), fp)
        fc = compute_in_memory(cfg, c.rxn.tolist(), fp)
        np.savez_compressed(path, Q=fq.X, C=fc.X, n_bits=fq.n_bits, q_ok=fq.ok, c_ok=fc.ok)
        log.info("%s: queries valid %d/%d, corpus valid %d/%d", fp, fq.ok.sum(), len(q),
                 fc.ok.sum(), len(c))


def dense(X, n_bits):
    return (np.unpackbits(X, axis=1, count=int(n_bits)) if int(n_bits) else X).astype(np.float32)


def score(r, part):
    out = out_dir()
    q = pd.read_csv(out / "queries.csv")
    c = pd.read_csv(out / "corpus.csv")
    level, k = r["label_level"], r["k"]
    sel = np.flatnonzero((q.part == part) & (q.n_relevant > 0))
    masters, m_idx = np.unique(c.master.to_numpy(), return_inverse=True)
    m_key = c.groupby("master").key.first().reindex(masters).to_numpy()
    m_lab = [{ec_prefix(e, lv) for e in s.split(";")} - {None} for lv in (1, 2, level)
             for s in c.groupby("master").ec_list.first().reindex(masters)]
    n_m = len(masters)
    m_lab = {lv: m_lab[i * n_m:(i + 1) * n_m] for i, lv in enumerate((1, 2, level))}
    per = []
    for path in sorted((out / "features").glob("*.npz")):
        z = np.load(path)
        name = path.stem
        Q, C = dense(z["Q"][sel], z["n_bits"]), dense(z["C"], z["n_bits"])
        C[~z["c_ok"]] = 0
        tani = bool(int(z["n_bits"]))
        # similarity to every directed corpus row, then max over the two directions
        S = np.full((len(sel), n_m), -1.0, dtype=np.float32)
        for s in range(0, len(sel), 256):
            if tani:
                inter = Q[s:s + 256] @ C.T
                union = Q[s:s + 256].sum(1)[:, None] + C.sum(1)[None, :] - inter
                sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            else:
                qn = Q[s:s + 256] / np.maximum(np.linalg.norm(Q[s:s + 256], axis=1, keepdims=True), 1e-12)
                cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
                sim = qn @ cn.T
            np.maximum.at(S[s:s + 256].T, m_idx, sim.T)
        for row, qi in enumerate(sel):
            rec = {"query_id": q.query_id[qi], "fingerprint": name, "stratum": q.stratum[qi],
                   "ec": q.ec[qi], "source": q.source[qi], "encoded": bool(z["q_ok"][qi])}
            if not rec["encoded"]:
                rec.update({f"p_at_{k}_ec{lv}": 0.0 for lv in (1, 2, level)}, wrong_class_rate=1.0)
                per.append(rec)
                continue
            s_row = S[row].copy()
            s_row[m_key == q.key[qi]] = -2.0            # the query's own reaction
            order = np.lexsort((np.arange(n_m), -s_row))[:k]   # stable ties by master id
            for lv in (1, 2, level):
                lab = ec_prefix(q.ec[qi], lv)
                rec[f"p_at_{k}_ec{lv}"] = float(np.mean([lab in m_lab[lv][j] for j in order]))
            rec["wrong_class_rate"] = 1.0 - rec[f"p_at_{k}_ec1"]
            per.append(rec)
        log.info("%s scored on %d %s queries", name, len(sel), part)
    per = pd.DataFrame(per)
    per.to_csv(out / f"per_query__{part}.csv", index=False)

    metric = f"p_at_{k}_ec{level}"
    wide = per.pivot(index="query_id", columns="fingerprint", values=metric)
    meta = per.drop_duplicates("query_id").set_index("query_id").loc[wide.index]

    def macro(w, strata):
        return w.groupby(strata).mean().mean()      # strata weighted equally

    summ = pd.DataFrame({
        "macro_p_at_k": macro(wide, meta.stratum.to_numpy()),
        "query_mean_p_at_k": wide.mean(),
        "encoded_fraction": per.groupby("fingerprint").encoded.mean(),
        "wrong_class_rate": per.groupby("fingerprint").wrong_class_rate.mean(),
        f"p_at_{k}_ec1": per.groupby("fingerprint")[f"p_at_{k}_ec1"].mean(),
        f"p_at_{k}_ec2": per.groupby("fingerprint")[f"p_at_{k}_ec2"].mean()})
    # paired cluster bootstrap over EC numbers (the independent source groups)
    syn = next((f for f in wide.columns if f.startswith("synrfp-")), None)
    rng = np.random.default_rng(r["seed"])
    groups = meta.ec.to_numpy()
    uniq = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in uniq}
    strata = meta.stratum.to_numpy()
    boot = {f: [] for f in wide.columns if f != syn}
    W = wide.to_numpy()
    cols = list(wide.columns)
    for _ in range(r["bootstrap"] if syn else 0):
        idx = np.concatenate([rows_of[g] for g in rng.choice(uniq, len(uniq))])
        m = pd.DataFrame(W[idx], columns=cols).groupby(strata[idx]).mean().mean()
        for f in boot:
            boot[f].append(m[syn] - m[f])
    for f, v in boot.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        summ.loc[f, "synrfp_minus_this"] = summ.loc[syn, "macro_p_at_k"] - summ.loc[f, "macro_p_at_k"]
        summ.loc[f, "ci_low"], summ.loc[f, "ci_high"] = lo, hi
    summ.to_csv(out / f"summary__{part}.csv")
    per.groupby(["fingerprint", "stratum"])[metric].agg(["mean", "size"]).to_csv(
        out / f"strata__{part}.csv")
    log.info("\n%s", summ.round(4).to_string())
    L.write_manifest(f"03_retrieval_proxy_{part}",
                     [out / f"per_query__{part}.csv", out / f"summary__{part}.csv",
                      out / f"strata__{part}.csv", out / "coverage.json"],
                     {"sources": json.loads((EXT / "SOURCES.json").read_text()),
                      "n_queries": int(len(sel)), "n_groups": int(len(uniq))})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["fetch", "build", "featurize", "score"])
    ap.add_argument("--part", default="development", choices=["development", "final"])
    args = ap.parse_args()
    setup_logging()
    st = L.study_settings()
    r, policy = st["retrieval"], st["primary_policy"]
    if args.stage == "fetch":
        fetch(r)
    elif args.stage == "build":
        build(r, policy)
    elif args.stage == "featurize":
        featurize(r, L.load_policy(policy))
    else:
        score(r, args.part)


if __name__ == "__main__":
    main()
