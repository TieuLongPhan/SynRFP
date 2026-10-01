#!/usr/bin/env python
"""Step 6. Aggregate every finished step and evaluate the frozen gates.

Reads only files under runs/. Missing steps are reported as missing, never
filled in. Classification replicates are the 25 grouped-CV splits; they share
training data, so their SD describes split variation and no p-value is
computed from them. Paired differences are always taken on the same split.

Outputs under runs/report/:
  inventory.csv                 expected and found classification cells
  classification_runs.csv       one row per run
  classification_means.csv      mean, SD, n per policy/model/endpoint/fingerprint
  classification_paired.csv     SynRFP minus each fingerprint per endpoint
  classification_family.csv     family-weighted mean and leave-one-family-out range
  policy_shift.csv              n1 (or n2) minus raw per fingerprint and endpoint
  attribution_contrasts.csv     predeclared paired contrasts
  gates.json, SUMMARY.md        gate results with the numbers they rest on
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import vlib as L
from synrfp_exp.config import default_config, load_config


def load_runs(policies):
    rows = []
    for pol in policies:
        root = L.RUNS / pol / "03_classification" / "results" / "benchmark"
        for p in sorted(root.glob("*/cv_s*/*.json")):
            d = json.loads(p.read_text())
            if "mcc" in d:
                rows.append({"policy": pol, **{k: d.get(k) for k in (
                    "dataset", "seed", "fingerprint", "model", "mcc", "accuracy", "macro_f1",
                    "n_train", "n_test", "n_unfeaturised_test")}})
    return pd.DataFrame(rows)


def classification(st, out, summary, gates):
    policies = [p for p in st["policies"] if L.policy_config_path(p).exists()]
    if not policies:
        summary.append("## Classification\n\nNo policy configs found; run 00_setup.py.")
        return
    cfg = load_config(L.policy_config_path(policies[0]))
    syn = default_config(cfg).name
    runs = load_runs(policies)
    runs.to_csv(out / "classification_runs.csv", index=False)
    fps = [syn if f == "synrfp_default" else f for f in st["benchmark"]["fingerprints"]]
    seeds = st["benchmark"]["schemes"]["cv"]
    exp = pd.MultiIndex.from_product(
        [policies, list(cfg["datasets"]), fps, st["benchmark"]["models"]],
        names=["policy", "dataset", "fingerprint", "model"]).to_frame(index=False)
    found = (runs.groupby(["policy", "dataset", "fingerprint", "model"]).size()
             .rename("found").reset_index() if len(runs) else
             pd.DataFrame(columns=["policy", "dataset", "fingerprint", "model", "found"]))
    inv = exp.merge(found, how="left").fillna({"found": 0})
    inv["expected"] = len(seeds)
    inv.to_csv(out / "inventory.csv", index=False)
    done = inv.groupby(["policy", "model"]).apply(
        lambda d: f"{int(d.found.sum())}/{int(d.expected.sum())}")
    summary.append("## Classification inventory (runs found / expected)\n\n"
                   + "\n".join(f"- {p} / {m}: {v}" for (p, m), v in done.items()))
    if not len(runs):
        return
    means = runs.groupby(["policy", "model", "dataset", "fingerprint"]).mcc.agg(
        ["mean", "std", "count"]).reset_index()
    means.to_csv(out / "classification_means.csv", index=False)

    fam_of = {d: f for f, ds in st["families"].items() for d in ds}
    wide = runs.pivot_table(index=["policy", "model", "dataset", "seed"], columns="fingerprint",
                            values="mcc")
    paired, family = [], []
    for (pol, model), w in wide.groupby(level=[0, 1]):
        if syn not in w:
            continue
        for fp in w.columns.drop(syn):
            diff = (w[syn] - w[fp]).dropna()
            if diff.empty:
                continue
            per_ds = diff.groupby(level="dataset").agg(["mean", "std", "count"])
            per_ds["splits_synrfp_ahead"] = (diff > 0).groupby(level="dataset").sum()
            for ds, r in per_ds.iterrows():
                paired.append({"policy": pol, "model": model, "fingerprint": fp, "dataset": ds,
                               "synrfp_minus_fp": r["mean"], "sd_over_splits": r["std"],
                               "n_splits": int(r["count"]),
                               "splits_synrfp_ahead": int(r["splits_synrfp_ahead"])})
            fam = per_ds["mean"].groupby(per_ds.index.map(fam_of)).mean()
            loo = [fam.drop(f).mean() for f in fam.index] if len(fam) > 1 else [np.nan]
            family.append({"policy": pol, "model": model, "fingerprint": fp,
                           "n_endpoints": len(per_ds), "n_families": len(fam),
                           "endpoint_mean": per_ds["mean"].mean(), "family_mean": fam.mean(),
                           "leave_one_family_out_min": float(np.min(loo)),
                           "leave_one_family_out_max": float(np.max(loo)),
                           "endpoints_synrfp_ahead": int((per_ds["mean"] > 0).sum())})
    paired, family = pd.DataFrame(paired), pd.DataFrame(family)
    paired.to_csv(out / "classification_paired.csv", index=False)
    family.to_csv(out / "classification_family.csv", index=False)

    shift = []
    if "raw" in policies:
        for pol in (p for p in policies if p != "raw"):
            a = runs[runs.policy.eq(pol)].set_index(["model", "dataset", "fingerprint", "seed"]).mcc
            b = runs[runs.policy.eq("raw")].set_index(["model", "dataset", "fingerprint", "seed"]).mcc
            d = (a - b).dropna()
            if len(d):
                g = d.groupby(level=["model", "dataset", "fingerprint"]).agg(["mean", "std", "count"])
                g.insert(0, "policy", pol)
                shift.append(g.reset_index())
    if shift:
        pd.concat(shift).to_csv(out / "policy_shift.csv", index=False)

    g = st["gates"]["A_common_input"]
    pol = st["primary_policy"]
    sel = family[(family.policy == pol) & (family.fingerprint == g["reference"])].set_index("model")
    full = bool(len(inv[(inv.policy == pol) & inv.model.isin(["knn", "mlp"])
                        & inv.fingerprint.isin([syn, g["reference"]])].query("found < expected")) == 0)
    res = {"policy": pol, "complete": full}
    if {"knn", "mlp"} <= set(sel.index):
        res.update(knn_family_mean=float(sel.loc["knn", "family_mean"]),
                   knn_leave_one_family_out_min=float(sel.loc["knn", "leave_one_family_out_min"]),
                   mlp_family_mean=float(sel.loc["mlp", "family_mean"]))
        res["pass"] = bool(full and res["knn_family_mean"] >= g["knn_family_mean_min"]
                           and res["knn_leave_one_family_out_min"] > g["knn_leave_one_family_out_min"]
                           and res["mlp_family_mean"] >= g["mlp_family_mean_min"])
    else:
        res["pass"] = None
    gates["A_common_input"] = {"thresholds": g, **res}
    summary.append("## SynRFP minus each fingerprint (family-weighted mean MCC)\n\n"
                   + family.round(4).to_markdown(index=False))


CONTRASTS = {
    "signed_minus_unsigned_set": ("set_signed", "set_unsigned"),
    "signed_minus_unsigned_multiset": ("multiset_signed", "multiset_unsigned"),
    "multiset_minus_set_unsigned": ("multiset_unsigned", "set_unsigned"),
    "multiset_minus_set_signed": ("multiset_signed", "set_signed"),
    "change_minus_pooled": ("multiset_signed", "pooled"),
    "change_minus_pooled_side_tagged": ("multiset_signed", "pooled_side_tagged"),
}


def attribution(st, out, summary, gates):
    a = st["attribution"]
    root = L.RUNS / "attribution" / st["primary_policy"]
    files = [p for p in sorted(root.glob("*__*.csv")) if not p.name.startswith("counter")]
    if not files:
        summary.append("## Attribution\n\nNot run yet.")
        gates["B_attribution"] = {"pass": None}
        return
    runs = pd.concat(pd.read_csv(p) for p in files)
    m = a["mlp"]
    rows = []
    for (vocab, ds, model), d in runs.groupby(["vocabulary", "dataset", "model"]):
        prim = d[(d.bits == m["bits"]) & (d.fold == "parity")]
        # average hash seeds within a split, then pair payloads on the split
        cell = prim.groupby(["payload", "seed"]).mcc.mean().unstack("payload")
        for name, (x, y) in CONTRASTS.items():
            if x in cell and y in cell:
                diff = (cell[x] - cell[y]).dropna()
                rows.append({"vocabulary": vocab, "dataset": ds, "model": model, "contrast": name,
                             "mean": diff.mean(), "sd_over_splits": diff.std(), "n_splits": len(diff)})
        if model == "knn":
            for bits in a["bits"]:
                e = d[d.bits == bits].groupby(["payload", "fold", "seed"]).mcc.mean().unstack("fold")
                if {"parity", "or"} <= set(e.columns):
                    diff = (e["parity"] - e["or"]).xs("multiset_signed", level="payload")
                    rows.append({"vocabulary": vocab, "dataset": ds, "model": model,
                                 "contrast": f"parity_minus_or_default_b{bits}",
                                 "mean": diff.mean(), "sd_over_splits": diff.std(),
                                 "n_splits": len(diff)})
            ex = d[d.fold == "exact"].set_index(["payload", "seed"]).mcc
            if len(ex) and "multiset_signed" in cell:
                diff = (ex.xs("multiset_signed", level="payload") - cell["multiset_signed"]).dropna()
                rows.append({"vocabulary": vocab, "dataset": ds, "model": model,
                             "contrast": "exact_minus_folded_default", "mean": diff.mean(),
                             "sd_over_splits": diff.std(), "n_splits": len(diff)})
    con = pd.DataFrame(rows)
    con.to_csv(out / "attribution_contrasts.csv", index=False)
    lev = (runs[(runs.fold.isin(["parity", "exact"])) & (runs.bits.isin([m["bits"], 0]))]
           .groupby(["vocabulary", "model", "fold", "payload", "dataset"]).mcc.mean().unstack("dataset"))
    lev.to_csv(out / "attribution_levels.csv")
    ce = [pd.read_csv(p).assign(file=p.stem) for p in sorted(root.glob("counterexamples__*.csv"))]
    if ce:
        pd.concat(ce).to_csv(out / "attribution_counterexamples.csv", index=False)
    g = st["gates"]["B_attribution"]
    wl = con[(con.vocabulary == "wl") & con.contrast.isin(g["contrasts"])]
    res = {}
    ok = True
    for name in g["contrasts"]:
        for model, need in (("knn", g["min_endpoints_same_sign_knn"]),
                            ("mlp", g["min_endpoints_same_sign_mlp"])):
            v = wl[(wl.contrast == name) & (wl.model == model)]["mean"]
            pos = int((v > 0).sum())
            res[f"{name}__{model}"] = {"endpoints_positive": pos, "endpoints": int(len(v)),
                                       "mean": float(v.mean()) if len(v) else None}
            ok = ok and len(v) >= need and pos >= need
    complete = wl.dataset.nunique() == len(a["datasets"])
    gates["B_attribution"] = {"thresholds": g, "complete": bool(complete), "contrasts": res,
                              "pass": bool(ok and complete)}
    piv = con[con.vocabulary == "wl"].pivot_table(index=["model", "contrast"], columns="dataset",
                                                  values="mean")
    summary.append("## Attribution contrasts, WL vocabulary (paired mean MCC difference)\n\n"
                   + piv.round(4).to_markdown())


def retrieval(st, out, summary, gates):
    root = L.RUNS / "retrieval_proxy"
    g = st["gates"]["C_retrieval_proxy"]
    res = {"thresholds": g, "pass": None}
    for part in ("development", "final"):
        p = root / f"summary__{part}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, index_col=0)
        s.to_csv(out / f"retrieval_summary__{part}.csv")
        summary.append(f"## Retrieval proxy, {part} queries\n\n" + s.round(4).to_markdown())
        if part != "final":
            continue
        syn = next(i for i in s.index if i.startswith("synrfp-"))
        label_free = [i for i in s.index if i != syn and i != "rxnfp"]
        best = s.loc[label_free, "macro_p_at_k"].idxmax()
        margin = float(s.loc[syn, "macro_p_at_k"] - s.loc[best, "macro_p_at_k"])
        learned = [i for i in s.index if i.startswith("rxnfp")]
        res.update(synrfp=float(s.loc[syn, "macro_p_at_k"]), best_label_free=best,
                   margin=margin, ci_low=float(s.loc[best, "ci_low"]),
                   ci_high=float(s.loc[best, "ci_high"]),
                   learned_encoders_present=learned)
        a = margin >= g["margin_over_best_label_free_min"] and s.loc[best, "ci_low"] > 0
        b = bool(learned) and (s.loc[learned, "macro_p_at_k"].max() - s.loc[syn, "macro_p_at_k"]
                               <= g["or_within_of_best_learned"])
        res["pass"] = bool(a or b)
        res["passed_by"] = "margin" if a else ("parity with learned encoder" if b else None)
    if (root / "coverage.json").exists():
        res["coverage"] = json.loads((root / "coverage.json").read_text())["steps"]
    gates["C_retrieval_proxy"] = res


def simple(out, summary):
    for title, rel in (("Serialization identity", "input_audit/serialization.csv"),
                       ("Locality bounds", "locality/bounds_summary.csv"),
                       ("Collision load", "locality/collisions.csv"),
                       ("Encoding cost", "cost/encoding.csv"),
                       ("Encoding cost, RXNFP", "cost/encoding_rxnfp.csv"),
                       ("Search cost", "cost/search.csv"),
                       ("Input policy changes", "setup_policy_report.csv")):
        p = L.RUNS / rel
        if not p.exists():
            summary.append(f"## {title}\n\nNot run yet.")
            continue
        d = pd.read_csv(p)
        d.to_csv(out / p.name, index=False)
        summary.append(f"## {title}\n\n" + d.round(4).to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    st = L.study_settings()
    out = L.RUNS / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary, gates = ["# Validation suite report\n\nGenerated by 06_report.py. "
                      "Missing steps are listed as missing."], {}
    classification(st, out, summary, gates)
    attribution(st, out, summary, gates)
    retrieval(st, out, summary, gates)
    simple(out, summary)
    setup = L.RUNS / "setup.json"
    if setup.exists():
        s = json.loads(setup.read_text())
        summary.insert(1, "## Split origin\n\n" + "\n".join(
            f"- {k}: {v}" for k, v in s["splits_origin"].items()))
    (out / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    summary.insert(1, "## Gates\n\n```json\n" + json.dumps(
        {k: {x: v.get(x) for x in ("pass", "complete", "passed_by") if x in v}
         for k, v in gates.items()}, indent=1) + "\n```")
    (out / "SUMMARY.md").write_text("\n\n".join(summary) + "\n")
    L.write_manifest("06_report", sorted(out.glob("*")))
    print((out / "SUMMARY.md").read_text()[:3000])


if __name__ == "__main__":
    main()
