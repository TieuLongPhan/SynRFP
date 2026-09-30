#!/usr/bin/env python
"""E4. Factorial analysis of the SynRFP design space.

Scores are averaged over replicate seeds per (dataset, configuration, model).
A fixed-effects ANOVA with dataset as a blocking factor estimates main effects
of model, tokenizer, radius, bits, sketch and mode, and all their pairwise
interactions; effect sizes are partial eta-squared (type II sums of squares).
Radius and bits are crossed in this design, so their effects are separable.

Outputs (04_factorial/results/):
  marginals.csv               factor-level mean/SD of MCC
  anova.csv                   full ANOVA table with partial eta^2
  representation_effects.csv  input for the paper's plotting script
  per_model_anova.csv         the same ANOVA fitted within each model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse
import itertools

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from synrfp_exp.config import (expand_fingerprints, iter_json, load_config, log,
                               setup_logging, results_dir)

FACTORS = ["model", "tokenizer", "radius", "bits", "sketch", "mode"]
PRETTY = {"model": "Model", "tokenizer": "Tokenizer", "radius": "Radius", "bits": "Bits",
          "sketch": "Sketch", "mode": "Mode"}


def validate_complete(runs, cfg, metric):
    """Require each planned model/configuration/split cell before final ANOVA."""
    spec = cfg["factorial"]
    fingerprints = expand_fingerprints(cfg, ["synrfp_grid"])
    keys = ["dataset", "model", "fingerprint", "seed"]
    required = set(keys + [metric, "scheme"])
    if not required.issubset(runs.columns):
        raise ValueError(f"missing result fields: {sorted(required - set(runs.columns))}")
    expected = {
        (dataset, model, fp, seed)
        for dataset in spec["datasets"]
        for model in spec["models"]
        for fp in fingerprints
        for seed in spec.get("seeds_per_model", {}).get(model, spec["seeds"])
    }
    observed = set(runs[keys].itertuples(index=False, name=None))
    missing, unexpected = expected - observed, observed - expected
    duplicates = int(runs.duplicated(keys).sum())
    if missing or unexpected or duplicates:
        raise ValueError(
            f"factorial matrix incomplete or inconsistent: {len(observed)}/{len(expected)} "
            f"unique cells; missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={duplicates}; missing examples={sorted(missing)[:3]}"
        )
    if not runs["scheme"].eq("random").all() or not np.isfinite(runs[metric].to_numpy(dtype=float)).all():
        raise ValueError("factorial results require random splits and finite metric values")


def anova(df, factors, metric):
    terms = [f"C({f})" for f in factors]
    inter = [f"C({a}):C({b})" for a, b in itertools.combinations(factors, 2)]
    formula = f"{metric} ~ C(dataset) + " + " + ".join(terms + inter)
    fit = smf.ols(formula, data=df).fit()
    tab = anova_lm(fit, typ=2)
    ss_res = tab.loc["Residual", "sum_sq"]
    tab["partial_eta_sq"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    tab.loc["Residual", "partial_eta_sq"] = float("nan")
    return tab.reset_index().rename(columns={"index": "term"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--metric", default="mcc")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)  # sets the results root
    runs = pd.DataFrame(iter_json(results_dir("03_classification") / "factorial"))
    if not runs.empty:
        runs = runs[runs["model"].isin(cfg["factorial"]["models"])]
    if runs.empty:
        raise SystemExit("no factorial runs found; run 03_classify.py --study factorial")
    m = args.metric
    try:
        validate_complete(runs, cfg, m)
    except ValueError as exc:
        raise SystemExit(f"ANOVA withheld: {exc}") from exc
    df = runs.groupby(["dataset"] + FACTORS, as_index=False)[m].mean()
    out = results_dir("04_factorial")
    out.mkdir(parents=True, exist_ok=True)
    log.info("%d configuration-level cells over %d datasets", len(df), df.dataset.nunique())

    marg = pd.concat([
        df.groupby(f)[m].agg(["mean", "std", "count"]).reset_index()
          .rename(columns={f: "level"}).assign(factor=f)
        for f in FACTORS])
    marg.to_csv(out / "marginals.csv", index=False)

    tab = anova(df, FACTORS, m)
    tab.to_csv(out / "anova.csv", index=False)

    per = []
    for model, sub in df.groupby("model"):
        per.append(anova(sub, FACTORS[1:], m).assign(model=model))
    pd.concat(per).to_csv(out / "per_model_anova.csv", index=False)

    rows = [{"kind": "main", "factor": PRETTY[r.factor], "level": r.level,
             "mean": round(r["mean"], 3), "sd": round(r["std"], 3), "effect": ""}
            for _, r in marg.iterrows()]
    for _, r in tab.iterrows():
        if r.term.count(":") == 1:
            a, b = (t[2:-1] for t in r.term.split(":"))
            rows.append({"kind": "interaction", "factor": "Interaction",
                         "level": f"{PRETTY[a]} × {PRETTY[b].lower()}", "mean": "",
                         "sd": "", "effect": round(r.partial_eta_sq, 3)})
    pd.DataFrame(rows).to_csv(out / "representation_effects.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
