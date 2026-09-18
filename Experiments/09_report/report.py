#!/usr/bin/env python
"""E9. Aggregate all experiments into tables, statistics and figures.

Classification (03)
  benchmark_long.csv         seed mean/SD per dataset x scheme x fingerprint x model
  benchmark_<scheme>_<model>.csv / .tex   datasets x fingerprints (mean +/- SD)
  stats_<scheme>_<model>.csv Friedman test, mean ranks, Holm-corrected Wilcoxon
                             signed-rank tests of SynRFP against each baseline
  cd_<scheme>_<model>.pdf    critical-difference diagram (Nemenyi)
Clustering (05)
  clustering_long.csv, kmeans.csv
  bitbirch_frozen.csv        threshold chosen by AMI on the calibration dataset,
                             then applied unchanged to every dataset
  bitbirch_oracle.csv        best threshold per dataset (upper bound, label-tuned)
  granularity_<dataset>.pdf  AMI against each hierarchy level vs threshold
Retrieval (06): retrieval.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from synrfp_exp.config import (default_config, iter_json, load_config, log,  # noqa: E402
                               results_dir, setup_logging)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def latex_table(wide_mean, wide_sd, path, higher_better=True):
    cols = list(wide_mean.columns)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Dataset & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\",
             "\\midrule"]
    for ds, row in wide_mean.iterrows():
        vals = row.to_numpy(dtype=float)
        order = np.argsort(-vals if higher_better else vals)
        cells = []
        for j, c in enumerate(cols):
            if np.isnan(vals[j]):
                cells.append("--")
                continue
            txt = f"{vals[j]:.3f} $\\pm$ {wide_sd.loc[ds, c]:.3f}"
            if j == order[0]:
                txt = f"\\textbf{{{txt}}}"
            elif len(order) > 1 and j == order[1]:
                txt = f"\\underline{{{txt}}}"
            cells.append(txt)
        lines.append(ds.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def report_classification(cfg, out, metric):
    runs = pd.DataFrame(iter_json(results_dir("03_classification") / "benchmark"))
    if runs.empty:
        log.warning("no benchmark runs")
        return
    agg = (runs.groupby(["dataset", "scheme", "fingerprint", "model"])[metric]
           .agg(["mean", "std", "count"]).reset_index())
    agg.to_csv(out / "benchmark_long.csv", index=False)
    ref = default_config(cfg).name
    for (scheme, model), sub in agg.groupby(["scheme", "model"]):
        wm = sub.pivot(index="dataset", columns="fingerprint", values="mean")
        ws = sub.pivot(index="dataset", columns="fingerprint", values="std")
        wm.loc["mean"] = wm.mean()
        ws.loc["mean"] = wm.drop(index="mean").std(ddof=0)
        wm.to_csv(out / f"benchmark_{scheme}_{model}.csv")
        latex_table(wm, ws, out / f"benchmark_{scheme}_{model}.tex")

        full = wm.drop(index="mean").dropna(axis=1)
        if full.shape[1] < 2 or full.shape[0] < 3:
            continue
        ranks = full.rank(axis=1, ascending=False).mean()
        rec = {"scheme": scheme, "model": model, "n_datasets": full.shape[0]}
        if full.shape[1] >= 3:
            fr = stats.friedmanchisquare(*[full[c] for c in full.columns])
            rec.update(friedman_chi2=fr.statistic, friedman_p=fr.pvalue)
        rows = []
        if ref in full:
            others = [c for c in full.columns if c != ref]
            ps, diffs = [], []
            for c in others:
                d = full[ref] - full[c]
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                ps.append(p)
                diffs.append(d.mean())
            for c, dmean, p, pa in zip(others, diffs, ps, holm(ps)):
                rows.append({**rec, "comparison": f"{ref} vs {c}", "mean_diff": dmean,
                             "wilcoxon_p": p, "holm_p": pa})
        pd.concat([pd.DataFrame(rows), ranks.rename("mean_rank").reset_index()
                   .assign(**rec)]).to_csv(out / f"stats_{scheme}_{model}.csv", index=False)
        try:
            import scikit_posthocs as sp

            if full.shape[1] >= 3:
                nem = sp.posthoc_nemenyi_friedman(full.to_numpy())
                nem.index = nem.columns = full.columns
                fig, ax = plt.subplots(figsize=(6, 1.8))
                sp.critical_difference_diagram(ranks, nem, ax=ax)
                ax.set_title(f"{scheme}, {model}")
                fig.savefig(out / f"cd_{scheme}_{model}.pdf", bbox_inches="tight")
                plt.close(fig)
        except Exception as e:  # plotting is optional
            log.warning("CD diagram %s/%s skipped: %s", scheme, model, e)


def report_clustering(cfg, out):
    runs = pd.DataFrame(iter_json(results_dir("05_clustering")))
    if runs.empty:
        log.warning("no clustering runs")
        return
    runs.to_csv(out / "clustering_long.csv", index=False)
    label_of = {d: cfg["datasets"][d]["label"] for d in runs["dataset"].unique()}

    def main_metric(r, m):
        return r[f"{label_of[r['dataset']]}__{m}"]

    for m in ("ari", "ami", "nmi", "mcc_hungarian", "n_clusters"):
        runs[m] = runs.apply(main_metric, axis=1, m=m)
    km = runs[runs.method == "kmeans"]
    (km.groupby(["dataset", "fingerprint"])[["mcc_hungarian", "ari", "ami", "nmi"]]
       .agg(["mean", "std"]).to_csv(out / "kmeans.csv"))

    bb = runs[runs.method == "bitbirch"]
    if bb.empty:
        return
    cal = cfg["clustering"]["calibration_dataset"]
    frozen = []
    for fp, sub in bb.groupby("fingerprint"):
        c = sub[sub.dataset == cal]
        if c.empty:
            continue
        t = float(c.loc[c["ami"].idxmax(), "param"])
        frozen.append(sub[np.isclose(sub.param, t)].assign(threshold=t))
    if frozen:
        pd.concat(frozen)[["dataset", "fingerprint", "threshold", "n_clusters", "ari",
                           "ami", "nmi", "mcc_hungarian"]].to_csv(out / "bitbirch_frozen.csv",
                                                                  index=False)
    idx = bb.groupby(["dataset", "fingerprint"])["ami"].idxmax()
    bb.loc[idx, ["dataset", "fingerprint", "param", "n_clusters", "ari", "ami", "nmi"]] \
        .to_csv(out / "bitbirch_oracle.csv", index=False)

    for dataset, sub in bb.groupby("dataset"):
        levels = cfg["datasets"][dataset].get("hierarchy")
        if not levels:
            continue
        fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 2.6), sharey=True)
        for ax, lev in zip(np.atleast_1d(axes), levels):
            for fp, s in sub.groupby("fingerprint"):
                s = s.sort_values("param")
                ax.plot(s["param"], s[f"{lev}__ami"], marker="o", ms=3, label=fp)
            ax.set_title(lev)
            ax.set_xlabel("BitBIRCH threshold")
        np.atleast_1d(axes)[0].set_ylabel("AMI")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"granularity_{dataset}.pdf")
        plt.close(fig)


def report_retrieval(out):
    runs = pd.DataFrame(iter_json(results_dir("06_retrieval")))
    if not runs.empty:
        runs.to_csv(out / "retrieval.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--metric", default="mcc")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    out = results_dir("09_report")
    report_classification(cfg, out, args.metric)
    report_clustering(cfg, out)
    report_retrieval(out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
