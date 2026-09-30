#!/usr/bin/env python
"""E9. Aggregate all experiments into tables, statistics and figures.

Classification (03) -- two tracks, never mixed in one table (exp.md §3):
  Track A (scheme == cv, fingerprint comparison, statistics)
    benchmark_long.csv                 seed mean/SD per dataset x fingerprint x model
    track_a_<model>.csv / .tex         datasets x fingerprints (mean +/- SD, marks vs SynRFP)
    anova_<model>.csv                  per-dataset one-way repeated-measures ANOVA (subject=seed)
    tukey_<model>.csv                  per-dataset paired seed differences with
                                       Holm-adjusted tests (legacy filename)
    sign_<model>.pdf                   extended sign plot: effect size + practical significance
    ci_<model>.pdf                     pairwise-difference CI plot (SI)
    stats_<model>.csv                  Friedman test, mean ranks, Holm-corrected Wilcoxon (SI summary)
    cd_<model>.pdf                     critical-difference diagram (Nemenyi, SI summary)
  Track B (scheme == fixed, reference comparison, single run / init seeds only)
    track_b_<model>.csv / .tex         our rows, plain mean +/- SD, no significance test
    track_b_reference_syncat.csv       published SynCat/RXNFP/DRFP numbers (exp.md §5, VERIFY)
    track_b_reference_claire.csv       published CLAIRE/Theia numbers (exp.md §5, VERIFY)

Clustering (05)
  clustering_long.csv, kmeans.csv
  bitbirch_frozen.csv        threshold chosen by AMI on the calibration dataset, then applied
                             unchanged to every dataset; calibration_leak=True flags the
                             calibration dataset's own row (label-tuned, exp.md §3 "Bug to fix")
  bitbirch_oracle.csv        best threshold per dataset (upper bound, label-tuned)
  granularity_<dataset>.pdf  AMI against each hierarchy level vs threshold
Retrieval (06): retrieval.csv

The Ash et al. 2025 (JCIM, DOI 10.1021/acs.jcim.5c01609) protocol is
implemented as: per-dataset one-way repeated-measures ANOVA (subject = CV
seed) as the omnibus test, and paired per-seed t intervals with Holm-adjusted
tests for the mean MCC difference against `synrfp_default`. This has not been checked against the
paper's full text (not accessible here) -- VERIFY before results enter the
paper (exp.md §6).
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
from statsmodels.stats.anova import AnovaRM  # noqa: E402

from synrfp_exp.config import (default_config, expand_fingerprints, iter_json,  # noqa: E402
                               load_config, log, results_dir, setup_logging)

# A non-significant or practically small interval does not establish
# equivalence.  Keep the mark compact in tables, but describe it accurately.
MARK_LABEL = {"+": "baseline better", "-": "SynRFP better", "=": "inconclusive"}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def plain_latex_table(wide_mean, wide_sd, path, marks=None):
    """Mean +/- SD per cell; ``marks`` (dataset, fingerprint) -> '+'/'='/'-' vs the
    reference column, if given. No bold/underline "best" formatting (exp.md §3/§7)."""
    cols = list(wide_mean.columns)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Dataset & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\",
             "\\midrule"]
    for ds, row in wide_mean.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("--")
                continue
            txt = f"{v:.3f} $\\pm$ {wide_sd.loc[ds, c]:.3f}"
            if marks is not None and (ds, c) in marks.index:
                txt += f" ({marks.loc[(ds, c)]})"
            cells.append(txt)
        lines.append(ds.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def track_a_stats(sub, ref, threshold, conf):
    """
    Per-dataset RM-ANOVA (subject=seed, within=fingerprint) and paired
    per-seed differences against ``ref``. P-values are Holm-adjusted within
    each dataset; confidence intervals are paired t intervals. The output
    keeps the historical ``tukey_*.csv`` filename for compatibility.

    :param sub: Long ``dataset, seed, fingerprint, mcc`` rows (scheme "cv").
    :return: ``(anova_rows, ci_rows)`` DataFrames.
    """
    anova_rows, ci_rows = [], []
    for dataset, d in sub.groupby("dataset"):
        wide = d.pivot_table(index="seed", columns="fingerprint", values="mcc").dropna()
        if wide.shape[1] < 2 or ref not in wide.columns or len(wide) < 3:
            continue
        long = wide.reset_index().melt(id_vars="seed", var_name="fingerprint", value_name="mcc")
        try:
            rm = AnovaRM(long, depvar="mcc", subject="seed", within=["fingerprint"]).fit()
            a = rm.anova_table.reset_index().iloc[0]
            anova_rows.append({"dataset": dataset, "f_value": float(a["F Value"]),
                               "df_num": float(a["Num DF"]), "df_den": float(a["Den DF"]),
                               "p_value": float(a["Pr > F"])})
        except Exception as e:
            log.warning("RM-ANOVA %s skipped: %s", dataset, e)
        raw = []
        for other in [c for c in wide.columns if c != ref]:
            diff = (wide[ref] - wide[other]).dropna().to_numpy(dtype=float)
            n = len(diff)
            if n < 2 or not np.isfinite(diff).all():
                continue
            mean = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(n)
            if se == 0:
                p_value = 0.0 if mean != 0 else 1.0
                ci_lo = ci_hi = mean
            else:
                tcrit = float(stats.t.ppf((1.0 + conf) / 2.0, n - 1))
                ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
                p_value = float(2.0 * stats.t.sf(abs(mean / se), n - 1))
            raw.append({"dataset": dataset, "fingerprint": other,
                        "mean_diff_ref_minus_fp": mean, "ci_low": ci_lo,
                        "ci_high": ci_hi, "p_value": p_value, "n": n})
        if not raw:
            continue
        for r, p_holm in zip(raw, holm(np.asarray([x["p_value"] for x in raw]))):
            r["p_holm"] = float(p_holm)
            r["reject"] = bool(p_holm <= 1.0 - conf)
            if r["reject"] and r["ci_low"] > threshold:
                r["mark"] = "-"   # SynRFP practically better than `other`
            elif r["reject"] and r["ci_high"] < -threshold:
                r["mark"] = "+"   # `other` practically better than SynRFP
            else:
                r["mark"] = "="
            r["method"] = "paired_t_holm"
            ci_rows.append(r)
    return pd.DataFrame(anova_rows), pd.DataFrame(ci_rows)


def sign_plot(ci, ref, model, path):
    """Extended sign plot: effect size (color) + practical-significance mark (text)."""
    if ci.empty:
        return
    eff = ci.pivot(index="dataset", columns="fingerprint", values="mean_diff_ref_minus_fp")
    mk = ci.pivot(index="dataset", columns="fingerprint", values="mark")
    fig, ax = plt.subplots(figsize=(1.4 * eff.shape[1] + 1.5, 0.45 * eff.shape[0] + 1))
    lim = max(0.01, np.nanmax(np.abs(eff.to_numpy())))
    im = ax.imshow(eff.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            v, m = eff.iloc[i, j], mk.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, m, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(eff.shape[1]), eff.columns, rotation=45, ha="right")
    ax.set_yticks(range(eff.shape[0]), eff.index)
    ax.set_title(f"{model}: {ref} - baseline (+/= /- vs practical threshold)")
    fig.colorbar(im, ax=ax, label="mean MCC diff (SynRFP - baseline)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ci_plot(ci, ref, model, path):
    """Pairwise-difference CI plot, one row per dataset (SI)."""
    if ci.empty:
        return
    datasets = sorted(ci["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(5, 1.4 * len(datasets)), sharex=True)
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        d = ci[ci.dataset == dataset].sort_values("fingerprint")
        y = np.arange(len(d))
        err = np.vstack([d["mean_diff_ref_minus_fp"] - d["ci_low"],
                         d["ci_high"] - d["mean_diff_ref_minus_fp"]])
        ax.errorbar(d["mean_diff_ref_minus_fp"], y, xerr=err, fmt="o", ms=3)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(y, d["fingerprint"])
        ax.set_ylabel(dataset, fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel(f"mean MCC diff ({ref} - baseline), 95% paired t interval")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report_track_a(cfg, runs, out, metric, threshold, conf):
    cv = runs[runs.scheme == "cv"]
    if cv.empty:
        log.warning("no Track A (cv) runs")
        return
    ref = default_config(cfg).name
    for model, sub in cv.groupby("model"):
        agg = sub.groupby(["dataset", "fingerprint"])[metric].agg(["mean", "std", "count"])
        wm = agg["mean"].unstack("fingerprint")
        ws = agg["std"].unstack("fingerprint")

        anova, ci = track_a_stats(
            sub[["dataset", "seed", "fingerprint", metric]].rename(columns={metric: "mcc"}),
            ref, threshold, conf)
        anova.to_csv(out / f"anova_{model}.csv", index=False)
        ci.to_csv(out / f"tukey_{model}.csv", index=False)
        marks = ci.set_index(["dataset", "fingerprint"])["mark"] if not ci.empty else None

        wm.to_csv(out / f"track_a_{model}_mean.csv")
        plain_latex_table(wm, ws, out / f"track_a_{model}.tex", marks=marks)
        sign_plot(ci, ref, model, out / f"sign_{model}.pdf")
        ci_plot(ci, ref, model, out / f"ci_{model}.pdf")

        # Friedman + Nemenyi CD + Holm-Wilcoxon: across-dataset summary (SI)
        full = wm.dropna(axis=1)
        if full.shape[1] < 2 or full.shape[0] < 3:
            continue
        ranks = full.rank(axis=1, ascending=False).mean()
        rec = {"model": model, "n_datasets": full.shape[0]}
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
                  .assign(**rec)]).to_csv(out / f"stats_{model}.csv", index=False)
        try:
            import scikit_posthocs as sp

            if full.shape[1] >= 3:
                nem = sp.posthoc_nemenyi_friedman(full.to_numpy())
                nem.index = nem.columns = full.columns
                fig, ax = plt.subplots(figsize=(6, 1.8))
                sp.critical_difference_diagram(ranks, nem, ax=ax)
                ax.set_title(model)
                fig.savefig(out / f"cd_{model}.pdf", bbox_inches="tight")
                plt.close(fig)
        except Exception as e:  # plotting is optional
            log.warning("CD diagram %s skipped: %s", model, e)


# Literal numbers from exp.md §5 -- **VERIFY** against the published sources
# before they enter the paper (SynCat SI not accessible from this checkout).
SYNCAT_REFERENCE = pd.DataFrame([
    {"model": "syncat_acc", "schneider_u": 0.981, "schneider_b": 0.973, "tpl_u": 0.991,
     "tpl_b": 0.988, "uspto50k_u": 0.995, "uspto50k_b": 0.996, "syntemp_r0": 0.996,
     "syntemp_r1": 0.992, "syntemp_r2": 0.982},
    {"model": "syncat_mcc", "schneider_u": 0.980, "schneider_b": 0.973, "tpl_u": 0.991,
     "tpl_b": 0.988, "uspto50k_u": 0.995, "uspto50k_b": 0.995, "syntemp_r0": 0.996,
     "syntemp_r1": 0.992, "syntemp_r2": 0.978},
    {"model": "rxnfp_mlp_mcc_their_run", "schneider_u": 0.985, "schneider_b": 0.956,
     "tpl_u": 0.988, "tpl_b": 0.971, "uspto50k_u": 0.993, "uspto50k_b": 0.989,
     "syntemp_r0": 0.994, "syntemp_r1": 0.954, "syntemp_r2": 0.758},
    {"model": "drfp_mlp_mcc_their_run", "schneider_u": 0.956, "schneider_b": 0.925,
     "tpl_u": 0.977, "tpl_b": 0.962, "uspto50k_u": 0.898, "uspto50k_b": 0.910,
     "syntemp_r0": 0.971, "syntemp_r1": 0.958, "syntemp_r2": 0.957},
]).set_index("model")

CLAIRE_REFERENCE = pd.DataFrame([
    {"model": "claire_weighted_f1", "ec1": 0.958, "ec2": 0.944, "ec3": 0.861},
    {"model": "theia_ecreact_weighted_f1", "ec1": 0.387, "ec2": 0.333, "ec3": 0.228},
]).set_index("model")


def report_track_b(runs, out, metric):
    fixed = runs[runs.scheme == "fixed"]
    if fixed.empty:
        log.warning("no Track B (fixed) runs")
        return
    for model, sub in fixed.groupby("model"):
        agg = sub.groupby(["dataset", "fingerprint"])[metric].agg(["mean", "std", "count"])
        wm, ws = agg["mean"].unstack("fingerprint"), agg["std"].unstack("fingerprint")
        wm.to_csv(out / f"track_b_{model}_mean.csv")
        plain_latex_table(wm, ws, out / f"track_b_{model}.tex")
    SYNCAT_REFERENCE.to_csv(out / "track_b_reference_syncat.csv")
    CLAIRE_REFERENCE.to_csv(out / "track_b_reference_claire.csv")


def report_classification(cfg, out, metric):
    runs = pd.DataFrame(iter_json(results_dir("03_classification") / "benchmark"))
    if not runs.empty:
        runs = runs[runs["model"].isin(cfg["benchmark"]["models"])]
    if runs.empty:
        log.warning("no benchmark runs")
        return
    runs.groupby(["dataset", "scheme", "fingerprint", "model"])[metric] \
        .agg(["mean", "std", "count"]).reset_index().to_csv(out / "benchmark_long.csv", index=False)
    cc = cfg.get("statistics", {})
    report_track_a(cfg, runs, out, metric, cc.get("practical_threshold", 0.01),
                   cc.get("confidence_level", 0.95))
    report_track_b(runs, out, metric)


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
    configured = cfg["clustering"]["bitbirch"].get("frozen_thresholds", {})
    frozen_thresholds = {}
    for name, threshold in configured.items():
        expanded = expand_fingerprints(cfg, [name])
        if len(expanded) != 1:
            raise ValueError(f"frozen BitBIRCH fingerprint must expand once: {name}")
        frozen_thresholds[expanded[0]] = float(threshold)
    frozen = []
    for fp, sub in bb.groupby("fingerprint"):
        c = sub[sub.dataset == cal]
        if c.empty:
            continue
        t = frozen_thresholds.get(fp)
        if t is None:
            t = float(c.loc[c["ami"].idxmax(), "param"])
            log.warning("no frozen BitBIRCH threshold for %s; selecting on %s", fp, cal)
        frozen.append(sub[np.isclose(sub.param, t)].assign(threshold=t))
    if frozen:
        out_df = pd.concat(frozen)
        # bitbirch is calibration-dataset-agnostic: the calibration dataset's
        # own reported row picked its threshold on itself (AMI-selected) and
        # is therefore label-tuned, unlike every other row here (exp.md §3,
        # "Bug to fix: the BitBIRCH threshold is chosen by AMI on uspto50k_b").
        out_df["calibration_leak"] = out_df["dataset"] == cal
        if out_df["calibration_leak"].any():
            log.warning("bitbirch_frozen.csv: the %s row is threshold-calibrated on "
                        "itself (label-tuned) -- marked calibration_leak=True", cal)
        cols = ["dataset", "fingerprint", "threshold", "n_clusters", "ari", "ami", "nmi",
                "mcc_hungarian", "calibration_leak"]
        out_df[cols].to_csv(out / "bitbirch_frozen.csv", index=False)
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
