"""Descriptive explanatory summaries from completed experiments; no model fits.

Run from any directory. Error bars describe split/initialization variability,
not independent-population confidence intervals. The factorial is random
holdout classification; the clustering ablation is full-data KMeans.
"""
from pathlib import Path
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
DEFAULT = "synrfp-wl-r2-b1024-parity-delta"
LABELS = {
    "schneider_ub": "Schneider UB", "schneider_b": "Schneider B",
    "uspto50k_ub": "USPTO50K UB", "uspto50k_b": "USPTO50K B",
    "syntemp_r0": "SynTemp R0", "ecreact_1": "ECReact L1",
    "ecreact_3": "ECReact L3", "mech_1": "Mechanistic L1",
    "mech_2": "Mechanistic L2", "tpl_b": "TPL B",
}
GRID = {"tokenizer": ["wl", "morgan", "path"], "radius": [1, 2, 3],
        "bits": [512, 1024, 2048], "sketch": ["parity", "minhash", "cw"],
        "mode": ["delta", "union"]}


def records(pattern):
    return pd.DataFrame(json.loads(p.read_text()) for p in EXP.glob(pattern))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = records("03_classification/results/factorial/*/random_s*/*__knn.json")
    keys = ["dataset", "seed", *GRID]
    expected = set(itertools.product(LABELS, range(5), *GRID.values()))
    observed = set(runs[keys].itertuples(index=False, name=None))
    if observed != expected or runs.duplicated(keys).any() or runs.mcc.isna().any():
        raise ValueError("Expected the complete, unique 8,100-cell kNN factorial")

    marginals = []
    for factor in GRID:
        sub = runs.groupby(factor).mcc.agg(["mean", "count"]).reset_index()
        sub = sub.rename(columns={factor: "level"}).assign(factor=factor)
        marginals.append(sub)
    pd.concat(marginals).to_csv(OUT / "knn_factor_marginals.csv", index=False)
    for a, b in [("sketch", "mode"), ("tokenizer", "radius")]:
        runs.groupby([a, b]).mcc.mean().unstack().to_csv(OUT / f"knn_{a}_{b}.csv")

    paired = runs.pivot(index=[k for k in keys if k != "mode"],
                        columns="mode", values="mcc")
    paired["delta_minus_union"] = paired.delta - paired.union
    split_diffs = paired.delta_minus_union.groupby(["dataset", "seed"]).mean()
    split_diffs.to_csv(OUT / "knn_mode_contrast_by_split.csv")
    mode_summary = split_diffs.groupby("dataset").agg(["mean", "std", "count"])
    mode_summary.to_csv(OUT / "knn_mode_contrast_summary.csv")

    ablation = records("05_clustering/results_followup/ablation/*/*.json")
    primary = {"schneider_b": "label", "uspto50k_b": "label",
               "syntemp_r0": "label_0", "mech_2": "level2"}
    ablation["ami"] = [r[f"{primary[r.dataset]}__ami"] for _, r in ablation.iterrows()]
    local = ablation[(ablation.tokenizer == "wl") & (ablation.radius == 2)
                     & (ablation.bits == 1024) & (ablation.sketch == "parity")]
    paired_clust = local.pivot(index=["dataset", "seed"], columns="mode", values="ami")
    if len(paired_clust) != 12 or paired_clust.isna().any().any():
        raise ValueError("Expected 12 matched delta/union clustering pairs")
    paired_clust["delta_minus_union"] = paired_clust.delta - paired_clust.union
    paired_clust.to_csv(OUT / "clustering_mode_contrast_by_seed.csv")
    cluster_summary = paired_clust.delta_minus_union.groupby("dataset").agg(["mean", "std", "count"])
    cluster_summary.to_csv(OUT / "clustering_mode_contrast_summary.csv")
    ablation.groupby(["dataset", "fingerprint"]).ami.agg(["mean", "std", "count"]).to_csv(
        OUT / "clustering_ablation_ami.csv")

    # Default-context radius results expose heterogeneous effects across tasks.
    radius = runs[(runs.tokenizer == "wl") & (runs.bits == 1024)
                  & (runs.sketch == "parity") & (runs["mode"] == "delta")]
    radius.groupby(["dataset", "radius"]).mcc.agg(["mean", "std", "count"]).to_csv(
        OUT / "knn_default_context_radius.csv")

    metric_rows = []
    for pattern, phase in [("05_clustering/results/*/*/kmeans_*.json", "small_kmeans"),
                           ("05_clustering/results_followup/large_minibatch/*/*.json", "large_minibatch")]:
        for p in EXP.glob(pattern):
            r = json.loads(p.read_text())
            ds = r["dataset"]
            label = {"syntemp_r0": "label_0", "mech_1": "level1", "mech_2": "level2",
                     "ecreact_1": "ec1", "ecreact_2": "ec2", "ecreact_3": "ec3"}.get(ds, "label")
            metric_rows.append({"phase": phase, "dataset": ds, "fingerprint": r["fingerprint"],
                                "ami": r[label + "__ami"], "ari": r[label + "__ari"]})
    pd.DataFrame(metric_rows).groupby(["phase", "dataset", "fingerprint"])[["ami", "ari"]].agg(
        ["mean", "std", "count"]).to_csv(OUT / "clustering_ami_ari.csv")

    # Hierarchy profiles use the same standard K-means partitions and expose
    # whether agreement survives when the label resolution changes.  These
    # are conditional profiles, not independent endpoint replications.
    hierarchy_rows = []
    hierarchy = {"syntemp_r0": ["label_0", "label_1", "label_2"],
                 "mech_2": ["level1", "level2"]}
    for p in EXP.glob("05_clustering/results/*/*/kmeans_*.json"):
        r = json.loads(p.read_text())
        ds = r.get("dataset")
        if ds not in hierarchy:
            continue
        for label in hierarchy[ds]:
            key = f"{label}__ami"
            if key in r:
                hierarchy_rows.append({"dataset": ds, "fingerprint": r["fingerprint"],
                                       "seed": r.get("seed"), "label": label,
                                       "ami": r[key]})
    hierarchy_df = pd.DataFrame(hierarchy_rows)
    if hierarchy_df.empty:
        raise ValueError("No hierarchy K-means scores found")
    hierarchy_df.to_csv(OUT / "clustering_hierarchy_ami_by_seed.csv", index=False)
    hierarchy_df.groupby(["dataset", "fingerprint", "label"]).ami.agg(
        ["mean", "std", "count"]).to_csv(OUT / "clustering_hierarchy_ami.csv")

    # Interim benchmark summaries are useful while the large TPL XGBoost
    # cells are still running.  The count column makes partial coverage
    # explicit; this file must be regenerated before final claims.
    benchmark_rows = []
    for p in EXP.glob("03_classification/results/benchmark/*/*/*.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("model") in {"knn", "mlp", "xgb"} and "mcc" in r:
            benchmark_rows.append({"model": r["model"], "dataset": r["dataset"],
                                   "fingerprint": r["fingerprint"], "mcc": r["mcc"]})
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.groupby(["model", "dataset", "fingerprint"]).mcc.agg(
        ["mean", "std", "count"]).reset_index().to_csv(
            OUT / "benchmark_completed_summary_partial.csv", index=False)
    # Label-free leaderboard view.  RXNFP checkpoints are retained in the
    # summary, but excluded here because RXNFP-FT is supervised and should
    # not be mixed into the representation ranking.
    label_free = {"synrfp-wl-r2-b1024-parity-delta", "drfp", "rdkit_struct",
                  "rdkit_diff", "rxnfp_mlm"}
    leaderboard = (benchmark_df[benchmark_df.fingerprint.isin(label_free)]
                   .groupby(["model", "dataset", "fingerprint"])
                   .mcc.agg(mean="mean", std="std", count="count")
                   .reset_index())
    leaderboard["rank"] = leaderboard.groupby(["model", "dataset"])["mean"].rank(
        ascending=False, method="min")
    leaderboard.to_csv(OUT / "benchmark_label_free_leaderboard_partial.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, summary, title, xlabel in [
        (axes[0], mode_summary, "kNN: complete factorial\n5 matched random splits",
         "MCC gain: delta minus union"),
        (axes[1], cluster_summary, "KMeans: default-context ablation\n3 initialization seeds",
         "AMI gain: delta minus union"),
    ]:
        summary = summary.sort_values("mean")
        ax.errorbar(summary["mean"], range(len(summary)), xerr=summary["std"], fmt="o",
                    color="#187b80", ecolor="#333333", capsize=3)
        ax.set_yticks(range(len(summary)), [LABELS[d] for d in summary.index])
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_xlim(-0.02, 0.36)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Delta encoding outperforms union across these evaluated tasks", fontsize=13)
    fig.text(0.5, 0.018, "Bars: ±1 sample SD of paired contrasts. Descriptive variability, not confidence intervals.\n"
             "Left: averaged over the other 81 factor combinations within each split. Right: WL/r2/1024/parity.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"delta_union_evidence.{ext}", dpi=180)
    plt.close(fig)
    print("Validated 8100 kNN factorial cells and 12 paired clustering contrasts.")
    print("Factorial mode means:", runs.groupby("mode").mcc.mean().to_dict())
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
