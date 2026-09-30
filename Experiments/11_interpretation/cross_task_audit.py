"""Cross-task consistency audit for the frozen label-free fingerprints.

This uses completed CV classification summaries and completed small-endpoint
K-means summaries.  It is descriptive: endpoints share source material and
the two tasks use different evaluation protocols, so it is not a pooled
significance test.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SYN = "synrfp-wl-r2-b1024-parity-delta"
BASELINES = ["drfp", "rdkit_diff", "rdkit_struct", "rxnfp_mlm"]
DATASETS = ["schneider_b", "uspto50k_b", "syntemp_r0", "ecreact_1",
            "mech_1", "mech_2"]


def main():
    cls = pd.read_csv(OUT / "classification_label_free_by_dataset.csv")
    cls = cls[(cls.scheme == "cv") & cls.dataset.isin(DATASETS)]
    cls = cls[cls.fingerprint.isin([SYN, *BASELINES])]
    # one row per model, endpoint, fingerprint; duplicate rows would make a
    # supposedly paired comparison ambiguous.
    cls = (cls.groupby(["model", "dataset", "fingerprint"], as_index=False)
              .mean(numeric_only=True))
    crows = []
    for (model, dataset), g in cls.groupby(["model", "dataset"]):
        vals = g.set_index("fingerprint")["mean"]
        if SYN not in vals:
            continue
        for baseline in BASELINES:
            if baseline in vals:
                crows.append({"task": "classification", "model": model,
                              "dataset": dataset, "baseline": baseline,
                              "synrfp_minus_baseline": vals[SYN] - vals[baseline]})
    cdf = pd.DataFrame(crows)

    raw = pd.read_csv(OUT / "clustering_ami_ari.csv", header=[0, 1])
    # inspect_existing writes a two-row metric header followed by the index
    # columns. Read the underlying table directly to avoid depending on the
    # display-oriented header shape.
    raw = pd.read_csv(OUT / "clustering_ami_ari.csv", skiprows=2,
                      names=["phase", "dataset", "fingerprint", "ami_mean",
                             "ami_std", "ami_count", "ari_mean", "ari_std",
                             "ari_count"])
    cl = raw[(raw.phase == "small_kmeans") & raw.dataset.isin(DATASETS)]
    cl = cl[cl.fingerprint.isin([SYN, *BASELINES])]
    crows = []
    for dataset, g in cl.groupby("dataset"):
        vals = g.set_index("fingerprint")["ami_mean"]
        if SYN not in vals:
            continue
        for baseline in BASELINES:
            if baseline in vals:
                crows.append({"task": "clustering", "model": "kmeans",
                              "dataset": dataset, "baseline": baseline,
                              "synrfp_minus_baseline": vals[SYN] - vals[baseline]})
    kdf = pd.DataFrame(crows)
    all_rows = pd.concat([cdf, kdf], ignore_index=True)
    all_rows.to_csv(OUT / "cross_task_endpoint_margins.csv", index=False)

    summary = (all_rows.groupby(["task", "model", "baseline"])
               .agg(endpoints=("synrfp_minus_baseline", "count"),
                    mean_margin=("synrfp_minus_baseline", "mean"),
                    median_margin=("synrfp_minus_baseline", "median"),
                    wins=("synrfp_minus_baseline", lambda x: int((x > 0).sum())),
                    ties=("synrfp_minus_baseline", lambda x: int((x == 0).sum())))
               .reset_index())
    summary["win_rate"] = summary.wins / summary.endpoints
    summary.to_csv(OUT / "cross_task_margin_summary.csv", index=False)

    # A compact matched-task view against DRFP, where the endpoint names are
    # directly shared. This is a descriptive rank-consistency diagnostic.
    drfp = all_rows[all_rows.baseline == "drfp"].pivot_table(
        index="dataset", columns="task", values="synrfp_minus_baseline")
    if not drfp.empty:
        drfp["same_sign"] = (drfp.get("classification", 0) *
                              drfp.get("clustering", 0) > 0)
        drfp["classification_margin"] = drfp.get("classification")
        drfp["clustering_margin"] = drfp.get("clustering")
    drfp.reset_index().to_csv(OUT / "cross_task_drfp_matched.csv", index=False)

    lines = ["# Cross-task representation consistency",
             "",
             "This is a descriptive audit of the frozen label-free fingerprints.",
             "Classification uses scaffold-CV MCC summaries; clustering uses",
             "small-endpoint K-means AMI summaries. Shared datasets are related",
             "views, so margins are not independent evidence or a pooled test.",
             "",
             "## Summary",
             ""]
    for _, r in summary.iterrows():
        lines.append(f"- {r.task} / {r.model} vs {r.baseline}: "
                     f"SynRFP wins {int(r.wins)}/{int(r.endpoints)} endpoints "
                     f"({r.win_rate:.1%}); mean margin {r.mean_margin:+.4f}.")
    lines += ["", "## Interpretation", "",
              "A positive margin in both tasks on the same endpoint supports a",
              "representation-level signal that is not tied to one estimator.",
              "A sign change identifies a task/protocol-dependent comparison and",
              "should be reported as such. This audit does not replace paired",
              "split or initialization inference.", ""]
    (HERE / "CROSS_TASK_CONSISTENCY.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
