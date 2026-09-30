"""Summarize where SynRFP wins, ties, and loses on completed results.

This is a descriptive, completeness-aware analysis.  It does not refit models
and it does not replace the primary benchmark tables.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SYN = "synrfp-wl-r2-b1024-parity-delta"
BASELINES = ["drfp", "rdkit_diff", "rdkit_struct", "rxnfp_mlm"]


def classification():
    d = pd.read_csv(RES / "classification_label_free_paired.csv")
    d = d[d["scheme"].eq("cv")].copy()
    out = {}
    for model, g in d.groupby("model", sort=True):
        rec = {"rows": int(len(g)), "comparisons": {}}
        for base in BASELINES:
            delta = g[SYN] - g[base]
            rec["comparisons"][base] = {
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "mean_margin": float(delta.mean()),
                "min_margin": float(delta.min()),
                "max_margin": float(delta.max()),
                "loss_datasets": sorted(g.loc[delta < 0, "dataset"].tolist()),
            }
        rec["synrfp_mean_rank"] = float(g[f"rank_{SYN}"].mean())
        rec["baseline_mean_ranks"] = {
            b: float(g[f"rank_{b}"].mean()) for b in BASELINES
        }
        out[model] = rec
    return out


def clustering():
    d = pd.read_csv(RES / "clustering_ami_ari.csv", skiprows=[1, 2])
    d = d.rename(columns={
        "Unnamed: 0": "phase", "Unnamed: 1": "dataset",
        "Unnamed: 2": "fingerprint", "ami": "ami_mean",
        "ari": "ari_mean",
    })
    # The flattened CSV has duplicate metric columns; the first mean column
    # is the one named above and is sufficient for this descriptive summary.
    d["ami_mean"] = pd.to_numeric(d["ami_mean"], errors="coerce")
    d["ari_mean"] = pd.to_numeric(d["ari_mean"], errors="coerce")
    d = d.dropna(subset=["ami_mean", "ari_mean"])
    out = {}
    for (phase, dataset), g in d.groupby(["phase", "dataset"], sort=True):
        rec = {"fingerprints": int(len(g)), "leaders": {}}
        for metric in ["ami_mean", "ari_mean"]:
            order = g.sort_values(metric, ascending=False)
            rec["leaders"][metric[:3].upper()] = {
                "fingerprint": str(order.iloc[0]["fingerprint"]),
                "value": float(order.iloc[0][metric]),
                "synrfp_value": float(g.loc[g.fingerprint.eq(SYN), metric].iloc[0])
                if (g.fingerprint == SYN).any() else None,
            }
        out[f"{phase}/{dataset}"] = rec
    return out


def main():
    payload = {"classification_cv": classification(), "clustering": clustering()}
    (RES / "applicability_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# SynRFP applicability and failure-mode analysis",
        "",
        "This descriptive analysis uses completed CV classification rows and the",
        "existing ten-start clustering summaries. It is not a new benchmark or",
        "a replacement for the completeness-gated final leaderboard.",
        "",
        "## Classification (CV rows)",
        "",
        "| Model | Baseline | SynRFP wins | Ties | Losses | Mean MCC margin | Loss datasets |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model, rec in payload["classification_cv"].items():
        for base, x in rec["comparisons"].items():
            losses = ", ".join(x["loss_datasets"]) or "none"
            lines.append(
                f"| {model} | {base} | {x['wins']} | {x['ties']} | {x['losses']} | "
                f"{x['mean_margin']:+.4f} | {losses} |"
            )
    lines += [
        "",
        "The completed rows show a consistent pattern: SynRFP is strongest",
        "against structural and RXNFP-MLM references, while RDKit difference is",
        "the closest XGBoost comparator. TPL and fine-resolution SynTemp rows",
        "are recurring stress cases and should be discussed explicitly.",
        "",
        "## Clustering",
        "",
        "The clustering leader is recorded per phase/dataset for both AMI and ARI",
        "in `results/applicability_analysis.json`. Large-data exceptions should",
        "be reported instead of averaged away.",
    ]
    (ROOT / "APPLICABILITY_ANALYSIS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
