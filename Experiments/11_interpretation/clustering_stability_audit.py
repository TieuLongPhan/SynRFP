"""Summarize paired K-means stability subsamples from saved results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
OUT = HERE / "results"
SYN = "synrfp-wl-r2-b1024-parity-delta"
BASELINES = ["drfp", "rdkit_struct", "rdkit_diff", "rxnfp_mlm"]


def main():
    rows = []
    for p in (EXP / "05_clustering" / "results_followup" / "stability80").rglob("*.json"):
        try: r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError): continue
        if r.get("fingerprint") in [SYN, *BASELINES] and "label__ami" in r:
            rows.append({"dataset": r["dataset"], "sample_seed": r["sample_seed"],
                         "fingerprint": r["fingerprint"], "ami": r["label__ami"],
                         "ari": r["label__ari"], "completeness": r["label__completeness"],
                         "singleton_fraction": r["label__singleton_fraction"]})
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit("No stability results found")
    out = []
    for (dataset, seed), g in df.groupby(["dataset", "sample_seed"]):
        wide = g.set_index("fingerprint")
        if SYN not in wide.index: continue
        for baseline in BASELINES:
            if baseline not in wide.index: continue
            for metric in ["ami", "ari", "completeness", "singleton_fraction"]:
                out.append({"dataset": dataset, "sample_seed": seed, "baseline": baseline,
                            "metric": metric, "delta": wide.loc[SYN, metric] - wide.loc[baseline, metric]})
    paired = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUT / "clustering_stability_paired_deltas.csv", index=False)
    summary = (paired.groupby(["dataset", "baseline", "metric"]).delta
               .agg(mean="mean", std="std", count="count",
                    wins=lambda x: int((x > 0).sum()), ties=lambda x: int((x == 0).sum()))
               .reset_index())
    summary.to_csv(OUT / "clustering_stability_paired_summary.csv", index=False)
    ami = paired[paired.metric.eq("ami")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels, vals = [], []
    for (dataset, baseline), g in ami.groupby(["dataset", "baseline"]):
        labels.append(f"{dataset}\nvs {baseline}"); vals.append(g.delta.to_numpy())
    ax.boxplot(vals, orientation="horizontal", tick_labels=labels, showmeans=True)
    ax.axvline(0, color="#777", lw=.8); ax.set_xlabel("Paired AMI difference: SynRFP − baseline")
    ax.set_title("K-means stability subsamples: paired AMI differences")
    fig.tight_layout(); fig.savefig(OUT / "clustering_stability_paired_ami.png", dpi=180)
    plt.close(fig)
    print(f"Summarized {len(paired)} paired metric contrasts across {len(ami)} AMI contrasts")


if __name__ == "__main__": main()
