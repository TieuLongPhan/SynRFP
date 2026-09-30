"""Verify manuscript-facing headline claims against saved result tables."""
from pathlib import Path
import json

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SYN = "synrfp-wl-r2-b1024-parity-delta"


def clustering_table():
    path = OUT / "clustering_ami_ari.csv"
    return pd.read_csv(path, skiprows=2,
                       names=["phase", "dataset", "fingerprint", "ami_mean",
                              "ami_sd", "ami_n", "ari_mean", "ari_sd", "ari_n"]).dropna()


def main():
    c = clustering_table()
    checks = []
    for phase, expected_n in [("small_kmeans", 7), ("large_minibatch", 5)]:
        sub = c[c.phase.eq(phase)]
        wins = []
        for dataset, g in sub.groupby("dataset"):
            top = g.loc[g.ami_mean.idxmax(), "fingerprint"]
            wins.append(top == SYN)
        expected_wins = 7 if phase == "small_kmeans" else 4
        checks.append({"claim": f"SynRFP AMI result on {phase}",
                       "datasets": len(wins), "expected_datasets": expected_n,
                       "wins": int(sum(wins)), "expected_wins": expected_wins,
                       "supported": len(wins) == expected_n and int(sum(wins)) == expected_wins})

    stab = pd.read_csv(OUT / "clustering_paired_inference.csv")
    ami = stab[stab.metric.eq("ami")]
    for baseline in ["drfp", "rdkit_struct", "rdkit_diff", "rxnfp_mlm"]:
        q = ami[ami.baseline.eq(baseline)]
        checks.append({"claim": f"SynRFP AMI beats {baseline} on every paired stability run",
                       "groups": len(q), "runs": int(q.n.sum()), "wins": int(q.wins.sum()),
                       "supported": bool(len(q) and (q.wins == q.n).all())})

    knn = pd.read_csv(OUT / "knn_mode_paired_inference.csv")
    checks.append({"claim": "kNN signed delta exceeds union on every represented endpoint",
                   "endpoints": len(knn), "runs": int(knn.n.sum()), "wins": int(knn.wins.sum()),
                   "supported": bool(len(knn) and (knn.wins == knn.n).all())})

    payload = {"claims": checks, "all_supported": all(x["supported"] for x in checks)}
    (OUT / "claim_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Claim audit", "", f"All checks supported: **{payload['all_supported']}**", "",
             "| Claim | Evidence | Status |", "|---|---:|---|"]
    for x in checks:
        evidence = x.get("wins", "")
        if "datasets" in x:
            evidence = f"{x['wins']}/{x['datasets']} datasets"
        elif "groups" in x:
            evidence = f"{x['wins']}/{x['runs']} paired runs"
        else:
            evidence = f"{x['wins']}/{x['runs']} paired runs"
        lines.append(f"| {x['claim']} | {evidence} | {'supported' if x['supported'] else 'FAIL'} |")
    (OUT / "CLAIM_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
