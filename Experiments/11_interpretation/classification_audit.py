"""Audit completed classification records without fitting new models.

The output deliberately keeps coverage visible.  It reports label-free
fingerprint comparisons at the dataset level and an equally weighted family
summary, so a large dataset family cannot silently dominate the headline.
"""
from pathlib import Path
import json

import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
OUT = HERE / "results"
LABEL_FREE = {
    "synrfp-wl-r2-b1024-parity-delta", "drfp", "rdkit_struct",
    "rdkit_diff", "rxnfp_mlm",
}
FAMILY = {
    "schneider_ub": "Schneider", "schneider_b": "Schneider",
    "tpl_ub": "TPL", "tpl_b": "TPL",
    "uspto50k_ub": "USPTO50K", "uspto50k_b": "USPTO50K",
    "syntemp_r0": "SynTemp", "syntemp_r1": "SynTemp", "syntemp_r2": "SynTemp",
    "ecreact_1": "ECReact", "ecreact_2": "ECReact", "ecreact_3": "ECReact",
    "mech_1": "Mech", "mech_2": "Mech",
}


def load_records():
    rows = []
    root = EXP / "03_classification" / "results" / "benchmark"
    for p in root.rglob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("fingerprint") in LABEL_FREE and rec.get("model") in {"knn", "mlp", "xgb"}:
            if "mcc" in rec:
                rows.append({k: rec.get(k) for k in
                             ["model", "dataset", "fingerprint", "scheme", "seed",
                              "mcc", "macro_f1", "balanced_accuracy", "epochs"]})
    if not rows:
        raise SystemExit("No completed label-free classification records found")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_records()
    df["family"] = df.dataset.map(FAMILY).fillna("Unknown")
    # One row per model/dataset/fingerprint/scheme, retaining replicate spread.
    summary = (df.groupby(["model", "dataset", "family", "fingerprint", "scheme"],
                          dropna=False)
                 .mcc.agg(mean="mean", std="std", count="count")
                 .reset_index())
    summary.to_csv(OUT / "classification_label_free_by_dataset.csv", index=False)

    # Rank only within matched model/dataset/scheme groups.  This makes an
    # incomplete TPL run visible in coverage instead of treating missing as 0.
    pivot = (summary.pivot_table(index=["model", "dataset", "family", "scheme"],
                                 columns="fingerprint", values="mean")
                    .reset_index())
    for fp in sorted(LABEL_FREE):
        if fp in pivot:
            pivot[f"rank_{fp}"] = pivot[list(LABEL_FREE & set(pivot.columns))].rank(
                axis=1, ascending=False, method="min")[fp]
    pivot.to_csv(OUT / "classification_label_free_paired.csv", index=False)

    # Family macro means first average datasets, then families.  A family with
    # two hierarchy endpoints therefore contributes one macro value, not two.
    fam = (summary.groupby(["model", "family", "fingerprint", "scheme"],
                           dropna=False).mean(numeric_only=True).reset_index())
    fam.to_csv(OUT / "classification_label_free_family_means.csv", index=False)
    fam_macro = (fam.groupby(["model", "fingerprint", "scheme"], dropna=False)
                    .mean(numeric_only=True).reset_index())
    fam_macro.to_csv(OUT / "classification_label_free_family_macro.csv", index=False)

    coverage = (df.groupby(["model", "dataset", "fingerprint", "scheme"], dropna=False)
                  .size().rename("n_records").reset_index())
    coverage.to_csv(OUT / "classification_label_free_coverage.csv", index=False)

    # Convergence evidence is descriptive: epoch metadata exists for MLP only.
    epochs = df[df.model.eq("mlp") & df.epochs.notna()]
    if not epochs.empty:
        (epochs.groupby("fingerprint").epochs.agg(median="median", max="max", count="count")
         .reset_index().to_csv(OUT / "classification_mlp_convergence.csv", index=False))

    print(f"Loaded {len(df)} label-free records; wrote classification audit to {OUT}")
    print("Models:", df.groupby("model").size().to_dict())
    print("Dataset/fingerprint groups:", len(summary))


if __name__ == "__main__":
    main()
