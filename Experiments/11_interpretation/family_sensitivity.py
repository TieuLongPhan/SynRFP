"""Family-balanced and leave-one-family-out classification sensitivity.

This is descriptive and uses only completed benchmark JSON records.  It keeps
dataset families as the unit of resampling, so repeated hierarchy/variant
endpoints do not silently dominate the comparison.
"""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
OUT = HERE / "results"
FAMILY = {
    "schneider_ub": "Schneider", "schneider_b": "Schneider",
    "tpl_ub": "TPL", "tpl_b": "TPL",
    "uspto50k_ub": "USPTO50K", "uspto50k_b": "USPTO50K",
    "syntemp_r0": "SynTemp", "syntemp_r1": "SynTemp", "syntemp_r2": "SynTemp",
    "ecreact_1": "ECReact", "ecreact_2": "ECReact", "ecreact_3": "ECReact",
    "mech_1": "Mech", "mech_2": "Mech",
}
FPS = ["synrfp-wl-r2-b1024-parity-delta", "drfp", "rdkit_struct", "rdkit_diff", "rxnfp_mlm"]

def load():
    rows = []
    root = EXP / "03_classification" / "results" / "benchmark"
    for p in root.rglob("*.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("fingerprint") in FPS and r.get("model") in {"knn", "mlp", "xgb"} and "mcc" in r:
            rows.append({k: r.get(k) for k in ["model", "dataset", "fingerprint", "scheme", "mcc"]})
    if not rows:
        raise SystemExit("no benchmark records")
    d = pd.DataFrame(rows)
    d["family"] = d["dataset"].map(FAMILY).fillna("Unknown")
    return d

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    d = load()
    # Collapse repeats, then retain only matched fingerprint sets within each
    # model/dataset/scheme. Missing TPL cells therefore remain visible.
    s = d.groupby(["model", "dataset", "family", "scheme", "fingerprint"], as_index=False).mcc.mean()
    wide = s.pivot_table(index=["model", "dataset", "family", "scheme"], columns="fingerprint", values="mcc")
    wide = wide.dropna(subset=["synrfp-wl-r2-b1024-parity-delta"])
    base = "synrfp-wl-r2-b1024-parity-delta"
    rows = []
    for model in sorted(wide.index.get_level_values("model").unique()):
        sub = wide.xs(model, level="model")
        for fp in FPS[1:]:
            if fp not in sub:
                continue
            delta = sub[base] - sub[fp]
            rows.append({"model": model, "comparison": f"SynRFP - {fp}", "scope": "all_matched",
                         "families": ",".join(sorted(sub.index.get_level_values("family").unique())),
                         "n_dataset_scheme": int(delta.notna().sum()), "mean_delta": delta.mean()})
            for fam in sorted(sub.index.get_level_values("family").unique()):
                loo = sub[sub.index.get_level_values("family") != fam]
                if fp not in loo:
                    continue
                dd = loo[base] - loo[fp]
                rows.append({"model": model, "comparison": f"SynRFP - {fp}", "scope": f"leave_out_{fam}",
                             "families": ",".join(sorted(loo.index.get_level_values("family").unique())),
                             "n_dataset_scheme": int(dd.notna().sum()), "mean_delta": dd.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "classification_family_leave_one_out.csv", index=False)
    coverage = (s.groupby(["model", "family", "fingerprint"]).size().rename("n").reset_index())
    coverage.to_csv(OUT / "classification_family_coverage.csv", index=False)
    print(f"Wrote {len(out)} leave-one-family-out contrasts and coverage to {OUT}")

if __name__ == "__main__":
    main()
