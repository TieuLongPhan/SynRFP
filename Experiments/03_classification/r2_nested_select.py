#!/usr/bin/env python3
"""Select one r2 MLP candidate per fingerprint and outer fold from inner MCC."""
import json
from collections import defaultdict
from pathlib import Path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in args.out.glob("outer*_inner*_*.json"):
        rec = json.loads(path.read_text())
        if rec.get("study") != "r2_grouped_development_screen_v1":
            continue
        p = rec["params"]
        key = (rec["outer_seed"], rec["fingerprint"], p["family"], p["aux_weight"], p["lr"])
        groups[key].append(rec)
    selected = []
    for outer in range(25):
        for fp in ["synrfp-wl-r2-b1024-parity-delta", "drfp"]:
            candidates = []
            for (o, f, family, aux, lr), records in groups.items():
                if o != outer or f != fp:
                    continue
                if len(records) != 3:
                    raise RuntimeError("missing inner fold for outer %d %s" % (outer, fp))
                candidates.append({"family": family, "aux_weight": aux, "lr": lr,
                                   "mean_validation_mcc": sum(r["validation_mcc"] for r in records) / 3,
                                   "inner_mcc": [r["validation_mcc"] for r in records],
                                   "refit_epochs": int(round(sorted(r["best_epoch"] for r in records)[1]))})
            if len(candidates) != 3:
                raise RuntimeError("expected 3 candidates for outer %d %s" % (outer, fp))
            candidates.sort(key=lambda c: c["mean_validation_mcc"], reverse=True)
            selected.append({"outer_seed": outer, "fingerprint": fp,
                             "selected": candidates[0], "candidates": candidates})
    result = {"study": "r2_nested_grouped_mlp_v1", "selection_metric": "mean inner scaffold-grouped MCC",
              "outer_test_used_for_selection": False, "selected": selected}
    (args.out / "selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
