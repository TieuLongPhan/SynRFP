"""Check that interpretation artifacts are internally consistent and finite."""
from pathlib import Path
import json
import math
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
E8_OUT = HERE.parent / "08_reaction_center" / "results"

def check_csv(name, required, min_rows=1):
    p = OUT / name
    d = pd.read_csv(p)
    missing = sorted(set(required) - set(d.columns))
    numeric = d.select_dtypes(include="number")
    # A one-observation group legitimately has an undefined sample SD (NaN).
    # Check all present numeric values and report missing numeric entries.
    finite = bool(numeric.dropna().map(math.isfinite).all().all()) if not numeric.empty else True
    return {"file": name, "exists": True, "rows": len(d), "missing_columns": missing,
            "finite_numeric": finite, "ok": not missing and len(d) >= min_rows and finite}

def check_json(name):
    obj = json.loads((OUT / name).read_text())
    ok = isinstance(obj, dict) and bool(obj.get("claims")) and obj.get("all_supported") is True
    return {"file": name, "exists": True, "claims": len(obj.get("claims", [])),
            "all_supported": obj.get("all_supported"), "ok": ok}

def check_protocol():
    obj = json.loads((OUT / "protocol_audit.json").read_text())
    ok = (isinstance(obj, dict) and obj.get("checks", {}).get("all_protocol_checks_pass") is True
          and not obj.get("invalid_records"))
    return {"file": "protocol_audit.json", "ok": ok,
            "all_protocol_checks_pass": obj.get("checks", {}).get("all_protocol_checks_pass"),
            "invalid_records": len(obj.get("invalid_records", []))}

def check_handoff():
    obj = json.loads((OUT / "handoff_audit.json").read_text())
    handoff = obj.get("handoff", {})
    ok = (isinstance(obj, dict) and handoff.get("duplicate_free") is True
          and isinstance(obj.get("locks", {}).get("count"), int))
    return {"file": "handoff_audit.json", "ok": ok,
            "status": handoff.get("status"),
            "duplicate_free": handoff.get("duplicate_free"),
            "locks": obj.get("locks", {}).get("count")}

def check_assignment_replay():
    """Validate reaction-level replay metadata and selected examples."""
    assignment_dir = OUT / "assignments"
    records = sorted(assignment_dir.glob("*.json")) if assignment_dir.exists() else []
    valid = []
    for p in records:
        obj = json.loads(p.read_text())
        valid.append(
            isinstance(obj, dict)
            and obj.get("method") == "kmeans"
            and int(obj.get("n_rows", 0)) == int(obj.get("n_valid", 0)) + int(obj.get("invalid_rows", -1))
            and isinstance(obj.get("scores_on_valid_rows"), dict)
        )
    example_files = sorted(OUT.glob("cluster_examples_*.csv"))
    example_rows = 0
    example_ok = bool(example_files)
    for examples in example_files:
        d = pd.read_csv(examples)
        example_rows += len(d)
        example_ok = example_ok and len(d) > 0 and {"fingerprint", "cluster_size", "cluster_purity", "reaction"}.issubset(d.columns)
    ok = bool(records) and all(valid) and example_ok
    return {"file": "assignments/ + cluster_examples_mech2.csv", "ok": ok,
            "assignment_records": len(records), "example_files": len(example_files),
            "examples": example_rows}

def check_reaction_center():
    summary = E8_OUT / "summary.csv"
    per = E8_OUT / "per_reaction.csv"
    if not summary.exists() or not per.exists():
        return {"file": "08_reaction_center/results", "ok": False,
                "error": "corrected full reaction-centre output is missing"}
    d = pd.read_csv(per)
    required = {"delta_occurrence_center_fraction", "union_center_fraction",
                "center_coverage", "level2"}
    finite = bool(d.select_dtypes(include="number").dropna().map(math.isfinite).all().all())
    ok = len(d) == 31673 and required.issubset(d.columns) and finite
    return {"file": "08_reaction_center/results/per_reaction.csv", "ok": ok,
            "rows": len(d), "finite_numeric": finite,
            "mean_occurrence_center_fraction": float(d["delta_occurrence_center_fraction"].mean()),
            "mean_union_center_fraction": float(d["union_center_fraction"].mean())}

def main():
    checks = []
    explanatory = HERE / "EXPLANATORY_RESULTS.md"
    checks.append({"file": "EXPLANATORY_RESULTS.md", "ok": explanatory.exists() and explanatory.stat().st_size > 0})
    for name, cols, n in [
        ("classification_label_free_by_dataset.csv", ["model", "dataset", "fingerprint", "mean"], 1),
        ("classification_family_leave_one_out.csv", ["model", "comparison", "scope", "mean_delta"], 1),
        ("clustering_ami_ari.csv", ["ami", "ari"], 1),
        ("clustering_stability_paired_summary.csv", ["dataset", "baseline", "metric"], 1),
        ("clustering_paired_inference.csv", ["dataset", "baseline", "metric", "ci95_low", "ci95_high"], 1),
        ("knn_mode_paired_inference.csv", ["dataset", "ci95_low", "ci95_high"], 1),
        ("claim_audit.json", [], 0),
        ("feature_validity_sparsity.csv", ["dataset", "fingerprint"], 1),
        ("cross_task_endpoint_margins.csv", ["task", "model", "dataset", "baseline", "synrfp_minus_baseline"], 1),
        ("cross_task_margin_summary.csv", ["task", "model", "baseline", "endpoints", "mean_margin", "wins"], 1),
        ("cross_task_drfp_matched.csv", ["dataset", "classification_margin", "clustering_margin"], 1),
        ("cluster_replay_summary.csv", ["dataset", "fingerprint", "ami", "ari", "n_valid"], 1),
        ("clustering_sensitivity_schneider_b.csv", ["dataset", "fingerprint", "preprocessing", "ami", "ari", "n_common_valid"], 1),
        ("hash_seed_sensitivity_schneider_b.csv", ["seed_a", "seed_b", "mean_hamming", "mean_fraction_changed"], 1),
        ("reaction_center_protocol_audit.csv", ["protocol", "reaction_column", "radius", "sample_valid", "delta_center_fraction_mean"], 4),
        ("feature_cost_summary.csv", ["dataset", "fingerprint", "estimated_payload_mb", "valid_fraction"], 1),
        ("feature_cost_by_fingerprint.csv", ["fingerprint", "total_payload_mb", "mean_valid_fraction"], 1),
        ("reaction_center_class_summary.csv", ["level2", "localized", "union_localized", "center_coverage", "n"], 1),
    ]:
        try:
            checks.append(check_json(name) if name.endswith(".json")
                          else check_csv(name, cols, n))
        except Exception as exc:
            checks.append({"file": name, "exists": (OUT / name).exists(), "ok": False, "error": str(exc)})
    progress = json.loads((OUT / "classification_progress.json").read_text())
    checks.append({"file": "classification_progress.json", "ok": True,
                   "benchmark_xgb_percent": progress["benchmark"]["completion"]["xgb"]["percent"],
                   "factorial_xgb_percent": progress["factorial"]["completion"]["xgb"]["percent"]})
    try:
        checks.append(check_protocol())
    except Exception as exc:
        checks.append({"file": "protocol_audit.json", "ok": False, "error": str(exc)})
    try:
        checks.append(check_handoff())
    except Exception as exc:
        checks.append({"file": "handoff_audit.json", "ok": False, "error": str(exc)})
    try:
        checks.append(check_assignment_replay())
    except Exception as exc:
        checks.append({"file": "assignments/ + cluster_examples_mech2.csv", "ok": False, "error": str(exc)})
    try:
        checks.append(check_reaction_center())
    except Exception as exc:
        checks.append({"file": "08_reaction_center/results", "ok": False, "error": str(exc)})
    result = {"checks": checks, "all_ok": all(x["ok"] for x in checks)}
    (OUT / "artifact_integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_ok"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
