"""Audit the declared split and replicate-seed protocol without fitting models."""
from pathlib import Path
import json
import itertools
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CFG = EXP / "config.yaml"
OUT = HERE / "results"


def main():
    cfg = yaml.safe_load(CFG.read_text())
    checks = {
        "fp_seed": cfg["fp_seed"],
        "fixed_random_seed": cfg["splits"]["fixed_random_seed"],
        "cv_seed_count": len(cfg["benchmark"]["schemes"]["cv"]),
        "fixed_seed_count": len(cfg["benchmark"]["schemes"]["fixed"]),
        "factorial_seed_count": len(cfg["factorial"]["seeds"]),
        "factorial_grid_size": 1,
    }
    grid = cfg["synrfp"]["grid"]
    for values in grid.values():
        checks["factorial_grid_size"] *= len(values)
    checks["expected_factorial_cells_per_model"] = (
        checks["factorial_grid_size"] * len(cfg["factorial"]["datasets"])
        * checks["factorial_seed_count"]
    )
    checks["cv_scheme_is_25_repeated_grouped_splits"] = (
        cfg["benchmark"]["schemes"]["cv"] == list(range(25))
        and cfg["splits"]["grouped_folds"] == 10
    )
    checks["track_b_fallback_is_seed_42"] = checks["fixed_random_seed"] == 42
    checks["synrfp_hash_seed_is_fixed"] = checks["fp_seed"] == 1

    # Result metadata must use only declared seeds and schemes. This remains
    # useful while queues are partial because it checks every saved record.
    bad = []
    root = EXP / "03_classification" / "results"
    for p in root.glob("**/*.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(r, dict) or not {"study", "dataset", "scheme", "seed", "fingerprint", "model"}.issubset(r):
            continue
        if r["study"] == "benchmark":
            allowed = set(cfg["benchmark"]["schemes"].get(r["scheme"], []))
        elif r["study"] == "factorial":
            allowed = set(cfg["factorial"]["seeds"])
        else:
            continue
        if r["seed"] not in allowed or r["scheme"] not in ("cv", "fixed", "random"):
            bad.append(str(p))
    checks["invalid_saved_seed_records"] = len(bad)
    checks["all_protocol_checks_pass"] = all([
        checks["cv_scheme_is_25_repeated_grouped_splits"],
        checks["track_b_fallback_is_seed_42"],
        checks["synrfp_hash_seed_is_fixed"],
        checks["invalid_saved_seed_records"] == 0,
    ])
    obj = {"checks": checks, "invalid_records": bad}
    (OUT / "protocol_audit.json").write_text(json.dumps(obj, indent=2) + "\n")
    lines = ["# Seed and split protocol audit", "",
             f"- SynRFP hash seed: **{checks['fp_seed']}** (fixed across runs).",
             f"- Track A: **{checks['cv_seed_count']}** grouped scaffold splits; "
             "seed `repeat*5 + fold`, with repeat-specific shuffled folds.",
             f"- Track B: **{checks['fixed_seed_count']}** model seeds on one split; "
             f"datasets without a published split use fallback seed **{checks['fixed_random_seed']}**.",
             f"- Factorial: **{checks['factorial_grid_size']}** configurations × "
             f"{len(cfg['factorial']['datasets'])} endpoints × "
             f"{checks['factorial_seed_count']} seeds = "
             f"**{checks['expected_factorial_cells_per_model']} cells/model**.",
             f"- Saved records with undeclared seed metadata: **{len(bad)}**.", "",
             "This audit separates hash/preprocessing determinism from split and",
             "model-initialization replicates; it does not assess statistical independence.", ""]
    (HERE / "PROTOCOL_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps(obj, indent=2))
    if not checks["all_protocol_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
