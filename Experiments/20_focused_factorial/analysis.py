"""Descriptive paired factorial contrasts after all2555 canonical cells finish.

Reference-level marginal main effects and two-factor differences of differences
are computed within each frozen dataset/split. Remaining axes are averaged
equally. Five correlated repeats supply descriptive sample SDs, not p-values,
confidence intervals, or independent-population uncertainty estimates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path.home() / ".local/share/synrfp-runs/focused_factorial_v1"
LEVELS = {"tokenizer": ["wl", "morgan", "path"], "radius": [2, 3],
          "sketch": ["parity", "minhash", "cw"], "mode": ["delta", "union", "delta_union", "roles"]}
REFERENCES = {"tokenizer": "wl", "radius": 2, "sketch": "parity", "mode": "delta"}
FACTORS = list(LEVELS)
METRICS = {"classification": ["mcc", "macro_f1", "accuracy", "fit_seconds"],
           "yield": ["r2", "mae", "rmse", "fit_seconds"],
           "clustering": ["ami", "ari", "nmi", "silhouette", "largest_cluster_fraction",
                          "reaction_singleton_fraction", "actual_clusters", "fit_seconds"]}
COUNTS = {"classification": 730, "yield": 1095, "clustering": 730}
SCHEMA = "synrfp-focused-factorial-descriptive-analysis-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def verify(root):
    spec = importlib.util.spec_from_file_location("focused_factorial_analysis_study", HERE / "study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plan = module.verify(root)
    primary = {name for name, value in plan["configs"].items() if not value.get("baseline", False)}
    if plan["schema"] != "synrfp-focused-factorial-v1" or plan["factors"] != LEVELS or \
            len(primary) != 72 or set(plan["configs"]) - primary != {"drfp"}:
        raise ValueError("Analysis requires the sealed72-cell crossing plus DRFP")
    profiles = {tuple(plan["configs"][name][factor] for factor in FACTORS) for name in primary}
    if profiles != set(itertools.product(*(LEVELS[factor] for factor in FACTORS))):
        raise ValueError("Primary configurations are not a complete crossing")
    actual = {"classification": len(plan["tasks"]["classification"]), "yield": len(plan["tasks"]["yield"]),
              "clustering": len(plan["clustering"]["tasks"])}
    if actual != COUNTS or plan["new_model_or_cluster_cells"] != 2555:
        raise ValueError("Analysis requires the registered2555-cell inventory")
    return plan, primary


def source_seal(root):
    expected = {"schema": SCHEMA, "plan_sha256": sha(root / "plan.json"),
                "analysis_source_sha256": sha(__file__), "levels": LEVELS, "references": REFERENCES,
                "metrics": METRICS, "change": "new descriptive analysis extension; sealed fitting sources unchanged"}
    path = root / "analysis_execution_seal.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Analysis source or design changed after binding")
    else:
        atomic_json(path, expected)
    return expected


def paired_effects(frame, primary_names, metrics):
    """Pure paired marginal contrasts; DRFP and other controls are excluded."""
    rows, refusals = [], []
    wanted = set(itertools.product(*(LEVELS[factor] for factor in FACTORS)))
    primary = frame[frame.fingerprint.isin(primary_names)]
    for (dataset, seed), group in primary.groupby(["dataset", "seed"], sort=True):
        profiles = set(map(tuple, group[FACTORS].to_numpy()))
        if len(group) != 72 or group.fingerprint.nunique() != 72 or profiles != wanted:
            raise ValueError("A paired split lacks the complete balanced72-cell grid")
        for metric in metrics:
            if not np.isfinite(group[metric].to_numpy(dtype=float)).all():
                refusals.append({"dataset": dataset, "seed": int(seed), "metric": metric,
                                 "reason": "Undefined/nonfinite metric prevents a complete-grid contrast"})
                continue
            for order in (1, 2):
                for axes in itertools.combinations(FACTORS, order):
                    choices = [[value for value in LEVELS[axis] if value != REFERENCES[axis]] for axis in axes]
                    for levels in itertools.product(*choices):
                        contrast = 0.0
                        for mask in itertools.product((0, 1), repeat=order):
                            selected = np.ones(len(group), dtype=bool)
                            for axis, level, use_level in zip(axes, levels, mask):
                                selected &= group[axis].to_numpy() == (level if use_level else REFERENCES[axis])
                            values = group.loc[selected, metric]
                            if len(values) != 72 // int(np.prod([len(LEVELS[axis]) for axis in axes])):
                                raise ValueError("Unbalanced marginal comparison")
                            contrast += (-1) ** (order - sum(mask)) * float(values.mean())
                        rows.append({"dataset": dataset, "seed": int(seed), "metric": metric, "order": order,
                                     "effect": ":".join(axes), "level": ":".join(map(str, levels)),
                                     "reference": ":".join(str(REFERENCES[axis]) for axis in axes), "contrast": contrast})
    return pd.DataFrame(rows), refusals


def summarize_effects(effects):
    keys = ["task", "dataset", "metric", "order", "effect", "level", "reference"]
    panel = effects.groupby(keys, sort=True).agg(paired_repeats=("contrast", "size"),
        mean=("contrast", "mean"), sample_std=("contrast", "std")).reset_index()
    panel["expected_paired_repeats"] = 5
    panel["complete"] = panel.paired_repeats == 5
    expected = effects.groupby("task").dataset.nunique().to_dict()
    family = panel[panel.complete].groupby([key for key in keys if key != "dataset"], sort=True).agg(
        completed_panels=("mean", "size"), equal_panel_mean=("mean", "mean"), panel_std=("mean", "std"),
        minimum_panel=("mean", "min"), maximum_panel=("mean", "max")).reset_index()
    family["expected_panels"] = family.task.map(expected)
    family["complete"] = family.completed_panels == family.expected_panels
    return panel, family


def completion_inputs(root, plan):
    definitions = {
        "classification": (root / "classification_COMPLETE.json", Path(plan["export"]), "classification.csv"),
        "yield": (root / "yield_canonical_REPORT_COMPLETE.json", Path(plan["yield"]["export"]), "yield.csv"),
        "clustering": (root / "clustering/COMPLETE.json", Path(plan["clustering"]["export"]), "cells.csv"),
    }
    if any(not path.exists() for path, _, _ in definitions.values()):
        return None
    inputs = {str(root / "plan.json"): sha(root / "plan.json")}
    for task, (path, export, filename) in definitions.items():
        value = json.loads(path.read_text())
        if value.get("plan_sha256") != sha(root / "plan.json"):
            raise ValueError("Completion receipt has a different plan: " + task)
        if task == "clustering":
            if value.get("cells") != COUNTS[task] or not all(value.get(key) is True for key in
                ("assignment_and_center_hashes_verified", "prediction_replay_exact", "metric_replay_verified")):
                raise ValueError("Clustering completion lacks the full canonical audit")
        else:
            metric_flag = "metrics_recomputed_from_canonical_predictions" if task == "classification" else "metrics_recomputed"
            if value.get("phase") != "complete" or value.get("completed") != COUNTS[task] or \
                    value.get("expected") != COUNTS[task] or value.get("artifacts_verified") is not True or value.get(metric_flag) is not True:
                raise ValueError("Model completion lacks the full canonical audit: " + task)
            for name, digest in value["reports_sha256"].items():
                if sha(export / name) != digest:
                    raise ValueError("Completed report bytes changed: " + task)
                inputs[str(export / name)] = digest
        inputs[str(path)] = sha(path)
        inputs[str(export / filename)] = sha(export / filename)
    return inputs


def frames(root, plan):
    rank, counts = {}, {}
    for split in plan["yield"]["splits"]:
        dataset = split["dataset"]
        rank[(dataset, split["split"])] = counts.get(dataset, 0)
        counts[dataset] = counts.get(dataset, 0) + 1
    if set(counts.values()) != {5}:
        raise ValueError("Yield panels do not have exactly five frozen splits")
    expected = {
        "classification": {(t["dataset"], t["config"], int(t["fold"])) for t in plan["tasks"]["classification"]},
        "yield": {(t["dataset"], t["config"], rank[(t["dataset"], t["split"])]) for t in plan["tasks"]["yield"]},
        "clustering": {(t["dataset"], t["fingerprint"], int(t["seed"])) for t in plan["clustering"]["tasks"]},
    }
    sources = {"classification": Path(plan["export"]) / "classification.csv",
               "yield": Path(plan["yield"]["export"]) / "yield.csv",
               "clustering": Path(plan["clustering"]["export"]) / "cells.csv"}
    out, source_hashes = {}, {}
    for task, path in sources.items():
        frame = pd.read_csv(path)
        if task == "classification":
            frame = frame.rename(columns={"config": "fingerprint", "fold": "seed"})
        elif task == "yield":
            frame = frame.rename(columns={"config": "fingerprint"})
            frame["seed"] = [rank[(row.dataset, row.split)] for row in frame.itertuples()]
        if not np.issubdtype(frame.seed.dtype, np.integer):
            raise ValueError("Replicate identities must be integral")
        identities = list(zip(frame.dataset, frame.fingerprint, frame.seed))
        if len(frame) != COUNTS[task] or len(set(identities)) != len(identities) or set(identities) != expected[task]:
            raise ValueError("Report inventory differs from canonical task identities: " + task)
        for field in FACTORS:
            frame[field] = [plan["configs"][fingerprint].get(field) for fingerprint in frame.fingerprint]
        # Bind reported metrics to the corresponding audited canonical record.
        for row in frame.itertuples():
            if task == "classification":
                record_path = root / "results/classification" / row.dataset / ("f%d" % row.seed) / (row.fingerprint + ".json")
            elif task == "yield":
                record_path = root / "results/yield" / row.dataset / row.split / (row.fingerprint + ".json")
            else:
                record_path = root / "clustering/cells" / row.dataset / row.fingerprint / (str(row.seed) + ".json")
            record = json.loads(record_path.read_text())
            digest = record.get("plan_sha256", record.get("identity", {}).get("plan_sha256"))
            if digest != sha(root / "plan.json"):
                raise ValueError("Canonical record has a different plan")
            for metric in METRICS[task]:
                observed, saved = getattr(row, metric), record[metric]
                if not ((pd.isna(observed) and saved is None) or np.isclose(observed, saved, atol=1e-12, rtol=1e-12, equal_nan=True)):
                    raise ValueError("Report metric differs from canonical record: " + metric)
            if task == "clustering":
                for suffix, field in ((".npz", "assignment_sha256"), (".model.npz", "model_sha256")):
                    artifact = record_path.with_suffix(suffix)
                    if sha(artifact) != record[field]:
                        raise ValueError("Completed clustering artifact changed")
                    source_hashes[str(artifact)] = record[field]
            source_hashes[str(record_path)] = sha(record_path)
        frame["task"] = task
        out[task] = frame
    return out, source_hashes


def report(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    plan, primary = verify(root)
    source_seal(root)
    inputs = completion_inputs(root, plan)
    if inputs is None:
        status = {"phase": "waiting_canonical_task_completions", "expected_cells": COUNTS,
                  "updated_utc": now(), "plan_sha256": sha(root / "plan.json")}
        atomic_json(root / "analysis_status.json", status)
        return status
    lock_path = root / "locks/analysis-report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        blocks, canonical_sources = frames(root, plan)
        effects, refusals, configuration_summaries = [], [], []
        for task, frame in blocks.items():
            contrasts, failed = paired_effects(frame, primary, METRICS[task])
            contrasts["task"] = task
            effects.append(contrasts)
            refusals.extend({"task": task, **failure} for failure in failed)
            for metric in METRICS[task]:
                summary = frame.groupby(["dataset", "fingerprint"], sort=True).agg(
                    repeats=(metric, "size"), mean=(metric, "mean"), sample_std=(metric, "std")).reset_index()
                summary["task"], summary["metric"] = task, metric
                summary["expected_repeats"], summary["complete"] = 5, summary.repeats == 5
                summary["factorial_cell"] = summary.fingerprint.isin(primary)
                configuration_summaries.append(summary)
        effects = pd.concat(effects, ignore_index=True)
        panel, family = summarize_effects(effects)
        export = Path(plan["export"]).parent / "analysis"
        export.mkdir(parents=True, exist_ok=True)
        outputs = {"paired_contrasts.csv": effects, "panel_contrasts.csv": panel,
                   "family_contrasts.csv": family,
                   "configuration_summary.csv": pd.concat(configuration_summaries, ignore_index=True)}
        for name, frame in outputs.items():
            atomic_csv(export / name, frame)
        atomic_json(export / "metric_refusals.json", refusals)
        protocol = {"schema": SCHEMA, "levels": LEVELS, "references": REFERENCES, "metrics": METRICS,
                    "repeat_mapping": "Classification original folds0..4; yield split order within dataset in sealed plan; clustering optimizer seeds0..4",
                    "contrast": "Reference-level marginal main effects and two-factor differences of differences within dataset/repeat; remaining crossed axes averaged equally",
                    "family_mean": "Equal weight per eligible dataset panel within each task; no pooling across different metrics/tasks",
                    "uncertainty": "Descriptive sample SD of five correlated repeats and between-panel SD; no p-values or confidence intervals",
                    "controls": "DRFP retained in configuration summary but excluded from all factor contrasts",
                    "metric_orientation": "Raw signed changes; positive favors higher metrics, negative favors error/time metrics",
                    "cw": "Primary unit support after full multiset algebra; not historical native weighted CW",
                    "scope": plan["scope"], "configuration_selection": None,
                    "analysis_source_sha256": sha(__file__), "plan_sha256": sha(root / "plan.json")}
        atomic_json(export / "analysis_protocol.json", protocol)
        verify(root)
        source_seal(root)
        current = completion_inputs(root, plan)
        if current != inputs or any(sha(path) != digest for path, digest in canonical_sources.items()):
            status = {"phase": "waiting_stable_completed_reports", "updated_utc": now(), "expected_cells": COUNTS}
            atomic_json(root / "analysis_status.json", status)
            return status
        input_receipt = {"schema": SCHEMA, "files_sha256": {**inputs, **canonical_sources},
                         "execution_seal_sha256": sha(root / "analysis_execution_seal.json")}
        atomic_json(export / "inputs.json", input_receipt)
        names = list(outputs) + ["metric_refusals.json", "analysis_protocol.json", "inputs.json"]
        completion = {"schema": SCHEMA, "phase": "complete", "finished_utc": now(),
                      "plan_sha256": sha(root / "plan.json"), "analysis_source_sha256": sha(__file__),
                      "canonical_cells": COUNTS, "total_canonical_cells": 2555,
                      "all_report_rows_match_canonical_model_or_cluster_receipts": True,
                      "all_three_task_canonical_completion_receipts_verified": True,
                      "primary_factorial_cells": 2520, "contextual_drfp_cells": 35,
                      "metric_refusals": len(refusals), "paired_contrast_rows": len(effects),
                      "reports_directory": str(export), "reports_sha256": {name: sha(export / name) for name in names},
                      "uncertainty": "Descriptive correlated-repeat SD only; no p-values/CI", "weights_exported": False}
        atomic_json(root / "ANALYSIS_COMPLETE.json", completion)
        atomic_json(root / "OVERALL_COMPLETE.json", completion)
        atomic_json(export / "completion.json", completion)
        atomic_json(root / "analysis_status.json", completion)
        return completion


def self_test():
    rows, primary = [], set()
    for dataset_index, dataset in enumerate(("small", "large")):
        for seed in range(5):
            for t, r, s, m in itertools.product(range(3), range(2), range(3), range(4)):
                name = f"cell-{t}-{r}-{s}-{m}"
                primary.add(name)
                rows.append({"dataset": dataset, "seed": seed, "fingerprint": name,
                    "tokenizer": LEVELS["tokenizer"][t], "radius": LEVELS["radius"][r],
                    "sketch": LEVELS["sketch"][s], "mode": LEVELS["mode"][m],
                    "score": 10 + (2+10*dataset_index)*t + 4*r + s + 5*m + 3*t*m + .2*seed})
            rows.append({"dataset": dataset, "seed": seed, "fingerprint": "drfp", "score": 1e9})
    frame = pd.DataFrame(rows)
    effects, failed = paired_effects(frame, primary, ["score"])
    assert not failed and len(effects) == 310
    def values(effect, level, dataset="small"):
        return effects.loc[(effects.effect == effect) & (effects.level == level) & (effects.dataset == dataset), "contrast"].to_numpy()
    np.testing.assert_allclose(values("tokenizer", "morgan"), 6.5, atol=1e-12)
    np.testing.assert_allclose(values("radius", "3"), 4, atol=1e-12)
    np.testing.assert_allclose(values("mode", "union"), 8, atol=1e-12)
    np.testing.assert_allclose(values("tokenizer:mode", "morgan:union"), 3, atol=1e-12)
    effects["task"] = "fixture"
    panel, family = summarize_effects(effects)
    selected = family[(family.effect == "tokenizer") & (family.level == "morgan")]
    np.testing.assert_allclose(selected.equal_panel_mean, 11.5, atol=1e-12)
    assert panel.complete.all() and family.complete.all() and not effects.effect.str.contains("drfp").any()
    try:
        paired_effects(frame.drop(frame.index[0]), primary, ["score"])
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete crossing was accepted")
    return {"self_test": "passed", "known_main_and_interaction_contrasts": True,
            "drfp_dummy_excluded": True, "incomplete_crossing_rejected": True, "equal_panel_mean_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()), flush=True)
        return
    while True:
        status = report(args.root)
        print(json.dumps(status), flush=True)
        if status["phase"] == "complete" or not args.watch or (args.root / "STOP").exists():
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
