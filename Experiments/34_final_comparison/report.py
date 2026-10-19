"""Versioned six-fingerprint consolidation with explicit incomplete components.

This file never trains, changes an original result, or selects a preset.  Its
seal inventories canonical metadata and the exact scientific construction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path.home() / ".local/share/synrfp-runs/final_comparison_v1"
EXPORT = HERE / "publication_metadata/final_comparison_v1"
SELECTED = "synrfp-wl-r2-b2048-parity-delta"
PRIMARY = "synrfp_selected"
BASELINES = ("drfp", "rdkit_struct", "rdkit_diff", "rxnfp_mlm", "crxnfp")
METHODS = (PRIMARY,) + BASELINES
EA_DATASETS = ("snar", "e2", "sn2", "e2sn2", "b97xd3", "rdb7", "rgd1")
EXPECTED = {"classification": 4200, "clustering": 1440, "yield": 384,
            "barrier": 1944, "feasibility": 30, "reagent": 60, "condition": 180}
METRICS = ("mcc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
           "ami", "ari", "nmi", "mcc_hungarian", "mae", "rmse", "r2",
           "auroc", "average_precision", "brier", "f1", "precision", "recall",
           "exact_set_accuracy", "micro_f1", "sample_jaccard",
           "top1_component_recall", "top3_component_recall", "top5_component_recall")
FROZEN = REPO / "Experiments/33_shared_preset_decision/publication_metadata/shared_preset_decision_v1/FROZEN_PRESET.json"
SOURCES = {
    "classification_primary": "12_validation/publication_metadata/classification_r2_2048_v1/cells.csv",
    "classification_reference": "12_validation/publication_metadata/crxnfp_full_v1/classification_cells.csv",
    "classification_plan": "12_validation/publication_metadata/classification_r2_2048_v1/plan.json",
    "crxnfp_plan": "12_validation/publication_metadata/crxnfp_full_v1/plan.json",
    "crxnfp_complete": "12_validation/publication_metadata/crxnfp_full_v1/status.json",
    "yield": "13_yield/publication_metadata/continuous_allfps_v1/results.csv",
    "yield_plan": "13_yield/publication_metadata/continuous_allfps_v1/plan.json",
    "barrier": "14_barrier/publication_metadata/barrier_allfps_v1/results.csv",
    "barrier_plan": "14_barrier/publication_metadata/barrier_allfps_v1/plan.json",
    "barrier_cv": "14_barrier/publication_metadata/barrier_allfps_v1/dual_cv_extension.json",
    "extension": "15_properties/publication_metadata/properties_allfps_v1/cells.csv",
    "extension_plan": "15_properties/publication_metadata/properties_allfps_v1/plan.json",
    "reagent": "23_reagent_prediction/publication_metadata/reagent_prediction_v1/cells.csv",
    "reagent_plan": "23_reagent_prediction/publication_metadata/reagent_prediction_v1/plan.json",
    "reagent_complete": "23_reagent_prediction/publication_metadata/reagent_prediction_v1/COMPLETE.json",
    "feasibility": "24_feasibility/publication_metadata/feasibility_frea_v1/expert_per_seed.csv",
    "feasibility_complete": "24_feasibility/publication_metadata/feasibility_frea_v1/AUDIT_COMPLETE.json",
    "feasibility_summary": "24_feasibility/publication_metadata/feasibility_frea_v1/plan_summary.json",
    "condition": "29_condition_prediction/publication_metadata/condition_prediction_v1/cells.csv",
    "condition_plan": "29_condition_prediction/publication_metadata/condition_prediction_v1/plan.json",
    "condition_complete": "29_condition_prediction/publication_metadata/condition_prediction_v1/COMPLETE.json",
    "condition_population": "29_condition_prediction/publication_metadata/condition_prediction_v1/common_population_audit.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_path(name):
    return REPO / "Experiments" / SOURCES[name]


def validate_preset(preset):
    need(preset.get("configuration_selected") == SELECTED, "Different frozen selection")
    expected = dict(tokenizer="wl", radius=2, sketch="parity", mode="delta", n_bits=2048, seed=1)
    cfg = preset.get("configuration", {})
    need(all(cfg.get(k) == v for k, v in expected.items()), "Changed selected fingerprint construction/width/sign")
    need(preset.get("interpretation") == "exploratory_after_prior_test_results_viewed", "Selection interpretation changed")
    need(preset.get("test_sets_independent") is False, "Selection cannot be called independent confirmation")


def metric_equal(a, b, metrics=METRICS):
    for k in metrics:
        if a.get(k, "") != "" and b.get(k, "") != "":
            need(math.isclose(float(a[k]), float(b[k]), rel_tol=1e-12, abs_tol=1e-12), "Canonical metric mismatch: " + k)


def id_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        need(key not in result, "Duplicate scientific cell: " + str(key))
        result[key] = row
    return result


def check_panel(rows, keys, expected=None, complete=True):
    """Check method-by-partition identities, never only cardinalities."""
    index = id_index(rows, tuple(keys) + ("method",))
    need(all(r["method"] in METHODS for r in rows), "Unapproved main fingerprint")
    signatures = {m: {k[:-1] for k in index if k[-1] == m} for m in METHODS}
    baseline = signatures[BASELINES[0]]
    for m in BASELINES:
        need(signatures[m] == baseline, "Mismatched baseline split/seed population: " + m)
    if complete:
        need(signatures[PRIMARY] == baseline, "Selected preset missing or different split/seed population")
    else:
        need(signatures[PRIMARY] <= baseline, "Selected preset has foreign split/seed population")
    if expected is not None:
        need(len(baseline) * len(METHODS) == expected, "Unexpected scientific population")
    return signatures


def add_identity(row, task, method, **fields):
    out = dict(row)
    out.update(task=task, method=method,
               configuration=SELECTED if method == PRIMARY else method,
               selection_interpretation="exploratory_after_prior_test_results_viewed")
    out.update(fields)
    return out


def normalize_native_method(method, primary):
    if method == primary:
        return PRIMARY
    return method if method in BASELINES else None


def pin_file(pins, path):
    path = Path(path).resolve()
    need(path.is_file(), "Missing provenance file: " + str(path))
    pins[str(path)] = sha(path)


def audit_json_csv(row, path, pins, fields=()):
    canonical = read_json(path)
    for key in fields:
        need(str(row[key]) == str(canonical[key]), "CSV/canonical identity mismatch: " + key)
    metric_equal(row, canonical)
    pin_file(pins, path)
    return canonical


def classification(pins):
    plan = read_json(source_path("classification_plan"))
    need(plan["fingerprint"] == SELECTED and plan["grouped_splits"] == 25, "Classification preset/split contract drift")
    ds = set(plan["datasets"])
    need(len(ds) == 14, "Classification endpoint coverage changed")
    primary = read_csv(source_path("classification_primary"))
    old = [r for r in read_csv(source_path("classification_reference")) if r["fingerprint"] in BASELINES]
    need(len(primary) == 700 and len(old) == 3500, "Classification canonical counts")
    croot = REPO / "Experiments/12_validation/runs/classification_r2_2048_v1/classification/03_classification/results/benchmark"
    xroot = REPO / "Experiments/12_validation/runs/crxnfp_full_v1/classification/03_classification/results/benchmark"
    refroot = Path(plan["reference_root"])
    splitpins = {Path(p).as_posix(): h for p, h in plan["pinned_files"].items() if "/work/splits/" in p}
    rows = []
    for row in primary + old:
        method = PRIMARY if row["fingerprint"] == SELECTED else row["fingerprint"]
        need(row["dataset"] in ds and row["model"] in ("knn", "mlp"), "Unexpected classification endpoint/model")
        seed = int(row["seed"])
        need(0 <= seed < 25, "Classification split seed outside 5x5")
        base = croot if method == PRIMARY else xroot if method == "crxnfp" else refroot
        path = base / row["dataset"] / ("cv_s" + str(seed)) / (row["fingerprint"] + "__" + row["model"] + ".json")
        native = audit_json_csv(row, path, pins, ("dataset", "fingerprint", "model", "seed"))
        need(native.get("scheme") == "cv", "Classification scheme is not CV")
        if row["model"] == "mlp":
            need(native.get("backend") == "torch_cuda", "Classification production MLP must be CUDA")
            need(native.get("protocol") == "sklearn_adam_fp32_v1", "Classification numerical MLP recipe drift")
        matches = [(p, h) for p, h in splitpins.items() if p.endswith("/" + row["dataset"] + "/cv_s" + str(seed) + ".npy")]
        need(len(matches) == 1, "No unique frozen classification split")
        path_split, splitsha = matches[0]
        if native.get("split_sha256"):
            need(native["split_sha256"] == splitsha, "Classification split certificate mismatch")
        pin_file(pins, path_split)
        need(pins[str(Path(path_split).resolve())] == splitsha, "Changed original classification split")
        out = add_identity(row, "classification", method, protocol="grouped_5_repeats_5_folds",
                           population="original_all_rows", repeat=seed // 5, fold=seed % 5,
                           seed=seed, split_sha256=splitsha, source_result_path=str(path.resolve()), source_result_sha256=sha(path))
        rows.append(out)
    check_panel(rows, ("dataset", "model", "seed"), EXPECTED["classification"])
    for group in group_rows(rows, ("dataset", "model", "seed")).values():
        need(len({(int(r["n_train"]), int(r["n_test"])) for r in group}) == 1, "Classification row counts differ across fingerprints")
    return rows


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return groups


def extension(pins):
    plan = read_json(source_path("extension_plan"))
    root = REPO / "Experiments/15_properties/runs/properties_allfps_v1"
    planhash = sha(root / "plan.json")
    need(plan["synrfp"].get("synrfp_r2b2048") == "WL r2 parity delta 2048 bits seed1", "Extension selected construction drift")
    rows = []
    for row in read_csv(source_path("extension")):
        if row["dataset"] not in EA_DATASETS or row["target"] != "ea":
            continue
        method = normalize_native_method(row["method"], "synrfp_r2b2048")
        if method is None:
            continue
        path = root / "models" / row["dataset"] / "ea" / row["split"] / row["method"] / "result.json"
        value = audit_json_csv(row, path, pins, ("dataset", "target", "split", "method"))
        need(value["plan_sha256"] == planhash and row["plan_sha256"] == planhash, "Extension source plan changed")
        need(str(value.get("physical_gpu")) in ("0", "1"), "Extension CUDA execution missing")
        split = next(s for s in plan["splits"] if s["dataset"] == row["dataset"] and s["split"] == row["split"])
        need(sha(split["path"]) == row["split_sha256"], "Extension split changed")
        pin_file(pins, split["path"])
        out = add_identity(row, "barrier", method, source_panel="extension15", population="original_all_rows",
                           fraction=1.0, report_group=row["protocol"], source_result_path=str(path.resolve()), source_result_sha256=sha(path))
        rows.append(out)
    need(len(rows) == 1284, "Extension seven-EA count mismatch")
    check_panel(rows, ("dataset", "target", "split"), 1284)
    return rows


def auxiliary(pins):
    rows = {}
    for task, primary, count in (("feasibility", "synrfp_wl_r2_delta", 30), ("reagent", "synrfp_delta_r2b2048", 60), ("condition", "synrfp_wl_r2_delta", 180)):
        result = []
        local = Path.home() / ".local/share/synrfp-runs" / {"feasibility": "feasibility_frea_v1", "reagent": "reagent_prediction_v1", "condition": "condition_prediction_v1"}[task]
        native_plan = read_json(local / "plan.json")
        pin_file(pins, local / "plan.json")
        native_plan_sha = sha(local / "plan.json")
        if task != "reagent":
            cfg = native_plan["configs"][primary]
            need(all(cfg.get(k) == v for k, v in dict(tokenizer="wl", radius=2, sketch="parity", mode="delta", n_bits=2048, seed=1).items()), "Auxiliary primary construction drift")
        native_index = {}
        for path in (local / "models").glob("*/result.json"):
            r = read_json(path)
            ident = (r["config"], str(r["seed"])) if task == "feasibility" else (r["method"], r["split"]["name"]) if task == "reagent" else (r["config"], r["split"]["name"])
            need(ident not in native_index, "Duplicate auxiliary canonical result")
            native_index[ident] = (path, r)
        for row in read_csv(source_path(task)):
            raw_method = row["method"] if task == "reagent" else row["config"]
            method = normalize_native_method(raw_method, primary)
            if method is None:
                continue
            ident = (raw_method, row["seed"]) if task == "feasibility" else (raw_method, row["split"])
            need(ident in native_index, "Missing auxiliary canonical fit")
            path, record = native_index[ident]
            need(record["phase"] == "complete", "Auxiliary fit incomplete")
            identity = record.get("identity", record)
            need(identity.get("plan_sha256") == native_plan_sha, "Auxiliary source plan mismatch")
            if task != "feasibility":
                need(str(record["split"]["seed"]) == row["seed"], "Auxiliary optimizer seed mismatch")
                need(record["split"]["protocol"] == (row["protocol"] if task == "reagent" else row["panel"]), "Auxiliary split protocol mismatch")
                pin_file(pins, record["split"]["path"])
            else:
                need(int(row["seed"]) in range(5), "FREA seed outside frozen initialization panel")
            actual = record["expert_metrics"] if task == "feasibility" else record["test"] if task == "reagent" else record["metrics"]
            metric_equal(row, actual)
            pin_file(pins, path)
            training = record.get("training", {})
            if task != "reagent":
                need(training.get("execution_device") == "cuda", "Auxiliary CUDA execution mismatch")
            else:
                need(str(record["physical_gpu"]) in ("0", "1"), "Reagent physical CUDA device missing")
            protocol = "external_expert_fixed_test" if task == "feasibility" else row["protocol"] if task == "reagent" else row["panel"]
            out = add_identity(row, task, method, dataset={"feasibility": "frea_expert", "reagent": "uspto500_reagent", "condition": "uspto_condition"}[task],
                               protocol=protocol, population="original_common_rows", source_result_path=str(path), source_result_sha256=sha(path))
            if task == "feasibility":
                out.update(split="expert_fixed", seed=int(row["seed"]))
            result.append(out)
        need(len(result) == count, "Auxiliary panel count mismatch: " + task)
        check_panel(result, ("dataset", "protocol", "split", "seed"), count)
        if task == "condition":
            complete = read_json(source_path("condition_complete"))
            for fn, digest in complete["outputs_sha256"].items():
                need(sha(source_path("condition").parent / fn) == digest, "Condition report hash drift")
        rows[task] = result
    return rows


def regression_baselines(pins, task):
    root = REPO / "Experiments" / ("13_yield/runs/continuous_allfps_v1" if task == "yield" else "14_barrier/runs/barrier_allfps_v1")
    plan = read_json(root / "plan.json")
    planhash = sha(root / "plan.json")
    rows = []
    for row in read_csv(source_path(task)):
        method = row.get("fingerprint", row.get("method"))
        if method not in BASELINES:
            continue
        fraction = float(row.get("fraction", 1))
        path = root / "models" / row["dataset"] / (row["split"] if task == "yield" else row["split"] + "_f" + str(round(fraction * 100))) / method / "result.json"
        fields = ("dataset", "split", "fingerprint") if task == "yield" else ("dataset", "split", "method")
        native = audit_json_csv(row, path, pins, fields)
        need(native["plan_sha256"] == planhash and row["plan_sha256"] == planhash, "Regression native plan drift")
        need(row.get("split_sha256") == native.get("split_sha256"), "Regression split identity drift")
        # Native result includes artifact hashes; preserve the canonical result digest and the catalog.
        need(isinstance(native.get("artifacts"), dict) and bool(native["artifacts"]), "Regression retained artifact catalog missing")
        splitpath = path.parent / "rows.npz"
        need(sha(splitpath) == native["artifacts"]["rows.npz"], "Regression native row selectors changed")
        split_catalog = plan["splits"] + (read_json(source_path("barrier_cv"))["extra_splits"] if task == "barrier" else [])
        original = next(s for s in split_catalog if s["dataset"] == row["dataset"] and s["split"] == row["split"])
        need(sha(original["path"]) == native["split_sha256"], "Regression original split changed")
        pin_file(pins, original["path"])
        pin_file(pins, splitpath)
        trainingpath = path.parent / "training.json"
        training = read_json(trainingpath)
        need(training.get("device", training.get("execution_device", "cuda")).startswith("cuda"), "Regression learner not CUDA")
        pin_file(pins, trainingpath)
        out = add_identity(row, task, method, fraction=fraction, protocol=row["report_group"], population="original_common_rows",
                           source_panel="yield13" if task == "yield" else "barrier14", source_result_path=str(path.resolve()), source_result_sha256=sha(path))
        rows.append(out)
    expected = 320 if task == "yield" else 550
    need(len(rows) == expected, "Regression baseline count mismatch")
    check_panel(rows, ("dataset", "split", "fraction"), expected // 5 * 6, complete=False)
    return rows


def stable_reuse(pins):
    rows = {"classification": classification(pins), "barrier": extension(pins)}
    rows.update(auxiliary(pins))
    rows["yield"] = regression_baselines(pins, "yield")
    rows["barrier"].extend(regression_baselines(pins, "barrier"))
    return rows


def prepare(root=ROOT):
    root = Path(root)
    path = root / "report/plan.json"
    if path.exists():
        return verify(root)
    validate_preset(read_json(FROZEN))
    pins = {}
    for file in (FROZEN, HERE / "report.py", HERE / "test_report.py", HERE / "PROTOCOL.md"):
        pin_file(pins, file)
    for name in SOURCES:
        pin_file(pins, source_path(name))
    for relative in ("Experiments/13_yield/cuda_mlp.py", "Experiments/23_reagent_prediction/worker.py",
                     "Experiments/24_feasibility/mlp_worker.py", "Experiments/29_condition_prediction/worker.py",
                     "Experiments/15_properties/run_properties.py", "Experiments/03_classification/mlp_cuda_worker.py",
                     "Experiments/synrfp_exp/metrics.py",
                     "Experiments/synrfp_exp/splits.py"):
        pin_file(pins, REPO / relative)
    rows = stable_reuse(pins)
    plan = dict(schema="synrfp-final-six-fingerprint-comparison-v1", created_utc=now(),
                selected=SELECTED, methods=list(METHODS), expected=EXPECTED,
                expected_total=sum(EXPECTED.values()), reuse_counts={k: len(v) for k, v in rows.items()},
                source_files_sha256=pins, selected_preset_sha256=sha(FROZEN),
                component_contracts={"regression": {"selected_cells": 174, "yield": 64, "barrier": 110},
                                     "clustering": {"selected_cells": 240, "baseline_cells": 1200}},
                selection_interpretation="exploratory_after_prior_test_results_viewed", public_api_default_changed=False,
                statistical_claim="Descriptive protocol-specific SD; correlated folds are not independent tests.")
    write_json(path, plan)
    write_json(root / "report/plan_seal.json", dict(sha256=sha(path)))
    write_json(EXPORT / "report_plan.json", plan)
    return plan


def verify(root=ROOT):
    root = Path(root)
    path = root / "report/plan.json"
    plan = read_json(path)
    need(sha(path) == read_json(root / "report/plan_seal.json")["sha256"], "Final report seal changed")
    need(plan["selected"] == SELECTED and tuple(plan["methods"]) == METHODS and plan["expected"] == EXPECTED, "Final report scientific scope changed")
    validate_preset(read_json(FROZEN))
    for path, digest in plan["source_files_sha256"].items():
        need(sha(path) == digest, "Changed reused provenance: " + path)
    return plan


def receipt_outputs(receipt, directory):
    outputs = receipt.get("outputs_sha256", receipt.get("files_sha256", {}))
    need(isinstance(outputs, dict) and outputs, "Completed component has no output hash catalog")
    for name, digest in outputs.items():
        path = Path(name) if Path(name).is_absolute() else directory / name
        need(sha(path) == digest, "Completed component output changed: " + str(path))
    return outputs


def components(root):
    """Only admitted completed components enter the primary comparison."""
    added = defaultdict(list)
    states = {}
    for component, filename in (("regression", "COMPLETE.json"), ("clustering", "final_component.json")):
        directory = EXPORT / component
        receiptpath = directory / filename
        if not receiptpath.exists():
            states[component] = dict(phase="pending", expected_selected=174 if component == "regression" else 240)
            continue
        receipt = read_json(receiptpath)
        need(receipt.get("phase") == "complete", "Noncomplete component receipt")
        outputs = receipt_outputs(receipt, directory)
        if component == "regression":
            need(receipt.get("full_artifact_audit") is True, "Regression component lacks full artifact closure")
        planpath = directory / "plan.json"
        need(planpath.exists(), "Completed component lacks plan")
        ph = sha(planpath)
        if receipt.get("plan_sha256"):
            need(receipt["plan_sha256"] == ph, "Component plan hash mismatch")
        plan = read_json(planpath)
        # Every component explicitly binds the same frozen selection or full config name.
        blob = json.dumps(plan, sort_keys=True)
        need(SELECTED in blob and (sha(FROZEN) in blob or str(FROZEN) in blob), "Component not bound to selected preset")
        csvpath = directory / ("results.csv" if component == "regression" else "cells.csv")
        need(csvpath.name in outputs or str(csvpath) in outputs, "Component cells not covered by completion hash")
        rows = read_csv(csvpath)
        expected = 174 if component == "regression" else 240
        need(len(rows) == expected, "Component primary cell count mismatch")
        for row in rows:
            rawmethod = row["method"]
            need(rawmethod in (PRIMARY, SELECTED), "Component contains a nonselected primary")
            if row.get("config"):
                need(row["config"] == SELECTED, "Component config mismatch")
            task = row["task"]
            need(task in (("yield", "barrier") if component == "regression" else ("clustering",)), "Component task mismatch")
            if component == "regression":
                need(row["execution_device"] == "cuda", "Fresh regression result is not CUDA")
            result_path = row.get("result_path")
            need(bool(result_path) and Path(result_path).is_file(), "Component lacks canonical result path")
            actual_result_sha = sha(result_path)
            recorded_sha = row.get("result_sha256")
            if not recorded_sha:
                catalog = receipt.get("result_sha256", {})
                recorded_sha = catalog.get(result_path, catalog.get(str(Path(result_path).resolve()), catalog.get(str(Path(result_path).parent.name))))
            need(recorded_sha == actual_result_sha, "Component canonical result changed or is not covered by its receipt")
            out = add_identity(row, task, PRIMARY, population=row.get("population", "original_common_rows"))
            if component == "regression":
                out.update(fraction=float(row["fraction"]), protocol=row["report_group"], source_panel="yield13" if task == "yield" else "barrier14")
            added[task].append(out)
        if component == "regression":
            need(Counter(r["task"] for r in rows) == {"yield": 64, "barrier": 110}, "Regression source coverage mismatch")
        else:
            baselinepath = directory / "baseline_cells.csv"
            need(baselinepath.name in outputs or str(baselinepath) in outputs, "Cluster baselines not covered by completion hash")
            baselines = read_csv(baselinepath)
            need(len(baselines) == 1200 and all(r["method"] in BASELINES for r in baselines), "Cluster baseline scope mismatch")
            added["clustering"].extend(add_identity(r, "clustering", r["method"]) for r in baselines)
        states[component] = dict(phase="complete", receipt_path=str(receiptpath), receipt_sha256=sha(receiptpath), plan_sha256=ph,
                                 cells_sha256=sha(csvpath), selected_cells=len(rows))
    return added, states


def repeated_index(row):
    split = str(row.get("split", ""))
    match = re.fullmatch(r"cv_r(\d+)_f(\d+)", split)
    if match:
        return int(match[1]), int(match[2])
    if row["task"] == "classification":
        return int(row["seed"]) // 5, int(row["seed"]) % 5
    if row["task"] == "yield" and row["dataset"] == "ncs_bh":
        match = re.fullmatch(r"i(\d+)_f(\d+)", split)
        need(match is not None, "Unknown NCS iteration/fold identity")
        return int(match[1]), int(match[2])
    return None


def summarize(rows):
    keys = ("task", "source_panel", "dataset", "target", "model", "method", "protocol", "population", "fraction", "report_group")
    result = []
    for key, group in sorted(group_rows(rows, keys).items()):
        base = dict(zip(keys, key))
        base.update(n_cells=len(group), configuration=SELECTED if base["method"] == PRIMARY else base["method"])
        mode = "across_recorded_partitions"
        blocks = None
        if base["task"] == "condition" and base["protocol"] == "group_disjoint":
            blocks = group_rows(group, ("fold",))
            need(len(blocks) == 5 and all(len(g) == 5 for g in blocks.values()), "Condition grouped fold/seed aggregation mismatch")
            mode = "across_five_fold_means_over_five_optimizer_seeds"
        elif all(repeated_index(r) is not None for r in group):
            blocks = defaultdict(list)
            for r in group:
                blocks[repeated_index(r)[0]].append(r)
            mode = "across_repeat_means" if base["task"] != "yield" else "across_ten_iteration_means"
        elif base["task"] in ("condition", "reagent", "feasibility") and base["protocol"] != "group_disjoint":
            mode = "optimizer_seed_variation_on_one_fixed_test"
        elif base["task"] == "clustering":
            mode = "initialization_seed_variation_at_fixed_known_k"
        base.update(sd_scope=mode, n_summary_units=len(blocks) if blocks is not None else len(group), independent_test_replicates=False)
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r.get(metric, "") != ""]
            if not values:
                continue
            need(len(values) == len(group) and all(math.isfinite(v) for v in values), "Missing/nonfinite metric within panel")
            units = ([statistics.mean(float(r[metric]) for r in g) for g in blocks.values()] if blocks is not None else values)
            base[metric + "_mean"] = statistics.mean(units)
            base[metric + "_sd"] = statistics.stdev(units) if len(units) > 1 else ""
            base[metric + "_raw_cell_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        result.append(base)
    return result


def validate_joins(rows, states):
    check_panel(rows["classification"], ("dataset", "model", "seed"), 4200)
    for task, expected in (("feasibility", 30), ("reagent", 60), ("condition", 180)):
        check_panel(rows[task], ("dataset", "protocol", "split", "seed"), expected)
    check_panel(rows["yield"], ("dataset", "split", "fraction"), 384, complete=states["regression"]["phase"] == "complete")
    # Separate 14/15 sources because several substrate collections are related.
    for panel, expected in (("barrier14", 660), ("extension15", 1284)):
        part = [r for r in rows["barrier"] if r["source_panel"] == panel]
        check_panel(part, ("dataset", "target", "split", "fraction"), expected,
                    complete=panel == "extension15" or states["regression"]["phase"] == "complete")
    if states["clustering"]["phase"] == "complete":
        check_panel(rows["clustering"], ("dataset", "population", "seed"), 1440)
    for task in ("yield", "barrier"):
        for group in group_rows(rows[task], ("source_panel", "dataset", "target", "split", "fraction")).values():
            signatures = {(str(r.get("ntrain", "")), str(r.get("nval", "")), str(r.get("ntest", "")), r.get("split_sha256", "")) for r in group}
            need(len(signatures) == 1, "Regression methods do not share original row selectors/population")


def context_inventory():
    candidates = {
        "yield_learned_references": "Experiments/17_yield_baselines/publication_metadata/matched_full_v3/unattended_v4/cells.csv",
        "syncat_classification": "Experiments/18_model_references/publication_metadata/syncat_reference_v1",
        "rxnfp_supervised_yield": str(source_path("yield").relative_to(REPO)),
        "dmpnn_barrier_and_supervised_rxnfp": str(source_path("barrier").relative_to(REPO)),
        "supervised_rxnfp_properties": str(source_path("extension").relative_to(REPO)),
    }
    records = []
    for label, relative in candidates.items():
        path = REPO / relative
        records.append(dict(reference=label, path=str(path), available=path.exists(), included_in_six_fp_aggregate=False,
                            sha256=sha(path) if path.is_file() else None))
    return records


def once(root=ROOT):
    root = Path(root)
    plan = verify(root)
    # No repeated artifact hashing is needed: verify checks every pinned canonical metadata digest.
    rows = stable_reuse({})
    added, states = components(root)
    rows.setdefault("clustering", [])
    for task, records in added.items():
        rows.setdefault(task, []).extend(records)
    validate_joins(rows, states)
    outputs = {}
    status = {}
    for task, expected in EXPECTED.items():
        cells = rows.get(task, [])
        path = EXPORT / (task + "_cells.csv")
        write_csv(path, cells)
        outputs[path.name] = sha(path)
        summary = EXPORT / (task + "_summary.csv")
        write_csv(summary, summarize(cells))
        outputs[summary.name] = sha(summary)
        status[task] = dict(completed_cells=len(cells), expected_cells=expected, complete=len(cells) == expected)
    write_json(EXPORT / "reference_inventory.json", context_inventory())
    outputs["reference_inventory.json"] = sha(EXPORT / "reference_inventory.json")
    complete = all(v["complete"] for v in status.values()) and all(s["phase"] == "complete" for s in states.values())
    text = ["# Final shared-preset fingerprint comparison", "", "Status: " + ("COMPLETE" if complete else "PARTIAL — replacement fits remain outstanding"), "",
            "Primary construction: WL, radius 2, 2,048 total bits, parity sketch, signed delta, fingerprint seed 1.", "",
            "Six fingerprints: selected SynRFP, DRFP, RDKit structural, RDKit difference, RXNFP-MLM and crxnfp. The historical 1,024-bit SynRFP is excluded.", "",
            "| Task | Audited cells | Expected |", "|---|---:|---:|"]
    text.extend("| " + t + " | " + str(v["completed_cells"]) + " | " + str(v["expected_cells"]) + " |" for t, v in status.items())
    text.extend(["", "Classification includes both kNN and MLP on the original five repetitions of five grouped folds. Clustering here is known-K K-means with ten initializations, separately reporting all rows and the original common valid population. It does not replace the label-blind unknown-K analyses.", "",
                 "Yield keeps original published random, OOD and clustered-OOD partitions. Barrier keeps random CV, learning fractions, grouped folds and grouped holdouts separate; extension15 adds seven activation-energy endpoints. GDB7/RDB7 and E2/SN2/combined source families are related and are not independent datasets.", "",
                 "Reagent uses five native initializations and one grouped five-fold evaluation. Condition uses one fixed grouped five-fold partition with five optimizer seeds per fold, plus five native initializations. Feasibility uses five training seeds on one fixed external expert test. None of these is described as five redrawn CV repetitions.", "",
                 "Summary SD uses repeat means for repeated CV; condition grouped SD uses fold means over optimizer seeds. Raw-cell SD is retained separately. These are descriptive dispersions, not confidence intervals or independent-fold significance tests.", "",
                 "Selection is exploratory: these test outcomes were already viewed when the common preset was chosen. Re-fitting with new seeds does not remove selection bias. This report makes no SOTA or untouched-confirmation claim. Metrics remain separate by task and protocol; MCC and R² are not averaged together.", "",
                 "Learned yield references, supervised RXNFP, D-MPNN and SynCat remain separately labelled context in reference_inventory.json. The completed queue does not include newly adapted feasibility SynCat, Parrot, T5Chem or a full SynCluster comparison."])
    (EXPORT / "report.md").write_text("\n".join(text) + "\n")
    outputs["report.md"] = sha(EXPORT / "report.md")
    receipt = dict(schema="synrfp-final-comparison-report-v1", phase="complete" if complete else "partial", updated_utc=now(),
                   plan_sha256=sha(root / "report/plan.json"), selected_preset_sha256=plan["selected_preset_sha256"],
                   configuration_selected=SELECTED, tasks=status, components=states, outputs_sha256=outputs,
                   total_completed=sum(v["completed_cells"] for v in status.values()), total_expected=sum(EXPECTED.values()),
                   logical_cells=sum(v["completed_cells"] for v in status.values()),
                   selection_interpretation="exploratory_after_prior_test_results_viewed", independent_confirmation=False)
    write_json(root / "report/status.json", receipt)
    write_json(EXPORT / "status.json", receipt)
    if complete:
        write_json(root / "report/COMPLETE.json", receipt)
        write_json(root / "COMPLETE.json", receipt)
        write_json(EXPORT / "COMPLETE.json", receipt)
    else:
        need(not (root / "report/COMPLETE.json").exists() and not (EXPORT / "COMPLETE.json").exists(), "Previously complete report lost a required component")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--once", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root) if args.prepare else once(args.root)
    print(json.dumps({k: result[k] for k in ("phase", "total_completed", "total_expected", "selected") if k in result}))


if __name__ == "__main__":
    main()
