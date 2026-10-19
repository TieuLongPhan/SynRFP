#!/usr/bin/env python
"""Frozen shared-preset completion of the native full-row clustering panel.

No GPU or baseline fitting. Exact native endpoint order, known K, preprocessing,
algorithm settings and hierarchical scoring are retained. The 240 logical cells
require 200 fits: 40 shared-population cells copy a verified identical all-row fit.
Use the native ``synrfp`` conda environment, not the base Python environment.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
V12 = EXP / "12_validation"
for directory in (REPO, EXP, V12):
    sys.path.insert(0, str(directory))

from synrfp_exp.config import SynRFPConfig, load_config
from synrfp_exp.data import dataset_table, reaction_table, rxn_key
from synrfp_exp.features import Features, _synrfp_chunk, load

spec = importlib.util.spec_from_file_location("final_native_clustering", V12 / "07_clustering.py")
NATIVE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NATIVE)

ROOT = Path.home() / ".local/share/synrfp-runs/final_comparison_v1/clustering"
EXPORT = HERE / "publication_metadata/final_comparison_v1/clustering"
OLD = (V12 / "runs/crxnfp_full_v1").resolve()
WIDE = (V12 / "runs/classification_r2_2048_v1").resolve()
MAIN = (V12 / "runs/clustering/n1/all_rows").resolve()
PRESET = EXP / "33_shared_preset_decision/publication_metadata/shared_preset_decision_v1"
FP = "synrfp-wl-r2-b2048-parity-delta"
LEGACY = "synrfp-wl-r2-b1024-parity-delta"
BASELINES = ["drfp", "rdkit_struct", "rdkit_diff", "rxnfp_mlm", "crxnfp"]
POLICIES = ["all_rows", "shared_valid_all_six"]
ENDPOINTS = NATIVE.SMALL + NATIVE.LARGE
PROTOCOL = "known_k_10_seeds"
_HASH_CACHE = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path, fresh=False):
    path = Path(path)
    st = path.stat()
    stamp = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if fresh or stamp not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        _HASH_CACHE[stamp] = h.hexdigest()
    return _HASH_CACHE[stamp]


def write(path, value):
    NATIVE.atomic_json(Path(path), value)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def check_pins(pins, fresh=False):
    for path, expected in pins.items():
        if sha(path, fresh=fresh) != expected:
            raise ValueError("Changed frozen source/input: " + path)


def environment():
    return {"numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def check_environment(expected):
    current = environment()
    if any(current[key] != expected[key] for key in current):
        raise ValueError(f"Use native synrfp conda environment: expected {expected}, got {current}")


def feature(path):
    with np.load(path, allow_pickle=False) as z:
        return Features(z["X"], int(z["n_bits"]), z["ok"])


def expected_tasks(panels):
    return [{"dataset": ds, "population": population, "seed": seed,
             "origin": "population_identical_copy" if population != "all_rows" and panels[ds]["shared_is_all"] else "fresh"}
            for ds in ENDPOINTS for population in POLICIES for seed in range(10)]


def identity(task):
    return task["dataset"], task["population"], int(task["seed"])


def stem(task):
    return ROOT / "cells" / task["dataset"] / task["population"] / f"s{task['seed']}"


def used_mask(panel, population):
    with np.load(panel["population_file"], allow_pickle=False) as z:
        return z["all_rows" if population == "all_rows" else "shared_valid"].copy()


def validate_assignment(path, ds, used, valid):
    with np.load(path, allow_pickle=False) as z:
        required = {"labels", "used", "valid", "source_row", "r_id", "y"}
        if not required <= set(z.files):
            raise ValueError("Missing assignment fields: " + str(path))
        if not np.array_equal(z["used"], used) or not np.array_equal(z["valid"], valid):
            raise ValueError("Assignment population/encoding mask differs")
        if not np.array_equal(z["source_row"], ds.row.to_numpy(np.int64)):
            raise ValueError("Assignment source row order differs")
        if not np.array_equal(z["r_id"].astype(str), ds.r_id.astype(str).to_numpy()):
            raise ValueError("Assignment record identity order differs")
        if not np.array_equal(z["y"], ds.y.to_numpy()):
            raise ValueError("Assignment endpoint labels differ")
        labels = z["labels"].copy()
        if len(labels) != len(ds) or np.any(labels[used] < 0) or np.any(labels[~used] != -1):
            raise ValueError("Assignment labels do not match population")
    return labels


def native_reference(ds, fp, population, seed):
    if population != "all_rows" or fp == "crxnfp":
        return OLD / "clustering" / population / ds / f"{fp}__s{seed}.json"
    method = "kmeans" if ds in NATIVE.SMALL else "minibatch_kmeans"
    return MAIN / ds / fp / f"{method}__s{seed}.json"


def validate_native_record(rec, ds, fp, population, seed, panel):
    method = "kmeans" if ds in NATIVE.SMALL else "minibatch_kmeans"
    for key, value in {"dataset": ds, "fingerprint": fp, "seed": seed,
                       "method": method, "params": panel["params"], "k": panel["k"],
                       "n_rows": panel["n_rows"]}.items():
        if rec.get(key) != value:
            raise ValueError(f"Native reference {key} differs: {ds}/{fp}/{population}/{seed}")
    if rec.get("row_policy") != population or rec.get("input_policy") != "n1":
        raise ValueError("Native reference population/input policy differs")
    if "baseline_feature_sha256" in panel:
        expected_feature = panel["baseline_feature_sha256"][fp]
        if "feature_sha256" in rec:
            if rec["feature_sha256"] != expected_feature or rec.get("dataset_sha256") != panel["dataset_sha256"]:
                raise ValueError("Native feature/dataset receipt differs")
        else:
            native_sources = rec["source_sha256"]
            feature_matches = [value for key, value in native_sources.items() if key.endswith(f"/features/{panel['source']}/{fp}.npz")]
            dataset_matches = [value for key, value in native_sources.items() if key.endswith(f"/tables/ds_{ds}.pkl")]
            if feature_matches != [expected_feature] or dataset_matches != [panel["dataset_sha256"]]:
                raise ValueError("Native original feature/dataset bindings differ")


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "plan.json").exists():
        return verify(fresh=True)
    preset = json.loads((PRESET / "FROZEN_PRESET.json").read_text())
    if preset["configuration_selected"] != FP:
        raise ValueError("Selected shared preset differs")
    decision = json.loads((PRESET / "COMPLETE.json").read_text())
    check_pins(decision["runtime_bindings"], fresh=True)
    check_pins(decision["input_sha256"], fresh=True)
    check_pins({str(PRESET / key): value for key, value in decision["outputs_sha256"].items()}, fresh=True)
    native_protocol = json.loads((MAIN / "protocol.json").read_text())
    native_versions = {key: native_protocol["environment"]["packages"][key] for key in environment()}
    check_environment(native_versions)
    cfg = load_config(OLD / "config.yaml")
    wide_plan = json.loads((WIDE / "plan.json").read_text())
    if sha(WIDE / "plan.json") != json.loads((WIDE / "plan_seal.json").read_text())["sha256"]:
        raise ValueError("Changed native wide feature plan")
    producer_pins = {path: digest for path, digest in wide_plan["pinned_files"].items()
                     if "/synrfp/" in path or path.endswith("/synrfp_exp/features.py")
                     or path == str(OLD / "config.yaml")}
    check_pins(producer_pins, fresh=True)
    if cfg["fp_seed"] != preset["configuration"]["seed"] or cfg["synrfp"]["bbit"] != preset["configuration"]["bbit"]:
        raise ValueError("Frozen fingerprint seed/sketch construction differs")
    pins = dict(decision["runtime_bindings"])
    pins.update(decision["input_sha256"])
    def pin(path):
        path = Path(path).resolve()
        pins[str(path)] = sha(path)
    for path in [Path(__file__), HERE / "test_clustering.py", OLD / "config.yaml", OLD / "plan.json",
                 OLD / "COMPLETE.json", WIDE / "plan.json", WIDE / "plan_seal.json",
                 MAIN / "protocol.json", cfg["paths"]["work"] / "tables/SOURCE.json",
                 V12 / "07_clustering.py", V12 / "run_crxnfp_full.py",
                 V12 / "run_classification_r2_wide.py", V12 / "study.yaml",
                 PRESET / "FROZEN_PRESET.json", PRESET / "COMPLETE.json"]:
        pin(path)
    for key in decision["outputs_sha256"]:
        pin(PRESET / key)
    for path in (REPO / "synrfp").rglob("*.py"):
        pin(path)
    for name in ("features.py", "config.py", "metrics.py", "data.py"):
        pin(EXP / "synrfp_exp" / name)
    sources = {}
    for key, original in wide_plan["sources"].items():
        path = WIDE / "features" / key / (FP + ".npz")
        receipt = path.with_suffix(".json")
        meta = json.loads(receipt.read_text())
        if meta["feature_sha256"] != sha(path) or meta["plan_sha256"] != sha(WIDE / "plan.json"):
            raise ValueError("Changed native 2048-bit feature receipt")
        if not meta["baseline_bitwise_equivalent"]:
            raise ValueError("Missing original construction equivalence")
        for filename in (path, receipt, original["table"], original["baseline"]):
            pin(filename)
        old, new = feature(original["baseline"]), feature(path)
        rt = reaction_table(cfg, key)
        if new.n_bits != 2048 or new.X.dtype != np.uint8 or new.X.shape != (len(rt), 256):
            raise ValueError("Wrong selected cache construction/dimension")
        if not np.array_equal(new.ok, old.ok):
            raise ValueError("Old/new full-source valid masks differ")
        # Independent re-encoding against the frozen source text, without relying
        # only on dimensional or validity checks. All failed records stay in place.
        sample = sorted(set([0, min(1, len(rt)-1), len(rt)//2, len(rt)-1] + np.flatnonzero(~new.ok)[:2].tolist()))
        generated, ok = _synrfp_chunk(rt.rxn.iloc[sample].tolist(), "wl", 2,
                                     [SynRFPConfig.from_name(FP)], cfg["synrfp"], cfg["fp_seed"])
        if not np.array_equal(generated[FP], new.X[sample]) or not np.array_equal(ok, new.ok[sample]):
            raise ValueError("Selected 2048-bit construction replay differs")
        sources[key] = {"path": str(path), "native_receipt": str(receipt), "rows": len(rt),
                        "all_valid_masks_bitwise_equal": True, "construction_rows": sample}
        del old, new, rt
    panels, baselines = {}, []
    for ds_name in ENDPOINTS:
        ds = dataset_table(cfg, ds_name)
        key = rxn_key(cfg, ds_name)
        rt = reaction_table(cfg, key)
        rows = ds.row.to_numpy(np.int64)
        if len(np.unique(rows)) != len(rows) or not np.array_equal(rt.r_id.iloc[rows].astype(str).to_numpy(), ds.r_id.astype(str).to_numpy()):
            raise ValueError("Native endpoint/source row alignment differs")
        if not np.array_equal(rt.rxn.iloc[rows].to_numpy(), ds.rxn.to_numpy()):
            raise ValueError("Native endpoint reaction text differs")
        for path in [cfg["paths"]["work"] / "tables" / f"ds_{ds_name}.pkl",
                     cfg["paths"]["work"] / "tables" / f"rxn_{key}.pkl"]:
            pin(path)
        new = feature(sources[key]["path"])
        valid = new.ok[rows]
        shared = valid.copy()
        masks, feature_digests = {}, {}
        for fp in BASELINES:
            path = OLD / "encoder/features" / f"{key}.npz" if fp == "crxnfp" else cfg["paths"]["work"] / "features" / key / (fp + ".npz")
            pin(path)
            feature_digests[fp] = sha(path)
            f = feature(path)
            masks[fp] = f.ok[rows].copy()
            shared &= masks[fp]
            del f
        coverage = OLD / "clustering/coverage" / (ds_name + ".json")
        pin(coverage)
        old_coverage = json.loads(coverage.read_text())
        if old_coverage["n_shared_valid"] != int(shared.sum()):
            raise ValueError("Changed shared comparison population")
        population_file = ROOT / "panels" / (ds_name + ".npz")
        NATIVE.atomic_npz(population_file, source_row=rows, valid=valid,
                          shared_valid=shared, all_rows=np.ones(len(ds), bool))
        pin(population_file)
        panel = {"source": key, "n_rows": len(ds), "k": int(ds.y.nunique()),
                 "params": NATIVE.SMALL_PARAMS if ds_name in NATIVE.SMALL else NATIVE.LARGE_PARAMS,
                 "method": "kmeans" if ds_name in NATIVE.SMALL else "minibatch_kmeans",
                 "population_file": str(population_file), "shared_is_all": bool(shared.all()),
                 "n_shared_valid": int(shared.sum()), "valid_rows": int(valid.sum()),
                 "baseline_feature_sha256": feature_digests,
                 "dataset_sha256": sha(cfg["paths"]["work"] / "tables" / f"ds_{ds_name}.pkl"),
                 "label": cfg["datasets"][ds_name]["label"]}
        panels[ds_name] = panel
        for population in POLICIES:
            used = np.ones(len(ds), bool) if population == "all_rows" else shared
            for fp in BASELINES:
                for seed in range(10):
                    record = native_reference(ds_name, fp, population, seed)
                    rec = json.loads(record.read_text())
                    validate_native_record(rec, ds_name, fp, population, seed, panel)
                    if rec["n_scored"] != int(used.sum()):
                        raise ValueError("Native scored population count differs")
                    assignment = record.with_suffix(".npz")
                    if sha(assignment) != rec["assignment_sha256"]:
                        raise ValueError("Changed native baseline assignment")
                    validate_assignment(assignment, ds, used, masks[fp])
                    pin(record)
                    pin(assignment)
                    scores = rec["scores"][panel["label"]]
                    baselines.append({"task": "clustering", "dataset": ds_name, "endpoint": ds_name,
                                      "protocol": PROTOCOL, "population": population, "seed": seed,
                                      "method": fp, "fingerprint": fp, "origin": "verified_reuse",
                                      "n_rows": len(ds), "n_scored": int(used.sum()), "k": panel["k"],
                                      "source_result_path": str(record), "source_result_sha256": sha(record),
                                      "source_assignment_path": str(assignment), "source_assignment_sha256": sha(assignment),
                                      **scores})
        del new, ds, rt
    tasks = expected_tasks(panels)
    if len(tasks) != 240 or sum(t["origin"] == "fresh" for t in tasks) != 200:
        raise ValueError("Expected 240 logical / 200 fresh / 40 copy cells")
    baseline_path = ROOT / "baseline_bindings.json"
    write(baseline_path, {"expected": 1200, "rows": baselines})
    pin(baseline_path)
    plan = {"schema": "synrfp-final-native-clustering-v1", "created_utc": now(), "fingerprint": FP,
            "protocol": PROTOCOL, "selection": "exploratory_after_prior_test_results_viewed",
            "known_k": True, "labels_used_for_k": True, "normalization": "none",
            "n_init_and_algorithm": "unchanged native 12 recipes", "sources": sources,
            "panels": panels, "tasks": tasks, "expected_cells": 240, "expected_fresh": 200,
            "expected_population_copies": 40, "baseline_cells": 1200,
            "baseline_bindings": str(baseline_path), "versions": native_versions,
            "pins": pins, "threads_per_worker": 2, "gpu": False,
            "weights_and_assignments_local": True}
    write(ROOT / "plan.json", plan)
    write(ROOT / "plan_seal.json", {"sha256": sha(ROOT / "plan.json")})
    write(EXPORT / "plan.json", plan)
    return plan


def verify(fresh=False):
    plan = json.loads((ROOT / "plan.json").read_text())
    digest = sha(ROOT / "plan.json", fresh=fresh)
    if digest != json.loads((ROOT / "plan_seal.json").read_text())["sha256"]:
        raise ValueError("Changed final clustering plan")
    check_pins(plan["pins"], fresh=fresh)
    check_environment(plan["versions"])
    if plan["fingerprint"] != FP or len(plan["tasks"]) != 240:
        raise ValueError("Changed final clustering task universe")
    return plan


def score_equal(a, b):
    if set(a) != set(b):
        return False
    return all(set(a[level]) == set(b[level]) and all(np.isclose(a[level][key], b[level][key], atol=1e-12, rtol=1e-12)
               for key in a[level]) for level in a)


def verify_metric_replay(task, plan, rec, ds=None):
    cfg = load_config(OLD / "config.yaml")
    ds = dataset_table(cfg, task["dataset"]) if ds is None else ds
    panel = plan["panels"][task["dataset"]]
    used = used_mask(panel, task["population"])
    with np.load(panel["population_file"], allow_pickle=False) as z:
        valid = z["valid"].copy()
    labels = validate_assignment(rec["assignment_path"], ds, used, valid)
    if not score_equal(NATIVE.scores(ds, cfg["datasets"][task["dataset"]], labels, used), rec["scores"]):
        raise ValueError("Final assignment metric replay differs")


def audit_cell(task, plan, ds=None, replay=False):
    path = stem(task).with_suffix(".json")
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    panel = plan["panels"][task["dataset"]]
    expected = {**task, "fingerprint": FP, "plan_sha256": sha(ROOT / "plan.json"),
                "k": panel["k"], "params": panel["params"], "n_rows": panel["n_rows"]}
    for key, value in expected.items():
        if rec.get(key) != value:
            raise ValueError(f"Changed final cell identity {key}: {path}")
    for name, digest in rec["artifacts_sha256"].items():
        if sha(name) != digest:
            raise ValueError("Changed final clustering artifact: " + name)
    cfg = load_config(OLD / "config.yaml") if ds is None or replay else None
    ds = dataset_table(cfg, task["dataset"]) if ds is None else ds
    with np.load(panel["population_file"], allow_pickle=False) as z:
        valid = z["valid"].copy()
    used = used_mask(panel, task["population"])
    labels = validate_assignment(rec["assignment_path"], ds, used, valid)
    if rec["origin"] == "population_identical_copy":
        source_task = {"dataset": task["dataset"], "population": "all_rows", "seed": task["seed"], "origin": "fresh"}
        source = stem(source_task).with_suffix(".json")
        if not used.all() or sha(source) != rec["copied_result_sha256"]:
            raise ValueError("Invalid identical-population copy source")
        original = json.loads(source.read_text())
        if rec["assignments_sha256"] != original["assignments_sha256"] or rec["centers_sha256"] != original["centers_sha256"]:
            raise ValueError("Copied artifacts differ")
    if replay:
        scores = NATIVE.scores(ds, cfg["datasets"][task["dataset"]], labels, used)
        if not score_equal(scores, rec["scores"]):
            raise ValueError("Final assignment metric replay differs")
        X = feature(plan["sources"][panel["source"]]["path"]).dense(ds.row.to_numpy(np.int64)[used])
        centers = np.load(rec["centers_path"], allow_pickle=False)
        with threadpool_limits(limits=2):
            predicted = pairwise_distances_argmin(X, centers, metric="euclidean")
        if not np.array_equal(predicted, labels[used]):
            raise ValueError("Persisted cluster-center prediction replay differs")
    return rec


def fit_cell(task, plan):
    panel = plan["panels"][task["dataset"]]
    cfg = load_config(OLD / "config.yaml")
    ds = dataset_table(cfg, task["dataset"])
    if audit_cell(task, plan, ds=ds) is not None:
        return False
    started = time.perf_counter()
    path = stem(task).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    used = used_mask(panel, task["population"])
    if task["origin"] == "population_identical_copy":
        if not used.all():
            raise ValueError("Cannot copy a nonidentical population")
        source_task = {**task, "population": "all_rows", "origin": "fresh"}
        rec = audit_cell(source_task, plan, ds=ds)
        if rec is None:
            raise ValueError("Identical all-row population not completed before copy")
        result = {**rec, **task, "completed_utc": now(), "elapsed_seconds": time.perf_counter()-started,
                  "row_policy": task["population"],
                  "copied_result_path": str(stem(source_task).with_suffix(".json")),
                  "copied_result_sha256": sha(stem(source_task).with_suffix(".json")),
                  "fit_seconds": 0.0}
        write(path, result)
        return True
    cache = plan["sources"][panel["source"]]["path"]
    feat = feature(cache)
    rows = ds.row.to_numpy(np.int64)
    valid = feat.ok[rows].copy()
    X = feat.dense(rows[used])
    del feat
    if not np.isfinite(X).all() or not np.isin(X, [0, 1]).all():
        raise ValueError("Native unscaled bit decoding differs")
    cls = KMeans if task["dataset"] in NATIVE.SMALL else MiniBatchKMeans
    fit_started = time.perf_counter()
    with threadpool_limits(limits=2):
        model = cls(n_clusters=panel["k"], random_state=task["seed"], **panel["params"])
        fitted = model.fit_predict(X)
    fit_seconds = time.perf_counter() - fit_started
    labels = np.full(len(ds), -1, dtype=np.int32)
    labels[used] = fitted
    assignment = stem(task).with_suffix(".npz")
    centers_path = stem(task).with_name(f"s{task['seed']}_centers.npy")
    NATIVE.atomic_npz(assignment, labels=labels, valid=valid, used=used,
                      source_row=rows, r_id=ds.r_id.astype(str).to_numpy(dtype=str), y=ds.y.to_numpy())
    temporary = centers_path.with_name(centers_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, model.cluster_centers_, allow_pickle=False)
    temporary.replace(centers_path)
    # Use estimator.predict for exact algorithm arithmetic and a separate nearest-
    # center implementation on a bounded deterministic sample as a mutation guard.
    with threadpool_limits(limits=2):
        predicted = model.predict(X)
        ix = np.unique(np.linspace(0, len(X)-1, min(len(X), 2048), dtype=np.int64))
        saved_centers = np.load(centers_path)
        if not np.array_equal(saved_centers, model.cluster_centers_):
            raise ValueError("Persisted centers differ from fitted centers")
        independent = pairwise_distances_argmin(X[ix], saved_centers, metric="euclidean")
    if not np.array_equal(predicted, fitted) or not np.array_equal(independent, fitted[ix]):
        raise ValueError("Saved center prediction replay differs")
    scores = NATIVE.scores(ds, cfg["datasets"][task["dataset"]], labels, used)
    rec = {**task, "task": "clustering", "fingerprint": FP, "protocol": PROTOCOL,
           "input_policy": "n1", "row_policy": task["population"], "method": panel["method"],
           "plan_sha256": sha(ROOT / "plan.json"), "params": panel["params"], "k": panel["k"],
           "n_rows": len(ds), "n_scored": int(used.sum()), "n_valid": int(valid.sum()),
           "feature_sha256": plan["pins"][str(Path(cache).resolve())], "scores": scores,
           "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter()-started,
           "inertia": float(model.inertia_), "n_iter": int(model.n_iter_),
           "assignment_path": str(assignment), "assignments_sha256": sha(assignment),
           "centers_path": str(centers_path), "centers_sha256": sha(centers_path),
           "artifacts_sha256": {str(assignment): sha(assignment), str(centers_path): sha(centers_path)},
           "prediction_replay": True, "independent_center_replay_rows": len(ix),
           "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
           "completed_utc": now()}
    write(path, rec)
    audit_cell(task, plan, ds=ds)
    return True


def preflight():
    plan = verify(fresh=True)
    import subprocess
    result = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_clustering"], cwd=HERE)
    if result.returncode:
        raise ValueError("Meaningful clustering contract tests failed")
    receipt = {"phase": "complete", "plan_sha256": sha(ROOT / "plan.json"),
               "checked_frozen_sources": len(plan["pins"]), "native_environment": environment(),
               "fresh_and_baseline_fits": 0, "completed_utc": now()}
    write(ROOT / "PREFLIGHT_READY.json", receipt)
    write(EXPORT / "preflight.json", receipt)


def pilot():
    plan = verify(fresh=True)
    pre = json.loads((ROOT / "PREFLIGHT_READY.json").read_text())
    if pre["plan_sha256"] != sha(ROOT / "plan.json"):
        raise ValueError("Pilot preflight plan differs")
    task = next(t for t in plan["tasks"] if t["dataset"] == "tpl_b" and t["population"] == "all_rows" and t["seed"] == 0)
    fit_cell(task, plan)
    rec = audit_cell(task, plan, replay=True)
    receipt = {"phase": "ready", "plan_sha256": sha(ROOT / "plan.json"),
               "task": task, "fit_seconds": rec["fit_seconds"], "peak_rss_gib": rec["peak_rss_gib"],
               "model_and_metric_replay": True, "completed_utc": now()}
    write(ROOT / "PILOT_READY.json", receipt)
    write(EXPORT / "pilot.json", receipt)
    report()


def run(worker):
    plan = verify(fresh=True)
    pilot_receipt = json.loads((ROOT / "PILOT_READY.json").read_text())
    if pilot_receipt["plan_sha256"] != sha(ROOT / "plan.json"):
        raise ValueError("Full-run pilot plan differs")
    # Endpoint-level partition keeps all/shared copies on one worker and caches
    # every frozen source hash within that process. No lease or GPU is touched.
    selected = [t for t in plan["tasks"] if ENDPOINTS.index(t["dataset"]) % 4 == worker]
    lock = (ROOT / f"worker{worker}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for index, task in enumerate(selected):
        verify()
        write(ROOT / f"worker{worker}_status.json", {"phase": "running", "current": task,
              "completed": index, "expected": len(selected), "updated_utc": now()})
        fit_cell(task, plan)
        if (index + 1) % 5 == 0:
            report()
    write(ROOT / f"worker{worker}_status.json", {"phase": "complete", "completed": len(selected), "updated_utc": now()})
    write(ROOT / f"WORKER_{worker}_COMPLETE.json", {"phase": "complete", "completed": len(selected),
          "plan_sha256": sha(ROOT / "plan.json"), "updated_utc": now()})
    report()


def report():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "report.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = verify()
        expected_paths = {stem(task).with_suffix(".json") for task in plan["tasks"]}
        actual_paths = set((ROOT / "cells").rglob("*.json"))
        if not actual_paths <= expected_paths:
            raise ValueError("Unexpected final clustering cell identities")
        rows = []
        cfg = load_config(OLD / "config.yaml")
        current_dataset, ds = None, None
        for task in plan["tasks"]:
            if not stem(task).with_suffix(".json").exists():
                continue
            if current_dataset != task["dataset"]:
                current_dataset = task["dataset"]
                ds = dataset_table(cfg, current_dataset)
            rec = audit_cell(task, plan, ds=ds)
            if rec is None:
                continue
            score = rec["scores"][plan["panels"][task["dataset"]]["label"]]
            path = stem(task).with_suffix(".json")
            rows.append({"task": "clustering", "dataset": task["dataset"], "endpoint": task["dataset"],
                         "protocol": PROTOCOL, "population": task["population"], "seed": task["seed"],
                         "method": FP, "fingerprint": FP, "origin": rec["origin"], "n_rows": rec["n_rows"],
                         "n_scored": rec["n_scored"], "k": rec["k"], "fit_seconds": rec["fit_seconds"],
                         "result_path": str(path), "result_sha256": sha(path),
                         "assignments_sha256": rec["assignments_sha256"], "centers_sha256": rec["centers_sha256"],
                         "plan_sha256": sha(ROOT / "plan.json"), **score})
        baselines = json.loads(Path(plan["baseline_bindings"]).read_text())["rows"]
        if len(baselines) != 1200:
            raise ValueError("Baseline population incomplete")
        write_csv(EXPORT / "cells.csv", rows)
        write_csv(EXPORT / "baseline_cells.csv", baselines)
        all_rows = rows + baselines
        df = pd.DataFrame(all_rows)
        summary = df.groupby(["dataset", "population", "method"]).agg(
            completed=("ami", "size"), ami_mean=("ami", "mean"), ami_std=("ami", "std"),
            ari_mean=("ari", "mean"), ari_std=("ari", "std"), nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std")).reset_index()
        write_csv(EXPORT / "summary.csv", summary.to_dict(orient="records"))
        status = {"phase": "complete" if len(rows) == 240 else "running", "completed": len(rows), "expected": 240,
                  "baseline_cells": 1200, "fresh_completed": sum(x["origin"] == "fresh" for x in rows),
                  "population_copies_completed": sum(x["origin"] == "population_identical_copy" for x in rows),
                  "plan_sha256": sha(ROOT / "plan.json"), "updated_utc": now()}
        if len(rows) == 240:
            verify(fresh=True)
            previous = ROOT / "COMPLETE.json"
            already_replayed = previous.exists() and json.loads(previous.read_text()).get("full_metric_replay") is True
            current_dataset, ds = None, None
            for task in plan["tasks"]:
                if task["origin"] == "fresh":
                    if current_dataset != task["dataset"]:
                        current_dataset = task["dataset"]
                        ds = dataset_table(cfg, current_dataset)
                    rec = audit_cell(task, plan, ds=ds)
                    if not rec["prediction_replay"]:
                        raise ValueError("Missing center replay proof")
                    if not already_replayed:
                        verify_metric_replay(task, plan, rec, ds=ds)
            status["all_artifact_hashes_verified"] = True
            status["full_metric_replay"] = True
            status["outputs_sha256"] = {name: sha(EXPORT / name, fresh=True) for name in ("cells.csv", "baseline_cells.csv", "summary.csv", "plan.json")}
            write(ROOT / "COMPLETE.json", status)
        write(ROOT / "progress.json", status)
        write(EXPORT / "final_component.json", status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    for stage in ("prepare", "preflight", "pilot", "run", "report"):
        stages.add_argument("--" + stage, action="store_true")
    parser.add_argument("--worker", type=int, choices=range(4))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.run and args.worker is None:
        parser.error("--run requires --worker")
    try:
        result = prepare() if args.prepare else preflight() if args.preflight else pilot() if args.pilot else run(args.worker) if args.run else report()
        if isinstance(result, dict):
            print(json.dumps({key: result[key] for key in ("phase", "completed", "expected", "expected_cells", "expected_fresh") if key in result}), flush=True)
    except BaseException as error:
        if ROOT.exists():
            write(ROOT / (f"worker{args.worker}_FAILED.json" if args.run else "FAILED.json"), {"error": repr(error), "updated_utc": now()})
        raise


if __name__ == "__main__":
    main()
