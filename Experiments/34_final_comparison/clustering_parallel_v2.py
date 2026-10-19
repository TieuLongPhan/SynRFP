#!/usr/bin/env python
"""Eight balanced CPU dispatchers for the sealed native clustering protocol.

Only scheduling changes. Native clustering.py remains byte-identical, owns all
scientific results, and supplies fitting, validation, reporting and final receipts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import clustering as C

HERE = Path(__file__).resolve().parent
ROOT = C.ROOT / "parallel_v2"
WORKERS = 8
PILOT_UNIT = "synrfp-final-comparison-clustering-pilot-v1.service"


def partition(tasks, costs, workers=WORKERS):
    """Deterministic largest-first scheduling of paired all/shared seed groups."""
    groups = {}
    for task in tasks:
        key = task["dataset"], int(task["seed"])
        groups.setdefault(key, []).append(dict(task))
    ordered = []
    for key, pair in groups.items():
        if len(pair) != 2 or {t["population"] for t in pair} != set(C.POLICIES):
            raise ValueError("Every seed group must contain its two native populations")
        if (key not in costs or isinstance(costs[key], bool) or not isinstance(costs[key], (int, float))
                or not 0 < costs[key] < float("inf")):
            raise ValueError("Invalid historical scheduling cost")
        pair.sort(key=lambda t: C.POLICIES.index(t["population"]))
        ordered.append({"dataset": key[0], "seed": key[1], "cost": float(costs[key]), "tasks": pair})
    ordered.sort(key=lambda group: (-group["cost"], group["dataset"], group["seed"]))
    bins = [{"worker": i, "cost": 0.0, "groups": []} for i in range(workers)]
    for group in ordered:
        target = min(bins, key=lambda item: (item["cost"], item["worker"]))
        target["groups"].append(group)
        target["cost"] += group["cost"]
    validate_partition(tasks, bins, workers)
    return bins


def validate_partition(tasks, bins, workers=WORKERS):
    if len(bins) != workers or {b["worker"] for b in bins} != set(range(workers)):
        raise ValueError("Unexpected worker population")
    expected = {C.identity(t): t for t in tasks}
    found = {}
    for bin in bins:
        for group in bin["groups"]:
            if len(group["tasks"]) != 2:
                raise ValueError("Split paired native populations")
            for task in group["tasks"]:
                key = C.identity(task)
                if key in found or key not in expected or task != expected[key]:
                    raise ValueError("Duplicate, missing, or changed scientific cell identity")
                if (task["dataset"], task["seed"]) != (group["dataset"], group["seed"]):
                    raise ValueError("Seed pair belongs to different groups")
                found[key] = bin["worker"]
            if [t["population"] for t in group["tasks"]] != C.POLICIES:
                raise ValueError("Copy/population dependency order changed")
    if set(found) != set(expected):
        raise ValueError("Incomplete scientific task universe")
    return True


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "plan.json").exists():
        return verify(fresh=True)
    native = C.verify(fresh=True)
    costs, pins = {}, {}
    for filename in [Path(__file__), HERE / "test_clustering_parallel_v2.py",
                     C.ROOT / "plan.json", C.ROOT / "plan_seal.json",
                     C.ROOT.parent / "execution.json", C.ROOT / "CODE_READY.json"]:
        pins[str(filename.resolve())] = C.sha(filename)
    for dataset in C.ENDPOINTS:
        method = native["panels"][dataset]["method"]
        for seed in range(10):
            path = C.MAIN / dataset / C.LEGACY / f"{method}__s{seed}.json"
            record = json.loads(path.read_text())
            if (record["dataset"] != dataset or record["seed"] != seed
                    or record["fingerprint"] != C.LEGACY or record["method"] != method):
                raise ValueError("Historical scheduling cost identity differs")
            costs[(dataset, seed)] = record["fit_seconds"] * (1 if native["panels"][dataset]["shared_is_all"] else 2)
            pins[str(path)] = C.sha(path)
    assignments = partition(native["tasks"], costs)
    plan = {"schema": "synrfp-final-native-clustering-balanced-dispatch-v2",
            "created_utc": C.now(), "native_plan_sha256": C.sha(C.ROOT / "plan.json"),
            "fingerprint": C.FP, "scientific_recipe_changed": False,
            "algorithm": "deterministic_largest_processing_time_first",
            "cost_scope": "historical native1024 timings, only scheduling; no score-dependent selection",
            "workers": WORKERS, "cpu_equivalents_per_worker": 2, "memory_gib_per_worker": 10,
            "expected_logical_cells": 240, "fresh_total_including_pilot": 200,
            "population_copies": 40, "assignments": assignments, "pins": pins,
            "native_primitive": "clustering.fit_cell and clustering.audit_cell, unchanged"}
    C.write(ROOT / "plan.json", plan)
    C.write(ROOT / "plan_seal.json", {"sha256": C.sha(ROOT / "plan.json")})
    C.write(ROOT / "PREPARED.json", {"phase": "complete", "plan_sha256": C.sha(ROOT / "plan.json"),
            "native_plan_sha256": plan["native_plan_sha256"], "expected": 240, "workers": 8, "completed_utc": C.now()})
    return plan


def verify(fresh=False):
    native = C.verify(fresh=fresh)
    path = ROOT / "plan.json"
    plan = json.loads(path.read_text())
    if C.sha(path, fresh=fresh) != json.loads((ROOT / "plan_seal.json").read_text())["sha256"]:
        raise ValueError("Changed balanced-dispatch plan")
    C.check_pins(plan["pins"], fresh=fresh)
    if C.sha(C.ROOT / "plan.json") != plan["native_plan_sha256"]:
        raise ValueError("Changed native scientific plan")
    validate_partition(native["tasks"], plan["assignments"])
    return plan


def preflight():
    plan = verify(fresh=True)
    result = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_clustering_parallel_v2"], cwd=HERE)
    if result.returncode:
        raise ValueError("Balanced dispatch contract tests failed")
    C.write(ROOT / "PREFLIGHT_READY.json", {"phase": "complete", "plan_sha256": C.sha(ROOT / "plan.json"),
            "native_plan_sha256": plan["native_plan_sha256"], "completed_utc": C.now(), "new_fits": 0})


def admission_ready(plan):
    preflight = json.loads((ROOT / "PREFLIGHT_READY.json").read_text())
    if preflight.get("phase") != "complete" or preflight.get("plan_sha256") != C.sha(ROOT / "plan.json"):
        raise ValueError("Balanced dispatch preflight differs")
    pilot = json.loads((C.ROOT / "PILOT_READY.json").read_text())
    if pilot.get("phase") != "ready" or pilot.get("plan_sha256") != plan["native_plan_sha256"]:
        raise ValueError("Native full-size pilot is not ready")
    expected_task = {"dataset": "tpl_b", "population": "all_rows", "seed": 0, "origin": "fresh"}
    if pilot.get("task") != expected_task or pilot.get("model_and_metric_replay") is not True:
        raise ValueError("Native full-size pilot identity/replay proof differs")
    if pilot.get("peak_rss_gib", float("inf")) >= 9:
        raise ValueError("Native pilot leaves insufficient 10GiB worker headroom")
    state = subprocess.run(["systemctl", "--user", "is-active", PILOT_UNIT], capture_output=True, text=True).stdout.strip()
    if state in ("active", "activating", "deactivating", "reloading"):
        raise ValueError("Wait for full native pilot cgroup to release before parallel fits")


def all_native_receipts(worker, plan, native):
    own = next(item for item in plan["assignments"] if item["worker"] == worker)
    outputs = {}
    for group in own["groups"]:
        for task in group["tasks"]:
            if C.audit_cell(task, native) is None:
                raise ValueError("Missing balanced worker native result")
            path = C.stem(task).with_suffix(".json")
            outputs[str(path)] = C.sha(path)
    return own, outputs


def run(worker):
    plan = verify(fresh=True)
    admission_ready(plan)
    native = C.verify()
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / f"worker{worker}.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # All eight dispatchers can share these locks. The four original serial
        # workers require exclusive ownership, so accidental concurrent launches
        # cannot race these scientifically identical cell paths.
        for i in range(4):
            handle = stack.enter_context((C.ROOT / f"worker{i}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        own = next(item for item in plan["assignments"] if item["worker"] == worker)
        tasks = [task for group in own["groups"] for task in group["tasks"]]
        for i, task in enumerate(tasks):
            verify()
            C.write(ROOT / f"worker{worker}_status.json", {"phase": "running", "current": task,
                    "completed": i, "expected": len(tasks), "updated_utc": C.now()})
            C.fit_cell(task, native)
        own, outputs = all_native_receipts(worker, plan, native)
        receipt = {"phase": "complete", "worker": worker, "completed": len(tasks), "expected": len(tasks),
                   "plan_sha256": C.sha(ROOT / "plan.json"), "native_plan_sha256": plan["native_plan_sha256"],
                   "outputs_sha256": outputs, "finished_utc": C.now()}
        C.write(ROOT / f"WORKER_{worker}_COMPLETE.json", receipt)
        C.write(ROOT / f"worker{worker}_status.json", receipt)


def finalize():
    plan = verify(fresh=True)
    native = C.verify()
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / "finalize.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Acquiring every private worker lock prevents finalization while one
        # dispatcher can still write a native result or its completion receipt.
        for worker in range(WORKERS):
            handle = stack.enter_context((ROOT / f"worker{worker}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rec = json.loads((ROOT / f"WORKER_{worker}_COMPLETE.json").read_text())
            own, outputs = all_native_receipts(worker, plan, native)
            expected = sum(len(g["tasks"]) for g in own["groups"])
            if (rec.get("phase") != "complete" or rec.get("completed") != expected
                    or rec.get("expected") != expected or rec.get("plan_sha256") != C.sha(ROOT / "plan.json")
                    or rec.get("outputs_sha256") != outputs):
                raise ValueError("Balanced worker completion binding differs")
        for task in native["tasks"]:
            if C.audit_cell(task, native) is None:
                raise ValueError("All 240 native cells are required before receipt consolidation")
        for worker in range(4):
            handle = stack.enter_context((C.ROOT / f"worker{worker}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No missing scientific cell exists at this point. The unchanged native
        # reporter performs its full artifact and metric audit exactly once.
        C.report()
        done = json.loads((C.ROOT / "COMPLETE.json").read_text())
        if done.get("phase") != "complete" or done.get("completed") != 240:
            raise ValueError("Native final clustering completion is incomplete")
        C.write(ROOT / "COMPLETE.json", {"phase": "complete", "completed": 240, "expected": 240,
                "plan_sha256": C.sha(ROOT / "plan.json"), "native_complete_sha256": C.sha(C.ROOT / "COMPLETE.json"),
                "finished_utc": C.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "preflight", "run", "finalize"):
        stages.add_argument("--" + name, action="store_true")
    parser.add_argument("--worker", type=int, choices=range(WORKERS))
    args = parser.parse_args()
    if args.run and args.worker is None:
        parser.error("--run requires --worker 0..7")
    if args.prepare:
        plan = prepare()
        print(json.dumps({"phase": "prepared", "workers": 8, "costs": [b["cost"] for b in plan["assignments"]]}), flush=True)
    elif args.preflight:
        preflight()
    elif args.run:
        run(args.worker)
    else:
        finalize()


if __name__ == "__main__":
    main()
