#!/usr/bin/env python
"""Validation-only repair of cross-implementation float32 center comparisons.

The frozen fitter, native full prediction equality, source caches, populations,
seeds, algorithms and numerical training remain unchanged. Only the secondary
cross-implementation comparison receives a documented forward-error bound.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import clustering as C
import clustering_parallel_v2 as P

ROOT = C.ROOT / "recovery_v3"
INCIDENT = C.ROOT.parent / "recovery/center_replay_v3/incident.json"
WORKERS = 8


def numerical_oracle(X, centers, native, alternative):
    """Check disagreements using direct float64 distances and a forward bound.

For float32 unit roundoff u, gamma_m=m*u/(1-m*u), m=2*d+8 bounds
the multiplications/additions in dot-product/norm distance evaluation. For
comparison of two distances, add their absolute expression scales:
sum(x²+c_a²+2|x*c_a|)+sum(x²+c_b²+2|x*c_b|). This permits documented
roundoff ambiguity, never a changed native prediction or changed center array.
"""
    X, centers = np.asarray(X), np.asarray(centers)
    native, alternative = np.asarray(native), np.asarray(alternative)
    if (X.ndim != 2 or centers.ndim != 2 or X.shape[1] != centers.shape[1]
            or native.shape != (len(X),) or alternative.shape != native.shape
            or not np.isfinite(X).all() or not np.isfinite(centers).all()
            or np.any(native < 0) or np.any(native >= len(centers))
            or np.any(alternative < 0) or np.any(alternative >= len(centers))):
        raise ValueError("Invalid independent center oracle input")
    mismatches = np.flatnonzero(native != alternative)
    u = np.finfo(np.float32).eps / 2
    m = 2 * X.shape[1] + 8
    gamma = m*u/(1-m*u)
    records = []
    c64 = centers.astype(np.float64)
    for i in mismatches:
        x = X[i].astype(np.float64)
        distances = np.sum((c64-x)**2, axis=1, dtype=np.float64)
        best, chosen = int(distances.argmin()), int(native[i])
        a, b = c64[chosen], c64[best]
        scale = float(np.sum(x*x+a*a+2*np.abs(x*a)) + np.sum(x*x+b*b+2*np.abs(x*b)))
        bound = gamma*scale + 16*np.finfo(np.float64).eps*max(1.0, scale)
        regret = float(distances[chosen]-distances[best])
        if regret > bound:
            raise ValueError(f"Independent center distance exceeds float32 forward bound: regret={regret}, bound={bound}")
        records.append({"sample_index": int(i), "native_label": chosen,
                        "independent_float32_label": int(alternative[i]), "float64_best_label": best,
                        "native_distance_squared_float64": float(distances[chosen]),
                        "best_distance_squared_float64": float(distances[best]),
                        "regret": regret, "forward_error_bound": bound, "expression_scale": scale})
    return {"schema": "float32-distance-forward-error-v1", "rows": len(X),
            "mismatches": len(records), "feature_dimensions": X.shape[1], "unit_roundoff": float(u),
            "gamma_operations": m, "gamma": float(gamma),
            "max_regret": max((r["regret"] for r in records), default=0.0),
            "max_forward_error_bound": max((r["forward_error_bound"] for r in records), default=0.0),
            "disagreements": records, "native_full_predict_equality_unchanged": True,
            "persisted_center_exact_equality_unchanged": True}


def adapted_fitter(write_result=None):
    source = inspect.getsource(C.fit_cell)
    tree = ast.parse(source)
    target = ast.parse("np.array_equal(independent, fitted[ix])", mode="eval").body
    replacement = ast.parse("_recovery_oracle(X[ix], saved_centers, fitted[ix], independent)", mode="eval").body
    expected = ast.dump(target, include_attributes=False)
    class Replace(ast.NodeTransformer):
        changed = 0
        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                self.changed += 1
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)
    rewrite = Replace()
    rewritten = rewrite.visit(tree)
    if rewrite.changed != 1:
        raise ValueError("Expected exactly one secondary-validation AST site")
    ast.fix_missing_locations(rewritten)
    proof = {}
    def oracle(*args):
        proof["oracle"] = numerical_oracle(*args)
        return True
    namespace = dict(C.__dict__)
    namespace["_recovery_oracle"] = oracle
    if write_result is not None:
        namespace["write"] = write_result
    exec(compile(rewritten, str(Path(__file__)), "exec"), namespace)
    binding = {"changed_ast_sites": 1, "replacement": "secondary center comparison only",
               "native_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "adapted_ast_sha256": hashlib.sha256(ast.dump(rewritten, include_attributes=False).encode()).hexdigest()}
    return namespace["fit_cell"], proof, binding


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "plan.json").exists():
        return verify(fresh=True)
    native = C.verify(fresh=True)
    dispatch = P.verify(fresh=True)
    incident = json.loads(INCIDENT.read_text())
    accepted = incident["completed_records_sha256"]
    C.check_pins(accepted, fresh=True)
    if len(accepted) != 48:
        raise ValueError("Expected the incident's 48 immutable accepted records")
    pins = dict(accepted)
    for path in (Path(__file__), Path(__file__).with_name("test_clustering_recovery_v3.py"),
                 Path(P.__file__), Path(P.__file__).with_name("test_clustering_parallel_v2.py"),
                 C.ROOT / "plan.json", P.ROOT / "plan.json", P.ROOT / "plan_seal.json", INCIDENT):
        pins[str(path.resolve())] = C.sha(path)
    archives = []
    for task in native["tasks"]:
        stem = C.stem(task)
        if stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
            continue
        for original in (stem.with_suffix(".npz"), stem.with_name(stem.name+"_centers.npy")):
            relative = original.relative_to(C.ROOT)
            target = ROOT / "retained_orphans" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = C.sha(original, fresh=True)
            if target.exists():
                if C.sha(target, fresh=True) != digest:
                    raise ValueError("Conflicting retained orphan artifact")
            else:
                temporary = target.with_name(target.name+".tmp")
                shutil.copy2(original, temporary)
                if C.sha(temporary, fresh=True) != digest:
                    raise ValueError("Retained orphan artifact copy differs")
                temporary.replace(target)
            pins[str(target)] = digest
            archives.append({"original": str(original), "retained": str(target), "sha256": digest})
    if len(archives) != 16:
        raise ValueError("Expected eight retained orphan assignment/center pairs")
    _, _, adapter = adapted_fitter()
    plan = {"schema": "synrfp-clustering-validation-recovery-v3", "created_utc": C.now(),
            "native_plan_sha256": C.sha(C.ROOT / "plan.json"), "dispatch_plan_sha256": C.sha(P.ROOT / "plan.json"),
            "fingerprint": C.FP, "accepted_records_sha256": accepted, "retained_orphans": archives,
            "assignments": dispatch["assignments"], "expected": 240, "workers": 8,
            "pins": pins, "adapter": adapter, "gpu": False,
            "training_numerical_arithmetic_changed": False, "full_native_prediction_equality_changed": False,
            "validation_change": "secondary cross-implementation discrepancies use declared data-dependent float32 forward-error bound"}
    C.write(ROOT / "plan.json", plan)
    C.write(ROOT / "plan_seal.json", {"sha256": C.sha(ROOT / "plan.json")})
    C.write(ROOT / "PREPARED.json", {"phase": "complete", "plan_sha256": C.sha(ROOT / "plan.json"),
            "immutable_completed": 48, "retained_orphan_artifacts": 16, "completed_utc": C.now()})
    return plan


def verify(fresh=False):
    native = C.verify(fresh=fresh)
    plan = json.loads((ROOT / "plan.json").read_text())
    if C.sha(ROOT / "plan.json", fresh=fresh) != json.loads((ROOT / "plan_seal.json").read_text())["sha256"]:
        raise ValueError("Changed replay-recovery plan")
    C.check_pins(plan["pins"], fresh=fresh)
    if C.sha(C.ROOT / "plan.json") != plan["native_plan_sha256"]:
        raise ValueError("Changed native scientific clustering plan")
    P.validate_partition(native["tasks"], plan["assignments"])
    return plan


def repaired_fit(task, native, plan):
    path = C.stem(task).with_suffix(".json")
    if path.exists():
        rec = C.audit_cell(task, native)
        if str(path) not in plan["accepted_records_sha256"]:
            overlay = rec.get("validation_overlay", {})
            if overlay.get("recovery_plan_sha256") != C.sha(ROOT / "plan.json"):
                raise ValueError("New clustering result lacks recovery validation binding")
        return False
    def atomic_result(filename, value):
        if Path(filename) != path:
            raise ValueError("Unexpected result write in scoped validation adapter")
        if task["origin"] == "fresh" and "oracle" not in proof:
            raise ValueError("Missing secondary center oracle proof")
        value = {**value, "validation_overlay": {"recovery_plan_sha256": C.sha(ROOT / "plan.json"),
                "source_sha256": C.sha(Path(__file__)), "adapter": adapter,
                "native_full_predict_vs_fit_exact": True, "persisted_centers_exact": True,
                "scope": "fresh_fit" if task["origin"] == "fresh" else "verified_identical_population_copy",
                "independent_oracle": proof.get("oracle")}}
        C.write(filename, value)
    fn, proof, adapter = adapted_fitter(write_result=atomic_result)
    fn(task, native)
    C.audit_cell(task, native)
    return True


def real_orphan_preflight(plan):
    """Read-only full native replay and bounded secondary audit of failed fits."""
    from threadpoolctl import threadpool_limits
    cfg = C.load_config(C.OLD / "config.yaml")
    native = C.verify()
    results = []
    archives = {entry["original"]: entry["retained"] for entry in plan["retained_orphans"]}
    for dataset, seed in (("tpl_b", 1), ("ecreact_3", 6)):
        task = next(t for t in native["tasks"] if t["dataset"] == dataset and t["seed"] == seed and t["population"] == "all_rows")
        panel = native["panels"][dataset]
        stem = C.stem(task)
        ds = C.dataset_table(cfg, dataset)
        used = C.used_mask(panel, "all_rows")
        with np.load(panel["population_file"]) as z:
            valid = z["valid"]
        labels = C.validate_assignment(archives[str(stem.with_suffix(".npz"))], ds, used, valid)
        centers = np.load(archives[str(stem.with_name(stem.name+"_centers.npy"))])
        feat = C.feature(native["sources"][panel["source"]]["path"])
        rows = ds.row.to_numpy(np.int64)
        cls = C.KMeans if dataset in C.NATIVE.SMALL else C.MiniBatchKMeans
        model = cls(n_clusters=panel["k"], random_state=seed, **panel["params"])
        model.cluster_centers_, model.n_features_in_, model._n_threads = centers, 2048, 2
        mismatches = 0
        with threadpool_limits(limits=2):
            for offset in range(0, len(rows), 8192):
                pred = model.predict(feat.dense(rows[offset:offset+8192]))
                mismatches += int(np.count_nonzero(pred != labels[offset:offset+8192]))
            if mismatches:
                raise ValueError("Full native retained-center replay differs from fit labels")
            ix = np.unique(np.linspace(0, len(rows)-1, min(2048, len(rows)), dtype=np.int64))
            X = feat.dense(rows[ix])
            alternative = C.pairwise_distances_argmin(X, centers, metric="euclidean")
            proof = numerical_oracle(X, centers, labels[ix], alternative)
        results.append({"dataset": dataset, "seed": seed, "native_rows": len(rows), "native_prediction_mismatches": 0,
                        "assignment_sha256": C.sha(archives[str(stem.with_suffix(".npz"))]),
                        "centers_sha256": C.sha(archives[str(stem.with_name(stem.name+"_centers.npy"))]),
                        "secondary_oracle": proof})
    C.write(ROOT / "REAL_ORPHAN_PREFLIGHT.json", {"phase": "complete", "cases": results,
            "plan_sha256": C.sha(ROOT / "plan.json"), "completed_utc": C.now()})
    return results


def preflight():
    plan = verify(fresh=True)
    result = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_clustering_recovery_v3"], cwd=Path(__file__).parent)
    if result.returncode:
        raise ValueError("Validation-recovery controls failed")
    cases = real_orphan_preflight(plan)
    C.write(ROOT / "PREFLIGHT_READY.json", {"phase": "complete", "plan_sha256": C.sha(ROOT / "plan.json"),
            "real_orphan_cases": len(cases), "full_native_replayed_rows": sum(x["native_rows"] for x in cases),
            "outputs_sha256": {"REAL_ORPHAN_PREFLIGHT.json": C.sha(ROOT / "REAL_ORPHAN_PREFLIGHT.json")}, "completed_utc": C.now()})


def pilot():
    plan = verify(fresh=True)
    pre = json.loads((ROOT / "PREFLIGHT_READY.json").read_text())
    if pre.get("phase") != "complete" or pre.get("plan_sha256") != C.sha(ROOT / "plan.json"):
        raise ValueError("Recovery preflight is not ready")
    native = C.verify()
    task = next(t for t in native["tasks"] if t["dataset"] == "ecreact_3" and t["seed"] == 6 and t["population"] == "all_rows")
    repaired_fit(task, native, plan)
    rec = C.audit_cell(task, native)
    C.verify_metric_replay(task, native, rec)
    C.write(ROOT / "PILOT_READY.json", {"phase": "ready", "task": task, "plan_sha256": C.sha(ROOT / "plan.json"),
            "fit_seconds": rec["fit_seconds"], "oracle": rec["validation_overlay"]["independent_oracle"],
            "native_exact_prediction_replay": True, "metric_replay": True, "completed_utc": C.now()})
    C.report()


def run(worker):
    plan = verify(fresh=True)
    pilot = json.loads((ROOT / "PILOT_READY.json").read_text())
    if pilot.get("phase") != "ready" or pilot.get("plan_sha256") != C.sha(ROOT / "plan.json"):
        raise ValueError("Recovery pilot not ready")
    native = C.verify()
    own = next(b for b in plan["assignments"] if b["worker"] == worker)
    tasks = [t for g in own["groups"] for t in g["tasks"]]
    with ExitStack() as stack:
        handle = stack.enter_context((ROOT / f"worker{worker}.lock").open("a+"))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle = stack.enter_context((P.ROOT / f"worker{worker}.lock").open("a+"))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in range(4):
            handle = stack.enter_context((C.ROOT / f"worker{i}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        for i, task in enumerate(tasks):
            verify()
            C.write(ROOT / f"worker{worker}_status.json", {"phase": "running", "current": task,
                    "completed": i, "expected": len(tasks), "updated_utc": C.now()})
            repaired_fit(task, native, plan)
        outputs = {}
        for task in tasks:
            repaired_fit(task, native, plan)
            path = C.stem(task).with_suffix(".json")
            outputs[str(path)] = C.sha(path)
        receipt = {"phase": "complete", "worker": worker, "completed": len(tasks), "expected": len(tasks),
                   "plan_sha256": C.sha(ROOT / "plan.json"), "outputs_sha256": outputs, "completed_utc": C.now()}
        C.write(ROOT / f"WORKER_{worker}_COMPLETE.json", receipt)
        C.write(ROOT / f"worker{worker}_status.json", receipt)


def finalize():
    plan = verify(fresh=True)
    native = C.verify()
    with ExitStack() as stack:
        for worker in range(8):
            handle = stack.enter_context((ROOT / f"worker{worker}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle = stack.enter_context((P.ROOT / f"worker{worker}.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = json.loads((ROOT / f"WORKER_{worker}_COMPLETE.json").read_text())
            own = next(b for b in plan["assignments"] if b["worker"] == worker)
            expected = sum(len(g["tasks"]) for g in own["groups"])
            if (receipt.get("phase") != "complete" or receipt.get("completed") != expected
                    or receipt.get("expected") != expected or receipt.get("plan_sha256") != C.sha(ROOT / "plan.json")):
                raise ValueError("Recovery worker completion binding differs")
            C.check_pins(receipt["outputs_sha256"], fresh=True)
        for task in native["tasks"]:
            if not C.stem(task).with_suffix(".json").exists():
                raise ValueError("Missing scientific clustering cell at finalization")
            repaired_fit(task, native, plan)
        C.check_pins(plan["accepted_records_sha256"], fresh=True)
        C.report()
        done = json.loads((C.ROOT / "COMPLETE.json").read_text())
        if done.get("phase") != "complete" or done.get("completed") != 240:
            raise ValueError("Native clustering finalization incomplete")
        C.write(ROOT / "COMPLETE.json", {"phase": "complete", "completed": 240, "expected": 240,
                "plan_sha256": C.sha(ROOT / "plan.json"), "native_complete_sha256": C.sha(C.ROOT / "COMPLETE.json"), "completed_utc": C.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "preflight", "pilot", "run", "finalize"):
        stages.add_argument("--"+flag, action="store_true")
    parser.add_argument("--worker", type=int, choices=range(8))
    args = parser.parse_args()
    if args.run and args.worker is None:
        parser.error("--run requires --worker")
    if args.prepare: prepare()
    elif args.preflight: preflight()
    elif args.pilot: pilot()
    elif args.run: run(args.worker)
    else: finalize()


if __name__ == "__main__":
    main()
