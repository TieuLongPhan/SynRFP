"""Run resumable full-cohort BitBIRCH using released fingerprint matrices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import sys
import time

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from prepare91 import HERE, REPO, ROOT, SEEDS, STANDARD, THRESHOLDS, read, sha, write

sys.path.insert(0, str(REPO / "Experiments"))
from synrfp_exp.clustering import bitbirch
from synrfp_exp.features import Features

STATE = {}


def load(path):
    return np.load(path, allow_pickle=False, mmap_mode="r")


def matrix(info, source):
    path = info["path"]
    if sha(path) != info["sha256"]:
        raise ValueError(f"Fingerprint cache contents changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        x, ok = data["X"], data["ok"]
        n = source["n"]
        assert len(x) == len(ok) == n and ok.dtype == bool and not np.any(x[~ok])
        if info["has_row_identity"]:
            np.testing.assert_array_equal(data["source_positions"], np.arange(n))
            np.testing.assert_array_equal(data["row_ids"], load(source["row_ids"]))
        conv = info["conversion"]
        if conv == "native":
            assert x.dtype == np.uint8
            return x, ok
        assert np.isfinite(x).all()
        return np.packbits(x > 0 if conv == "positive" else x != 0, axis=1), ok


def identity(source, group, fp, seed):
    return ROOT / "results" / source / f"{group}__{fp}__{seed}.json"


def score(y, assignment, size):
    return dict(ami=float(STATE.get("ami", adjusted_mutual_info_score)(y, assignment)),
                ari=float(adjusted_rand_score(y, assignment)),
                nmi=float(normalized_mutual_info_score(y, assignment)),
                n_clusters=int(np.count_nonzero(size)),
                singleton_share=float(np.count_nonzero(size == 1) / len(y)),
                largest_cluster_share=float(size.max() / len(y)))


def existing(path, expected):
    if path.exists():
        value = read(path)
        for key, v in expected.items():
            if value[key] != v:
                raise ValueError(f"Changed key or inputs for {path}: {key}")
        return value
    return expected | {"scores": {}}


def job(work):
    source_name, group_name, fp, seeds, thresholds = work
    plan, plan_sha = STATE["plan"], STATE["plan_sha"]
    source = plan["sources"][source_name]
    feature = source["feature_files"][fp]
    group = source["groups"][group_name]
    header = dict(plan_sha256=plan_sha, feature_sha256=feature["sha256"],
                  representation=fp, conversion=feature["conversion"],
                  source=source_name, population=group_name)
    pending = []
    for seed in seeds:
        path = identity(source_name, group_name, fp, seed)
        data = existing(path, header | {"order_seed": seed})
        if any(str(t) not in data["scores"] for t in thresholds):
            pending.append((seed, path, data))
    if not pending:
        return 0, 0.
    x, ok = matrix(feature, source)
    rows = load(group["rows"])
    labels = {name: load(path) for name, path in group["endpoints"].items()}
    assert len(rows) == group["n"] and all(y.shape == rows.shape for y in labels.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    completed, cpu_seconds = 0, 0.
    for seed, path, data in pending:
        order = np.random.default_rng(seed).permutation(len(rows))
        take = rows[order]
        f = Features(x[take], 8 * x.shape[1], ok[take])
        refused = int((~f.ok).sum())
        ordered = {name: y[order] for name, y in labels.items()}
        for t in thresholds:
            if str(t) in data["scores"]:
                continue
            start = time.monotonic()
            assignment = bitbirch(f, t / 100, plan["branching_factor"], plan["merge_criterion"])
            fit_time = time.monotonic() - start
            counts = np.bincount(assignment)
            digest = sha256(assignment).hexdigest()
            stats = [{"endpoint": name, **score(y, assignment, counts)} for name, y in ordered.items()]
            data["scores"][str(t)] = dict(fit_seconds=fit_time,
                cpu_batch_seconds=time.monotonic()-start, n_rows=len(rows),
                ami_arithmetic="grouped_hypergeometric" if "ami" in STATE else "sklearn_native",
                n_refusal_placeholders=refused, assignment_sha256=digest, endpoints=stats)
            write(path, data)
            completed += 1
            cpu_seconds += data["scores"][str(t)]["cpu_batch_seconds"]
            if STATE["pilot"]:
                print("Pilot", group_name, fp, t, len(rows), round(fit_time, 3),
                      "fit seconds and", round(data["scores"][str(t)]["cpu_batch_seconds"], 3),
                      "fit plus scoring seconds", flush=True)
    return completed, cpu_seconds


def works(plan, pilot):
    result = []
    for source_name, source in plan["sources"].items():
        for group_name, g in source["groups"].items():
            if pilot and group_name not in ("tpl_b", "tpl_ub", "ecreact_1", "uspto50k_b", "mech_2"):
                continue
            for fp in source["feature_files"]:
                if pilot and fp not in (STANDARD, "synrfp-morgan-r3-b2048-cw-roles",
                                        "rdkit_struct", "rxnfp_mlm"):
                    continue
                result.append((source_name, group_name, fp, SEEDS[:1] if pilot else SEEDS,
                               (20,35,80) if pilot else (35, *[t for t in THRESHOLDS if t != 35])))
    result.sort(key=lambda a: (-plan["sources"][a[0]]["groups"][a[1]]["n"], a[1], a[2]))
    return result


def init():
    f = ROOT / "PREPARED.json"
    plan = read(f)
    for path, expected in plan["pinned_inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"Changed input file {path}")
    assert all(version(name) == expected for name, expected in plan["versions"].items())
    STATE.update(plan=plan, plan_sha=sha(f))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--jobs", type=int, required=True)
    cli.add_argument("--pilot", action="store_true")
    cli.add_argument("--grouped-ami", action="store_true")
    args = cli.parse_args()
    init()
    if args.grouped_ami:
        from metrics91 import adjusted_info
        proof = read(ROOT/"GROUPED_AMI_VERIFIED.json")
        assert proof["metrics_sha256"] == sha(HERE/"metrics91.py")
        assert proof["max_absolute_error"] <= 1e-8 and len(proof["controls"]) >= 10
        adjusted_info(np.array([0,0,1,1]), np.array([0,1,1,0]))
        STATE["ami"] = adjusted_info
    STATE["pilot"] = args.pilot
    plan = STATE["plan"]
    tickets = works(plan, args.pilot)
    start = time.monotonic()
    done, seconds = 0, 0.
    failures = []
    # Keep the bounded submission queue in input order while releasing finished
    # batches out of order, so a slow large population never blocks a worker.
    from concurrent.futures import FIRST_COMPLETED, wait
    pending = iter(tickets)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        inflight = {pool.submit(job, ticket): ticket
                    for ticket in (next(pending, None) for _ in range(args.jobs)) if ticket}
        finished = 0
        while inflight:
            completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in completed:
                ticket = inflight.pop(future)
                try:
                    n, wall = future.result()
                except Exception as exc:
                    failures.append(dict(ticket=ticket, error=repr(exc)))
                    print("Failed", ticket[:3], repr(exc), flush=True)
                else:
                    done += n
                    seconds += wall
                finished += 1
                ticket = next(pending, None)
                if ticket:
                    inflight[pool.submit(job, ticket)] = ticket
            snapshot = dict(tickets=len(tickets), finished_tickets=finished, new_fits=done,
                            failures=failures, elapsed_seconds=time.monotonic()-start,
                            plan_sha256=STATE["plan_sha"], jobs=args.jobs,
                            grouped_ami=args.grouped_ami,
                            fit_and_score_worker_seconds=seconds)
            write(ROOT / ("PILOT_PROGRESS.json" if args.pilot else "PROGRESS.json"), snapshot)
            print("Batches", finished, "/", len(tickets), "updated fits", done, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} batches require recovery")
    write(ROOT / ("PILOT_COMPLETE.json" if args.pilot else "FIT_COMPLETE.json"), snapshot)


if __name__ == "__main__":
    main()
