#!/usr/bin/env python
"""Audit classification ownership before and after the deferred GPU handoff.

The audit is read-only. It does not stop workers, remove locks, or launch jobs.
It treats completed JSON files as authoritative and reports duplicate logical
cells separately from transient lock files.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "03_classification" / "results"
OUT = HERE / "results"


def process_rows():
    raw = subprocess.run(["ps", "-eo", "pid=,stat=,comm=,args="],
                         check=True, capture_output=True, text=True).stdout
    rows = []
    for line in raw.splitlines():
        p = line.strip().split(None, 3)
        if len(p) == 4:
            rows.append({"pid": int(p[0]), "stat": p[1], "comm": p[2], "args": p[3]})
    return rows


def active_jobs(rows):
    def py(a):
        return "python" in a["comm"]

    return {
        "tpl_ub_benchmark": [r for r in rows if py(r) and "classify.py" in r["args"]
                             and "--study benchmark" in r["args"] and "--datasets tpl_ub" in r["args"]],
        "tpl_b_benchmark": [r for r in rows if py(r) and "classify.py" in r["args"]
                            and "--study benchmark" in r["args"] and "--datasets tpl_b" in r["args"]],
        "cpu_factorial_xgb": [r for r in rows if py(r) and "classify.py" in r["args"]
                              and "--study factorial" in r["args"] and "--models xgb" in r["args"]],
        "deferred_scheduler": [r for r in rows if "deferred_gpu_safe.sh" in r["args"]],
    }


def json_cells(study):
    root = RESULTS / study
    records = []
    if not root.exists():
        return records
    for path in root.rglob("*.json"):
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if all(k in obj for k in ("dataset", "scheme", "seed", "fingerprint", "model")):
            records.append((obj, path))
    return records


def main():
    rows = process_rows()
    jobs = active_jobs(rows)
    payload = {"jobs": {k: [{"pid": r["pid"], "stat": r["stat"]} for r in v]
                         for k, v in jobs.items()}}
    for study in ("benchmark", "factorial"):
        records = json_cells(study)
        keys = [(r["dataset"], r["scheme"], int(r["seed"]), r["fingerprint"], r["model"])
                for r, _ in records]
        counts = Counter(keys)
        payload[study] = {
            "json_files": len(records),
            "duplicate_logical_cells": sum(n - 1 for n in counts.values() if n > 1),
            "duplicate_examples": [list(k) for k, n in counts.items() if n > 1][:5],
        }

    locks = list(RESULTS.rglob("*.json.lock")) if RESULTS.exists() else []
    payload["locks"] = {"count": len(locks), "examples": [str(p) for p in locks[:10]]}
    tpl_active = bool(jobs["tpl_ub_benchmark"] or jobs["tpl_b_benchmark"])
    cpu_active = bool(jobs["cpu_factorial_xgb"])
    payload["handoff"] = {
        "tpl_workers_active": tpl_active,
        "cpu_factorial_workers_active": cpu_active,
        "scheduler_active": bool(jobs["deferred_scheduler"]),
        "safe_to_reclaim_cpu_now": not tpl_active and cpu_active,
        "duplicate_free": all(payload[s]["duplicate_logical_cells"] == 0
                               for s in ("benchmark", "factorial")),
        "status": "waiting_for_tpl" if tpl_active else ("reclaim_ready" if cpu_active else "post_reclaim"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "handoff_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Classification handoff audit", "", f"Status: **{payload['handoff']['status']}**", "",
             "| Check | Value |", "|---|---:|"]
    lines += [f"| TPL workers active | {payload['handoff']['tpl_workers_active']} |",
              f"| CPU factorial XGB active | {payload['handoff']['cpu_factorial_workers_active']} |",
              f"| Deferred scheduler active | {payload['handoff']['scheduler_active']} |",
              f"| JSON locks | {payload['locks']['count']} |",
              f"| Duplicate logical cells | {payload['factorial']['duplicate_logical_cells'] + payload['benchmark']['duplicate_logical_cells']} |",
              f"| Safe to reclaim CPU now | {payload['handoff']['safe_to_reclaim_cpu_now']} |"]
    (OUT / "HANDOFF_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
