"""Estimate the remaining TPL XGBoost tail from observed per-cell durations."""
from pathlib import Path
import json
import re
import statistics
import sys
from datetime import datetime

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
sys.path.insert(0, str(EXP))
from synrfp_exp.config import load_config
from importlib.machinery import SourceFileLoader

QUEUE = SourceFileLoader("cpu_queue", str(EXP / "03_classification" / "cpu_benchmark_queue.py")).load_module()


def durations(dataset):
    path = EXP / "03_classification" / "results" / "cpu_parallel_finish" / f"xgb_gpu_tpl_{dataset}.log"
    starts, values = {}, {}
    completed = set()
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d) INFO (START )?tpl_" + dataset + r" cv s(\d+) (\S+) xgb", line)
        if m:
            starts[(int(m[5]), m[6])] = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        m = re.match(r"(\d\d):(\d\d):(\d\d) INFO tpl_" + dataset + r" cv s(\d+) (\S+)\s+xgb\s+MCC=", line)
        if m:
            key = (int(m[4]), m[5])
            completed.add(key)
            if key in starts:
                values.setdefault(m[5], []).append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) - starts[key]) % 86400)
    active = [key for key in starts if key not in completed]
    return values, completed, active, starts


def main():
    cfg = load_config(str(EXP / "config.yaml"))
    estimates = {}
    all_medians = []
    for ds in ("ub", "b"):
        observed, completed, active, starts = durations(ds)
        med = {fp: statistics.median(vals) for fp, vals in observed.items() if vals}
        all_medians.extend(med.values())
        estimates[ds] = (med, observed, completed, active, starts)
    fallback = statistics.median(all_medians)
    expected, _ = QUEUE.expected_results(cfg, "xgb")
    missing = QUEUE.missing_results(expected, "xgb")
    by_ds = {"tpl_ub": [], "tpl_b": []}
    for name in missing:
        p = Path(name)
        dataset = p.parts[-3]
        if dataset not in by_ds:
            continue
        fp = p.name.split("__xgb.json")[0]
        ds_short = dataset.removeprefix("tpl_")
        median = estimates[ds_short][0].get(fp, fallback)
        by_ds[dataset].append(median)
    result = {
        "method": "median observed duration by fingerprint; independent UB/B workers",
        "missing_cells": {k: len(v) for k, v in by_ds.items()},
        "completed_cells_in_log": {
            f"tpl_{ds}": len(estimates[ds][2]) for ds in ("ub", "b")
        },
        "active_cell": {
            f"tpl_{ds}": ([{"split": s, "fingerprint": fp} for s, fp in estimates[ds][3]] or None)
            for ds in ("ub", "b")
        },
        "active_elapsed_hours": {
            f"tpl_{ds}": [round(((datetime.now().hour * 3600 + datetime.now().minute * 60 + datetime.now().second) - estimates[ds][4][(s, fp)]) % 86400 / 3600, 2)
                             for s, fp in estimates[ds][3]]
            for ds in ("ub", "b")
        },
        "estimated_remaining_hours": {k: round(sum(v) / 3600, 1) for k, v in by_ds.items()},
        "wall_hours_if_both_workers_remain_active": round(max(sum(v) for v in by_ds.values()) / 3600, 1),
        "caveat": "The currently running cells may be partially complete; this is a conservative tail estimate, not a guarantee.",
    }
    out = HERE / "results" / "tpl_tail_eta.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
