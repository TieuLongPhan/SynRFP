#!/usr/bin/env python3
"""Nested grouped r2 MLP selection screen for SynRFP and DRFP.

Each outer fold gets its own three grouped inner folds.  No outer test labels
are supplied to the CUDA trial processes.  A later refit stage consumes only
the selected inner-validation records.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYN_PYTHON = Path("/home/labhhc4/anaconda3/envs/synrfp/bin/python")
CUDA_PYTHON = Path("/home/labhhc4/anaconda3/envs/rxnfp/bin/python")
CONFIGS = [
    ("synrfp-wl-r2-b1024-parity-delta", "plain", 0.0, 0.0003),
    ("synrfp-wl-r2-b1024-parity-delta", "plain", 0.0, 0.001),
    ("synrfp-wl-r2-b1024-parity-delta", "auxiliary", 0.05, 0.001),
    ("drfp", "plain", 0.0, 0.0003),
    ("drfp", "plain", 0.0, 0.001),
    ("drfp", "auxiliary", 0.02, 0.001),
]


def name(outer, inner, fp, family, aux, lr):
    short = "synrfp" if fp.startswith("synrfp") else fp
    return "outer%d_inner%d_%s_%s_aux%s_lr%s" % (
        outer, inner, short, family, str(aux).replace(".", "p"), str(lr).replace(".", "p"))


def prepare(out, outer):
    split = out / ("outer%d_inner.npy" % outer)
    if split.exists() and split.with_suffix(".json").exists():
        return split
    cmd = [str(SYN_PYTHON), str(ROOT / "03_classification" / "r2_grouped_dev_prepare.py"),
           "--outer-seed", str(outer), "--inner-seed", str(1000 + outer), "--out", str(split)]
    subprocess.run(cmd, check=True)
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {outer: prepare(args.out, outer) for outer in range(25)}
    jobs = [(outer, inner, fp, family, aux, lr)
            for outer in range(25) for inner in range(3)
            for fp, family, aux, lr in CONFIGS]
    pending = [job for job in jobs if not (args.out / (name(*job) + ".json")).exists()]
    status = {"study": "r2_nested_grouped_mlp_v1", "stage": "inner_selection",
              "outer_test_read": False, "candidate_count_per_fingerprint": 3,
              "total_jobs": len(jobs), "pending": [name(*j) for j in pending],
              "finished": [], "failed": []}
    (args.out / "status.json").write_text(json.dumps(status, indent=2))
    active = {}
    while pending or active:
        for device in args.devices:
            if device in active or not pending:
                continue
            outer, inner, fp, family, aux, lr = pending.pop(0)
            job_name = name(outer, inner, fp, family, aux, lr)
            output = args.out / (job_name + ".json")
            log = output.with_suffix(".log").open("w")
            cmd = [str(CUDA_PYTHON), "-u", str(ROOT / "03_classification" / "r2_grouped_dev_trial.py"),
                   "--family", family, "--aux-weight", str(aux), "--fingerprint", fp,
                   "--outer-seed", str(outer), "--inner-split", str(splits[outer]),
                   "--inner-fold", str(inner), "--width", "1024", "--dropout", "0.1",
                   "--batch-size", "256", "--lr", str(lr), "--device", "cuda:%d" % device,
                   "--out", str(output)]
            active[device] = (job_name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(1)
        for device, (job_name, process, log) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            (status["finished"] if process.returncode == 0 else status["failed"]).append(job_name)
            del active[device]
            status["pending"] = [name(*j) for j in pending]
            status["active"] = {str(k): value[0] for k, value in active.items()}
            (args.out / "status.json").write_text(json.dumps(status, indent=2))
    status["status"] = "complete" if not status["failed"] else "failed"
    (args.out / "status.json").write_text(json.dumps(status, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
