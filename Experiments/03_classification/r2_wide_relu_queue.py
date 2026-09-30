#!/usr/bin/env python3
"""Run the pre-selected SynTemp-r2 MLP head across benchmark outer folds.

The wider ReLU head was selected from outer-validation MCC over seeds 0--4.
This launcher refits the frozen choice on each outer training partition and
evaluates it once on the corresponding untouched outer test fold.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RXN_PYTHON = Path("/home/labhhc4/anaconda3/envs/rxnfp/bin/python")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pending = [s for s in range(25) if not (args.out / ("syntemp_r2_s%d_wide_relu.json" % s)).exists()]
    status = {"study": "r2_mlp_tuning_evaluation", "candidate": "wide_relu",
              "selection": "outer validation MCC over seeds 0--4", "test_used_for_selection": False,
              "refit": "outer train+validation with a fresh 10% inner early-stop split",
              "pending": pending, "finished": [], "failed": []}
    (args.out / "status.json").write_text(json.dumps(status, indent=2))
    running = {}
    while pending or running:
        for device in args.devices:
            if device in running or not pending:
                continue
            seed = pending.pop(0)
            out = args.out / ("syntemp_r2_s%d_wide_relu.json" % seed)
            log = out.with_suffix(".log").open("w")
            cmd = [str(RXN_PYTHON), "-u", str(ROOT / "03_classification" / "mlp_model_selection.py"),
                   "--dataset", "syntemp_r2", "--seed", str(seed), "--candidate", "wide_relu",
                   "--device", "cuda:%d" % device, "--final-train-outer", "--evaluate-test", "--out", str(out)]
            running[device] = (seed, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(1)
        for device, (seed, process, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            (status["finished"] if process.returncode == 0 else status["failed"]).append(seed)
            del running[device]
            status["pending"] = pending
            (args.out / "status.json").write_text(json.dumps(status, indent=2))
    status["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (args.out / "status.json").write_text(json.dumps(status, indent=2))
    if status["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
