#!/usr/bin/env python3
"""Repeat the preselected r2 development finalists with initialization seed 1."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from r2_grouped_dev_screen_queue import CUDA_PYTHON, identifier

ROOT = Path(__file__).resolve().parents[1]
FINALISTS = [
    ("synrfp-wl-r2-b1024-parity-delta", {"width": 1024, "dropout": 0.1, "batch_size": 256, "lr": 0.001}),
    ("synrfp-wl-r2-b1024-parity-delta", {"width": 1024, "dropout": 0.1, "batch_size": 256, "lr": 0.0003}),
    ("drfp", {"width": 1024, "dropout": 0.1, "batch_size": 256, "lr": 0.001}),
    ("drfp", {"width": 1024, "dropout": 0.1, "batch_size": 256, "lr": 0.0003}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    jobs = [(fp, settings, fold) for fp, settings in FINALISTS for fold in range(3)]
    pending = [(fp, settings, fold) for fp, settings, fold in jobs if not (
        args.out / ("init1_inner%d_" % fold + identifier(fp, settings) + ".json")
    ).exists()]
    status = {"study": "r2_grouped_development_screen_v1", "stage": "finalists_init_seed_1",
              "outer_test_read": False, "pending": ["init1_inner%d_%s" % (j[2], identifier(j[0], j[1])) for j in pending],
              "finished": [], "failed": []}
    (args.out / "seed1_status.json").write_text(json.dumps(status, indent=2))
    active = {}
    while pending or active:
        for device in args.devices:
            if device in active or not pending:
                continue
            fp, settings, fold = pending.pop(0)
            name = "init1_inner%d_%s" % (fold, identifier(fp, settings))
            out = args.out / (name + ".json")
            log = out.with_suffix(".log").open("w")
            cmd = [str(CUDA_PYTHON), "-u", str(ROOT / "03_classification" / "r2_grouped_dev_trial.py"),
                   "--fingerprint", fp, "--inner-split", str(args.inner_split), "--inner-fold", str(fold),
                   "--width", str(settings["width"]), "--dropout", str(settings["dropout"]),
                   "--batch-size", str(settings["batch_size"]), "--lr", str(settings["lr"]),
                   "--init-seed", "1", "--device", "cuda:%d" % device, "--out", str(out)]
            active[device] = (name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(1)
        for device, (name, proc, log) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            (status["finished"] if proc.returncode == 0 else status["failed"]).append(name)
            del active[device]
            status["pending"] = ["init1_inner%d_%s" % (j[2], identifier(j[0], j[1])) for j in pending]
            status["active"] = {str(k): v[0] for k, v in active.items()}
            (args.out / "seed1_status.json").write_text(json.dumps(status, indent=2))
    status["status"] = "complete" if not status["failed"] else "failed"
    (args.out / "seed1_status.json").write_text(json.dumps(status, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
