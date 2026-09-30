#!/usr/bin/env python3
"""Validate the six strongest first-stage r2 settings on the other inner folds."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from r2_grouped_dev_screen_queue import CUDA_PYTHON, FPS, identifier


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    selected = []
    for fp in FPS:
        initial = []
        for path in args.out.glob("*_w*.json"):
            rec = json.loads(path.read_text())
            if rec.get("fingerprint") == fp and rec.get("inner_fold") == 0:
                initial.append(rec)
        if len(initial) != 8:
            raise RuntimeError("expected 8 completed stage-one results for %s, got %d" % (fp, len(initial)))
        initial.sort(key=lambda r: r["validation_mcc"], reverse=True)
        selected.extend((fp, rec["params"]) for rec in initial[:6])
    jobs = [(fp, setting, fold) for fp, setting in selected for fold in (1, 2)]
    pending = [(fp, setting, fold) for fp, setting, fold in jobs
               if not (args.out / ("inner%d_" % fold + identifier(fp, setting) + ".json")).exists()]
    status = {"study": "r2_grouped_development_screen_v1", "stage": "top6_cross_inner_validation",
              "outer_test_read": False, "pending": ["inner%d_%s" % (j[2], identifier(j[0], j[1])) for j in pending],
              "finished": [], "failed": []}
    (args.out / "advance_status.json").write_text(json.dumps(status, indent=2))
    active = {}
    while pending or active:
        for device in args.devices:
            if device in active or not pending:
                continue
            fp, setting, fold = pending.pop(0)
            name = "inner%d_%s" % (fold, identifier(fp, setting))
            out = args.out / (name + ".json")
            log = out.with_suffix(".log").open("w")
            cmd = [str(CUDA_PYTHON), "-u", str(ROOT / "03_classification" / "r2_grouped_dev_trial.py"),
                   "--fingerprint", fp, "--inner-split", str(args.inner_split), "--inner-fold", str(fold),
                   "--width", str(setting["width"]), "--dropout", str(setting["dropout"]),
                   "--batch-size", str(setting["batch_size"]), "--lr", str(setting["lr"]),
                   "--device", "cuda:%d" % device, "--out", str(out)]
            active[device] = (name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(1)
        for device, (name, proc, log) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            (status["finished"] if proc.returncode == 0 else status["failed"]).append(name)
            del active[device]
            status["pending"] = ["inner%d_%s" % (j[2], identifier(j[0], j[1])) for j in pending]
            status["active"] = {str(k): v[0] for k, v in active.items()}
            (args.out / "advance_status.json").write_text(json.dumps(status, indent=2))
    status["status"] = "complete" if not status["failed"] else "failed"
    (args.out / "advance_status.json").write_text(json.dumps(status, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
