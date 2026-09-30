#!/usr/bin/env python3
"""Queue the frozen first-stage r2 grouped development screen on two GPUs."""
import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_PYTHON = Path("/home/labhhc4/anaconda3/envs/rxnfp/bin/python")
FPS = ["synrfp-wl-r2-b1024-parity-delta", "drfp"]
SETTINGS = [
    {"width": 512, "dropout": 0.0, "batch_size": batch, "lr": lr}
    for batch in (256, 1024) for lr in (0.0003, 0.001)
] + [
    {"width": 1024, "dropout": 0.1, "batch_size": batch, "lr": lr}
    for batch in (256, 1024) for lr in (0.0003, 0.001)
]


def identifier(fp, s):
    short = "synrfp" if fp.startswith("synrfp") else fp
    return "%s_w%d_d%s_b%d_lr%s" % (
        short, s["width"], str(s["dropout"]).replace(".", "p"),
        s["batch_size"], str(s["lr"]).replace(".", "p"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(fp, setting) for setting in SETTINGS for fp in FPS]
    pending = [(fp, s) for fp, s in jobs if not (args.out / (identifier(fp, s) + ".json")).exists()]
    status = {"study": "r2_grouped_development_screen_v1", "outer_test_read": False,
              "outer_seed": 0, "inner_fold": 0, "inner_split": str(args.inner_split),
              "settings_per_fingerprint": len(SETTINGS), "pending": [identifier(*j) for j in pending],
              "finished": [], "failed": []}
    (args.out / "queue_status.json").write_text(json.dumps(status, indent=2))
    active = {}
    while pending or active:
        for device in args.devices:
            if device in active or not pending:
                continue
            fp, setting = pending.pop(0)
            name = identifier(fp, setting)
            out = args.out / (name + ".json")
            log = out.with_suffix(".log").open("w")
            cmd = [str(CUDA_PYTHON), "-u", str(ROOT / "03_classification" / "r2_grouped_dev_trial.py"),
                   "--fingerprint", fp, "--inner-split", str(args.inner_split), "--inner-fold", "0",
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
            status["pending"] = [identifier(*j) for j in pending]
            status["active"] = {str(k): v[0] for k, v in active.items()}
            (args.out / "queue_status.json").write_text(json.dumps(status, indent=2))
    status["status"] = "complete" if not status["failed"] else "failed"
    (args.out / "queue_status.json").write_text(json.dumps(status, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
