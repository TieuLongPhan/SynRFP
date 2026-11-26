"""Read only the prepared training state and shared feature archive for an EC-level fit."""
import argparse
import fcntl
from hashlib import file_digest
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",type=Path)
    args=p.parse_args()
    task=args.task
    with (task/".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (task/"COMPLETE.json").exists():
            return
        train(task)


def train(task):
    job=json.loads((task/"job.json").read_text())
    root=task.parents[2]
    state=task.parent/"state.npz"
    indices=task.parent/"selectors.npz"
    native=Path(__file__).resolve().parents[3]/"SynRFP/Experiments/16_modes/classification_worker.py"
    packed=root/f"{job['arm']}.npy"
    for path,tag in ((native,"native_worker"),(packed,"packed"),(state,"state"),(indices,"selectors")):
        assert sha(path)==job[f"{tag}_sha256"]
    attempt=task/f"attempt-{time.time_ns()}"
    attempt.mkdir()
    (attempt/"state.npz").symlink_to(state)
    (attempt/"job.json").symlink_to(task/"job.json")
    spec=importlib.util.spec_from_file_location("archived_cpu_worker",native)
    worker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    with np.load(indices,allow_pickle=False) as z:
        selector={k:z[k] for k in ("outer","test")}
        classes=z["classes"]
    matrix=np.load(packed,mmap_mode="r",allow_pickle=False)

    def unpack(directory,stem):
        rows=selector["outer" if stem=="train" else "test"]
        return np.unpackbits(matrix[rows],axis=1).astype(np.float32)
    worker.load_matrix=unpack
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out=worker.train(attempt)
    preds=np.load(attempt/"predictions.npy",allow_pickle=False)
    assert preds.shape==selector["test"].shape and np.all((preds>=0)&(preds<len(classes)))
    np.save(task/"decoded.npy",classes[preds],allow_pickle=False)
    files=[attempt/"best_weights.npz",attempt/"predictions.npy",task/"decoded.npy"]
    out.update(plan_job_sha256=sha(task/"job.json"),files_sha256={str(f.relative_to(task)):sha(f) for f in files})
    temp=task/".COMPLETE.tmp"
    temp.write_text(json.dumps(out,indent=2)+"\n")
    temp.replace(task/"COMPLETE.json")


if __name__=="__main__":
    main()
