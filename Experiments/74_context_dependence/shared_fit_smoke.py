"""Check bit-exact predictions and measure shared-matrix worker cost on training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from joblib import Parallel, delayed
import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from audit_inputs import sha256
from run_quantumpioneer import ARMS, e1, fit_cell, matrices
from freeze_qp_prediction import verified_artifacts


def check_fit_replay(features, matrices_root, cohort):
    arm = "d2048"
    parts = tuple(cohort["train_indices"][start:stop] for start, stop in
                  ((0, 320), (320, 384), (384, 448)))
    y = cohort["train_targets"][:384, 0]
    packed = np.load(features / f"{arm}.npy", mmap_mode="r", allow_pickle=False)
    arrays = matrices(packed, parts)
    # Compare the same training-only source positions in the shared cache.
    cache = np.load(matrices_root / f"{arm}__train.npy", mmap_mode="r", allow_pickle=False)
    np.testing.assert_array_equal(cache[:448], np.concatenate(arrays))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "private_path").mkdir()
        (root / "shared_path").mkdir()
        for split, values in zip(("train", "validation", "test"), arrays, strict=True):
            np.save(root / f"{split}.npy", values, allow_pickle=False)
        paths = tuple(root / f"{split}.npy" for split in ("train", "validation", "test"))
        common = (parts, y[:320], y[320:], arm, "barrier", 17)
        original = fit_cell(packed, *common, root / "private_path", "smoke")
        shared = fit_cell(None, *common, root / "shared_path", "smoke", paths)
        for key in ("epochs", "best_epoch", "validation_standardized_mse", "outputs_sha256"):
            if original[key] != shared[key]:
                raise ValueError(f"Matrix sharing changed {key}")
    return original["epochs"]


def one_epoch(path, y, arm, seed):
    start = time.monotonic()
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    target = (y - y.mean()) / y.std()
    model = MLPRegressor(**e1.FIT_OPTIONS, random_state=np.random.RandomState(seed))
    with threadpool_limits(limits=1):
        model.partial_fit(x, target)
    return {"arm": arm, "seed": seed, "rows": len(x), "epoch_seconds": time.monotonic()-start}


def memory_peak():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::"))
    path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    return int((path / "memory.peak").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "frozen", "shared", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    verified_artifacts(args.shared)
    with np.load(args.frozen / "cohort.npz", allow_pickle=False) as file:
        cohort = {key: file[key] for key in ("train_targets", "train_indices")}
    epochs = check_fit_replay(args.features, args.shared, cohort)
    print(f"Shared and private fixtures produced identical checkpoints after {epochs} epochs.", flush=True)
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_epoch)(args.shared / f"{arm}__train.npy", cohort["train_targets"][:, i % 2],
                           arm, 17 + i)
        for i, arm in enumerate((ARMS * args.jobs)[:args.jobs])
    )
    receipt = {
        "phase": "shared_training_pilot_complete",
        "jobs": args.jobs,
        "shared_matrix_receipt_sha256": sha256(args.shared / "COMPLETE.json"),
        "bit_exact_checkpoint_and_prediction_replay": True,
        "pilot_training_epochs": rows, "memory_peak_bytes": memory_peak(),
        "heldout_outcomes_opened": False,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
