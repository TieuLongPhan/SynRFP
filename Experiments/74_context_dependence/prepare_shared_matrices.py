"""Decode the frozen matrices once and verify their complete row order and values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_inputs import sha256
from freeze_qp_prediction import verified_artifacts

PARTITIONS = ("train", "validation", "test")


def prepare(features, frozen, output):
    packed = verified_artifacts(features)
    prediction_path = frozen / "prediction.json"
    prediction = json.loads(prediction_path.read_text())
    if prediction["feature_receipt_sha256"] != sha256(features / "COMPLETE.json"):
        raise ValueError("Packed feature receipt differs from the frozen prediction")
    cohort_file = frozen / "cohort.npz"
    if sha256(cohort_file) != prediction["artifact_sha256"]["cohort.npz"]:
        raise ValueError("Cohort membership changed")
    output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    with np.load(cohort_file, allow_pickle=False) as cohort:
        indices = {part: cohort[f"{part}_indices"] for part in PARTITIONS}
    for arm, width in packed["widths"].items():
        archive = np.load(features / f"{arm}.npy", mmap_mode="r", allow_pickle=False)
        for part, rows in indices.items():
            path = output / f"{arm}__{part}.npy"
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                shape=(len(rows), width))
            for start in range(0, len(rows), 4096):
                chunk = rows[start:start + 4096]
                matrix[start:start + len(chunk)] = np.unpackbits(archive[chunk], axis=1)
            matrix.flush()
            del matrix
            verification = np.load(path, mmap_mode="r", allow_pickle=False)
            for start in range(0, len(rows), 4096):
                chunk = rows[start:start + 4096]
                np.testing.assert_array_equal(
                    verification[start:start + len(chunk)], np.unpackbits(archive[chunk], axis=1)
                )
            outputs[path.name] = sha256(path)
        print(f"Verified {arm}: three frozen partition matrices", flush=True)
    payload = {
        "phase": "shared_matrices_verified",
        "feature_receipt_sha256": sha256(features / "COMPLETE.json"),
        "frozen_prediction_sha256": sha256(prediction_path),
        "outputs_sha256": outputs,
    }
    receipt = output / "COMPLETE.json"
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "frozen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.features, args.frozen, args.output)


if __name__ == "__main__":
    main()
