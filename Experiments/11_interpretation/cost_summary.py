#!/usr/bin/env python
"""Summarize cached feature size, validity, and density for label-free FPs."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main() -> None:
    src = OUT / "feature_validity_sparsity.csv"
    d = pd.read_csv(src)
    # Cached matrices are dense NumPy arrays; estimate their resident payload
    # from the recorded shape and dtype, independent of filesystem compression.
    d["bytes_per_value"] = d["dtype"].map({"uint8": 1, "float32": 4, "float64": 8})
    if d["bytes_per_value"].isna().any():
        raise ValueError("unknown dtype in feature audit")
    d["estimated_payload_mb"] = (
        d["n_rows"] * d["n_features"] * d["bytes_per_value"] / 2**20
    )
    d["valid_fraction"] = d["n_ok"] / d["n_rows"]
    cols = ["dataset", "fingerprint", "n_rows", "n_features", "dtype",
            "estimated_payload_mb", "valid_fraction", "zero_rate",
            "mean_nonzero_units"]
    d[cols].to_csv(OUT / "feature_cost_summary.csv", index=False)
    grouped = (d.groupby("fingerprint", as_index=False)
                 .agg(datasets=("dataset", "nunique"),
                      total_payload_mb=("estimated_payload_mb", "sum"),
                      mean_valid_fraction=("valid_fraction", "mean"),
                      mean_zero_rate=("zero_rate", "mean"),
                      mean_nonzero_units=("mean_nonzero_units", "mean")))
    grouped.to_csv(OUT / "feature_cost_by_fingerprint.csv", index=False)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
