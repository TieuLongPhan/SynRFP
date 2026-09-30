"""Paired uncertainty summaries for completed clustering and kNN contrasts.

This is descriptive inference on the already completed paired runs.  The
bootstrap resamples paired rows within an endpoint; it does not treat
endpoints or overlapping subsamples as independent chemical datasets.
"""
from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Experiments/11_interpretation/results"
RNG = np.random.default_rng(42)


def sign_p(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    n = len(values)
    if n == 0:
        return np.nan
    k = int((values > 0).sum())
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(group, keys, contrast):
    x = np.asarray(group[contrast], dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return None
    draws = x[RNG.integers(0, n, size=(20000, n))].mean(axis=1)
    return {
        **{k: group.iloc[0][k] for k in keys},
        "n": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int((x > 0).sum()),
        "ties": int((x == 0).sum()),
        "exact_sign_p_two_sided": float(sign_p(x)),
    }


def main():
    rows = []
    st = pd.read_csv(OUT / "clustering_stability_paired_deltas.csv")
    for (dataset, baseline, metric), g in st.groupby(["dataset", "baseline", "metric"], sort=True):
        row = summarize(g, ["dataset", "baseline", "metric"], "delta")
        if row:
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "clustering_paired_inference.csv", index=False)

    knn = pd.read_csv(OUT / "knn_mode_contrast_by_split.csv")
    rows = []
    for dataset, g in knn.groupby("dataset", sort=True):
        row = summarize(g, ["dataset"], "delta_minus_union")
        if row:
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "knn_mode_paired_inference.csv", index=False)
    print(f"Wrote {len(rows)} kNN and {len(pd.read_csv(OUT / 'clustering_paired_inference.csv'))} clustering summaries")


if __name__ == "__main__":
    main()
