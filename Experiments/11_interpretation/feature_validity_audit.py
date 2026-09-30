"""Summarize cached feature validity and sparsity; never fits a model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FEATURES = HERE.parents[0] / "work" / "features"
OUT = HERE / "results"

# These are the label-free representations used by the clustering benchmark.
TARGETS = {
    "schneider_b": "schneider_b",
    "uspto50k_b": "uspto_50k_b",
    "syntemp_r0": "syntemp",
    "mech_1": "mech",
    "mech_2": "mech",
    "ecreact_1": "ecreact",
    "ecreact_3": "ecreact",
}
FINGERPRINTS = {
    "synrfp_default": "synrfp-wl-r2-b1024-parity-delta",
    "drfp": "drfp",
    "rdkit_struct": "rdkit_struct",
    "rdkit_diff": "rdkit_diff",
    "rxnfp_mlm": "rxnfp_mlm",
}


def main():
    rows, missing = [], []
    for dataset, directory in TARGETS.items():
        root = FEATURES / directory
        for name, stem in FINGERPRINTS.items():
            path = root / f"{stem}.npz"
            if not path.exists():
                missing.append((dataset, name, str(path)))
                continue
            with np.load(path, allow_pickle=False) as z:
                X, ok = np.asarray(z["X"]), np.asarray(z["ok"], dtype=bool)
                # Packed bits are represented as uint8 bytes; their nonzero
                # byte fraction is still a useful cache-level sparsity check.
                row_nonzero = np.any(X != 0, axis=tuple(range(1, X.ndim))) if X.ndim > 1 else X != 0
                rows.append({
                    "dataset": dataset, "fingerprint": name,
                    "n_rows": int(len(ok)), "n_features": int(np.prod(X.shape[1:])) if X.ndim > 1 else 1,
                    "n_ok": int(ok.sum()), "n_failed": int((~ok).sum()),
                    "failure_rate": float((~ok).mean()),
                    "zero_rows": int((~row_nonzero).sum()),
                    "zero_rate": float((~row_nonzero).mean()),
                    "mean_nonzero_units": float(np.count_nonzero(X, axis=tuple(range(1, X.ndim))).mean())
                    if X.ndim > 1 else float(np.count_nonzero(X).mean()),
                    "dtype": str(X.dtype),
                })
    if not rows:
        # Feature matrices live in the rebuildable local ``work/`` cache and
        # are intentionally removed from a paper/results checkout.  Reuse a
        # previously generated validity table when it is preserved alongside
        # the other analysis artifacts.
        preserved = OUT / "feature_validity_sparsity.csv"
        if preserved.exists() and preserved.stat().st_size > 0:
            print(f"No cached features found; preserved audit retained at {preserved}")
            return
        raise SystemExit("No cached features found and no preserved audit table exists")
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values(["dataset", "fingerprint"])
    out.to_csv(OUT / "feature_validity_sparsity.csv", index=False)
    pd.DataFrame(missing, columns=["dataset", "fingerprint", "path"]).to_csv(
        OUT / "feature_validity_missing.csv", index=False)
    # A compact quality-control panel; rates are shown on a log scale because
    # most representations have zero or near-zero invalid/empty rows.
    display = out.pivot(index="dataset", columns="fingerprint", values="failure_rate").fillna(0)
    display = display.reindex(columns=FINGERPRINTS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals = display.to_numpy()
    vals = np.maximum(vals, 1e-6)
    im = ax.imshow(vals, aspect="auto", cmap="magma", vmin=1e-6)
    ax.set_xticks(range(len(display.columns)), display.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(display.index)), display.index)
    ax.set_title("Cached feature encoding failure rate")
    ax.set_xlabel("Label-free representation")
    ax.set_ylabel("Dataset")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Failure rate (log-scaled color; zero shown at 10⁻⁶)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"feature_validity_failure_rates.{ext}", dpi=180)
    plt.close(fig)
    print(f"Audited {len(out)} cached matrices; missing {len(missing)}")
    print(f"Wrote {OUT / 'feature_validity_sparsity.csv'}")


if __name__ == "__main__":
    main()
