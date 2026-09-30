#!/usr/bin/env python
"""Render a manuscript-ready reaction-centre localization figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "08_reaction_center" / "results" / "per_reaction.csv"
OUT = HERE / "results"


def main() -> None:
    d = pd.read_csv(SRC)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.35]})
    bins = 25
    ax0.hist(d["delta_occurrence_center_fraction"], bins=bins, alpha=.75,
             label="SynRFP signed delta", color="#0072B2")
    ax0.hist(d["union_center_fraction"], bins=bins, alpha=.65,
             label="Union baseline", color="#D55E00")
    ax0.set(xlabel="Fraction of token occurrences near mapped centre",
            ylabel="Reactions", xlim=(0, 1))
    ax0.legend(frameon=False, fontsize=8)
    ax0.set_title("Reaction-level localization")

    means = (d.groupby("level2", as_index=False)
               .agg(localized=("delta_occurrence_center_fraction", "mean"),
                    union_localized=("union_center_fraction", "mean"),
                    center_coverage=("center_coverage", "mean"),
                    n=("level2", "size"))
               .sort_values("localized"))
    means.to_csv(OUT / "reaction_center_class_summary.csv", index=False)
    shown = means.tail(min(15, len(means)))
    ax1.barh(shown["level2"], shown["localized"], color="#009E73")
    ax1.set(xlabel="Mean localized occurrence fraction", xlim=(0, 1))
    ax1.set_title("Highest-localization mechanistic classes")
    ax1.tick_params(axis="y", labelsize=7)
    fig.suptitle("SynRFP changed-token localization (Mech31k; radius 2)", y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "reaction_center_localization.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(OUT / "reaction_center_localization.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'reaction_center_localization.png'}")


if __name__ == "__main__":
    main()
