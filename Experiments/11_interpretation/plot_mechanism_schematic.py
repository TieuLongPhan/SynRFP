"""Render a compact conceptual schematic of SynRFP's signed difference."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "synrfp_mechanism_schematic.png"


def box(ax, x, y, text, color, width=1.35, height=0.62):
    p = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03",
                       facecolor=color, edgecolor="#333333", linewidth=0.8)
    ax.add_patch(p)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=9)


def panel(ax, title, product, reactant, delta, note):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(0, 4.16, title, fontsize=12, weight="bold", va="top")
    ax.text(0, 3.48, "product token counts", fontsize=9, color="#555555")
    for i, (t, c) in enumerate(product):
        box(ax, 0.05 + i * 1.55, 2.72, t, c)
    ax.text(0, 2.40, "− reactant token counts", fontsize=9, color="#555555")
    for i, (t, c) in enumerate(reactant):
        box(ax, 0.05 + i * 1.55, 1.64, t, c)
    ax.add_patch(FancyArrowPatch((4.9, 1.48), (4.9, 1.03), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.0))
    ax.text(0, 0.80, "signed delta", fontsize=9, color="#555555")
    for i, (t, c) in enumerate(delta):
        box(ax, 0.05 + i * 1.55, 0.08, t, c)
    ax.text(6.65, 1.15, note, fontsize=9, color="#444444",
            va="center", ha="left", wrap=True)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), constrained_layout=True)
panel(
    axes[0], "A  Cancellation of unchanged context",
    [("spectator", "#d9d9d9"), ("C=O", "#f4b183"), ("C–N", "#f4b183")],
    [("spectator", "#d9d9d9"), ("C–C", "#b4c7e7"), ("C–N", "#f4b183")],
    [("0", "#eeeeee"), ("+ C=O", "#f4b183"), ("− C–C", "#b4c7e7")],
    "Shared spectator contribution\ncancels before hashing.",
)
panel(
    axes[1], "B  Direction is retained",
    [("C=O", "#f4b183"), ("C–N", "#f4b183")],
    [("C–C", "#b4c7e7"), ("C–N", "#f4b183")],
    [("+ C=O", "#f4b183"), ("− C–C", "#b4c7e7")],
    "Reversal swaps product/reactant\nroles and flips the signs.",
)
fig.suptitle("SynRFP encodes product − reactant graph-token counts", fontsize=15, weight="bold")
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
