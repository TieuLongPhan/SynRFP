"""Render fixed-rule cluster examples for the interpretation package."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import AllChem, Draw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def choose_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fp in sorted(frame["fingerprint"].unique()):
        sub = frame[frame["fingerprint"] == fp].sort_values(
            ["cluster_purity", "cluster_size"], ascending=[False, False]
        )
        rows.append(sub.iloc[0])
        # Keep one low-purity SynRFP example as an explicit counterexample.
        if fp == "synrfp-wl-r2-b1024-parity-delta":
            rows.append(sub.iloc[-1])
    return pd.DataFrame(rows)


def render(path: Path, output: Path) -> None:
    frame = pd.read_csv(path)
    chosen = choose_rows(frame)
    images = []
    labels = []
    for _, row in chosen.iterrows():
        reaction = AllChem.ReactionFromSmarts(row["reaction"], useSmiles=True)
        image = Draw.ReactionToImage(reaction, subImgSize=(900, 230),
                                     useSVG=False)
        image = image.convert("RGB")
        labels.append(
            f"{row['fingerprint']} | {row['r_id']} | "
            f"cluster n={int(row['cluster_size'])}, "
            f"purity={row['cluster_purity']:.3f}"
        )
        images.append(image)

    width = max(im.width for im in images)
    row_h = 270
    canvas = Image.new("RGB", (width, row_h * len(images)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        y = i * row_h
        canvas.paste(im, (0, y + 30))
        draw.text((8, y + 5), label, fill="black")
    canvas.save(output, dpi=(180, 180))


if __name__ == "__main__":
    render(OUT / "cluster_examples_mech2.csv", OUT / "cluster_examples_mech2_panel.png")
    render(OUT / "cluster_examples_schneider_b.csv", OUT / "cluster_examples_schneider_b_panel.png")
