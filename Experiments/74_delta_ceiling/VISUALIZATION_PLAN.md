# Visualization plan for request 74 (Supplementary Information)

Written 25 November 2026 by the paper agent. This plan says what each figure must
show, why, and which columns of the experiment outputs it needs, so the workstation
agent can export exactly those and the paper agent can draw them without further
questions. See `REQUEST.md` for the experiment itself.

## Question the figures answer

Why does moving half of the delta bits to pooled context help some classification
endpoints (ECReact, TPL B/UB, Schneider B) and hurt others (Mech, SynTemp,
USPTO50K)? Working explanation: when the label is a function of the signed counts,
delta alone is enough and halving it only costs resolution. When different labels
share the same signed counts, only context can separate them.

The paper already contains the per-endpoint effect (mean MCC change, D+U minus D).
The new figures add direct evidence for the explanation, using reactions.

## SI-A: ceilings against effect (one figure, two panels)

- Panel a, dumbbell per endpoint (14 rows, ordered by the MCC effect): delta-only
  ceiling (grey dot) joined to the (D,U) ceiling (purple dot). A long line means
  pooled context could add a lot, a short line means it cannot.
- Panel b, scatter: x = (D,U) ceiling minus D ceiling, y = mean MCC change from the
  paper. One labelled point per endpoint, Spearman rho in the corner. If the
  explanation holds, the points rise from left to right.
- Needs: `endpoint_summary.csv` with `endpoint, tokenizer, channels, n_reactions,
  n_groups, collision_share, ceiling` for channels D, U, D+U (D,U together as one
  key) and D+S. The paper agent joins the MCC effects itself.

## SI-B: reactions that delta cannot separate (ECReact)

- A grid of reaction pairs: rows are EC level 1, 2 and 3, two or three pairs per
  row. Each pair shows reaction A and reaction B as structures, with their EC
  labels beneath, and the caption states that both have identical signed counts.
- Choose small reactions and pairs with different substrates so the difference is
  chemically readable. Prefer pairs whose labels differ at the shallowest level
  (a level-1 mismatch is the strongest case).
- Needs: `examples.csv` with `endpoint, pair_id, side (A or B), r_id, rxn, ec1, ec2,
  ec3, group_id, group_size, n_labels_in_group, n_heavy_atoms`. At least 8 pairs per
  EC level, sorted by `n_heavy_atoms`.

## SI-C: the contrast, reactions that delta separates perfectly

- Same layout for the endpoints that lose with context: one group per endpoint
  (Mech L1, SynTemp R2, USPTO50K B) showing three reactions that share signed counts
  and the same label, with the label beneath. This is the pure group that makes
  context unnecessary.
- Needs: `pure_examples.csv` with `endpoint, group_id, r_id, rxn, label, group_size`
  for groups of size >= 3 and one label, at least 10 groups per endpoint, small
  reactions first.

## SI-D: how mixed are the groups (one panel, can join SI-A)

- Per endpoint, a violin or box of the label purity of reaction groups, weighted by
  group size (purity = majority share inside the group). Pure endpoints sit at 1,
  mixed endpoints spread below.
- Needs: `group_table.csv.gz` with `endpoint, tokenizer, channels, group_id, size,
  n_labels, majority_share` (no SMILES).

## Optional, main text

If SI-A panel b shows a clear rise, one small scatter panel could be added to the
Discussion or to Fig. 2 as an extra column. Decide after seeing the data.

## Extra columns to export beyond REQUEST.md

- `endpoint_effects.csv`: copy of the per-endpoint D+U minus D MCC change used in
  the paper, so the join can be checked.
- For SynTemp R0, R1, R2: the ceilings separately, because the paper finds that the
  loss grows with radius and the ceiling should explain or refute that.

## What would change the plan

- If the ceilings are all near 1 for every endpoint, the explanation is wrong, and
  the figures become a null result. Say so plainly and still export SI-A.
- If the ceiling depends strongly on the tokenizer, export all four tokenizers and
  draw SI-A for the WL radius 2 standard preset only, with the others in a table.

The paper agent will write `code/delta_ceiling_figures.py` to read the exported
files and draw SI-A to SI-D in the same style as the existing SI figures.
