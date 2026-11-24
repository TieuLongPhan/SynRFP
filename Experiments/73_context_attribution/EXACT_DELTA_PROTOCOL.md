# E3. Predicting yield differences at the same signed delta

**Question.** Can pooled substrate and product context predict the yield
difference between two held-out Buchwald–Hartwig precursor forms with the
same full signed WL counts and the same recorded recipe?

E1 found a gain over a second hash of delta in grouped yield regression.
Its folds put every same-delta, same-recipe precursor pair in separate
test folds. Comparing the two predictions would therefore also compare
different fitted models. This follow-up holds both precursor forms out
together and predicts their yields with the same model.

## Design

Use the pinned classic Buchwald–Hartwig input from E1. With recipe
components outside the molecular graphs, the 15 precursor forms have
nine distinct signed WL count vectors. Three vectors each have three
forms with different shared counts. The Suzuki–Miyaura input has no
such pairs and does not enter this experiment.

For each of the three Buchwald–Hartwig classes, hold out each of the
three unordered precursor-form pairs in turn. The remaining form stays
in the fitting rows, so the signed count vector occurs in training.
One form from another class supplies validation data for all three
holdouts of that class. The runner selects it from sorted eligible
forms with NumPy's default RNG and the seed 1337 plus the zero-based
class index. Class order follows the sorted signed count vectors.
Grouping, validation and recipe matching use inputs alone.

Use the same radius-2 WL parity fingerprints and scikit-learn MLP as in
E1, including target scaling on fitting rows and the lowest-validation-MSE
checkpoint. For every arm, build the condition-component indicator
columns from fitting rows and append them outside the reaction
fingerprint. Run three 2,048-bit arms on each split.

| Arm | Fingerprint |
| --- | --- |
| Pooled | 1,024 signed-delta bits and 1,024 pooled-count support bits |
| Rehashed delta | The same first 1,024 delta bits and a seed-2 hash of delta |
| Ordinary delta | 2,048 signed-delta bits with seed 1 |

For each holdout, score recipes assayed for both forms using their exact
canonical component multiset. Order the two forms by canonical precursor
identifier. Subtract the second yield from the first for both observed
and predicted yields. The primary outcome is the reduction in squared
yield-gap error for pooled context relative to rehashed delta, in squared
yield percentage points. Because both control encodings and the recipe
indicators are identical within a matched pair, either delta control
should predict a zero gap. The runner verifies that equality and counts
any pooled-fingerprint collisions without dropping them.

Average gap losses over matched recipes within each precursor pair, over
the three pairs within each class, and equally over the three classes.
A positive reduction favours pooled context. Report all three class
effects alongside the mean. Row-level RMSE and R² for each fit are
secondary. The 27 fits cover nine holdouts and 2,368 matched gaps.
Each form appears in two holdouts, and training sets overlap. Treat the
three classes and the aggregate as descriptive results within one
screen. Independent campaign transfer requires a separate evaluation.

## Workstation run

Fetch the branch into a separate worktree and install the same
environment as E1.

~~~bash
git fetch origin exp/exact-delta-e3
git worktree add --detach ../SynRFP-exact-delta origin/exp/exact-delta-e3
cd ../SynRFP-exact-delta
python -m pip install -e . -r Experiments/73_context_attribution/requirements.txt
~~~

Point the script at the directory containing the exact
`bh_classic.csv.gz` used by E1. The previous E1 feature, split and
prediction files are unnecessary. If the input is missing, reconstruct
it with `prepare_inputs.py` as described in the [E1 guide](README.md).

~~~bash
data_dir=/path/to/classic_yield
run_dir="Experiments/work/context_attribution/E3-$(date -u +%Y%m%dT%H%M%SZ)"
python Experiments/73_context_attribution/run_exact_delta_pairs.py \
  --data-dir "$data_dir" --check-only
python Experiments/73_context_attribution/run_exact_delta_pairs.py \
  --data-dir "$data_dir" --output-dir "$run_dir" --jobs 8
~~~

The preflight must report nine joint holdouts, 2,368 matched gaps and
27 planned fits. Use a new output directory for each execution.

## Results to return

When `status.json` reports `complete` and 27 fits, return it together
with `manifest.json`, `fit_scores.csv`, `pair_gap_scores.csv` and
`class_gap_scores.csv`. The pooled `macro` row gives the primary
reduction against the zero-gap control. The pair scores show recipe
counts, fingerprint collisions and observed zero-gap loss. Keep
`splits.csv.gz` and `predictions.csv.gz` on the workstation. Their
digests are recorded separately in `status.json`.
