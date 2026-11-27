# Request 93 yield learner comparison

The study fits 621 random-forest regressors on the 69 released
split instances, including five publication- and duplicate-grouped USPTO folds.
The seven benchmark representations reuse the original saved standardized train
matrices and test matrices. Both context variants use the same held-out selectors.
The USPTO D and D+U caches were built in the existing literal-input profile since
the full-row variants were unavailable. All 143,535 archived D+S vectors were
replayed to verify the profile. The old USPTO input refusals use the same three
row-specific literal graph adapters.

The forest uses 500 trees, square-root feature sampling, minimum leaf size one,
and seed 42. There was no test-set tuning. XGBoost was excluded per the project
decision. The seven-representation ranking uses mean R² and Kendall's tau against
the released MLP ranks. The matched MLP construction contrasts are published
where the archived D and D+U selectors match this cohort. Their comparison scope
is explicit in the context table.

The study fits 320 success classifiers on the exact expanded
Buchwald–Hartwig fit-train and validation selectors. The forest classifier uses
balanced class weights. The CPU MLP retains the 512-unit ReLU architecture and
original optimizer and checkpoint-selection settings, with sigmoid output and
binary cross-entropy on the author labels. On this source, 183
released labels differ from applying yield >10 to the continuous value.
ROC AUC and balanced accuracy use the released binary labels. A 0.5 cutoff
applies to probabilities, and yield-mean controls use yield >10.

The condition ranking uses the same 609 pair-by-holdout instances, recipe
replicate averaging and stable recipe tie order as Request 92. A top-ranked recipe
succeeds when its observed mean yield exceeds 10%. Per-recipe baselines use
fit-train recipes and use the respective fit-train global mean for unseen recipes.
The classifier table includes pooled test occurrences and split mean and SD.
Repeated test occurrences do not represent independent observations.

The run used 20 CPU workers on the AMD Ryzen Threadripper 7970X 32-Cores for
3.12 fit wall-clock hours. Model files are
retained locally. Package versions, row bindings and code hashes are in the
completion manifest.
