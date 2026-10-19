# Complete SynRFP construction factorial across seven tasks

This new version completes the existing construction analysis for barrier,
feasibility, reagent-set and condition-role prediction. Classification, clustering
and yield already have the full study 22 crossing and remain historical,
separately sealed results. EHReact stays deferred from the seven-task main story.
No new task, new joint-role implementation, XGBoost model or stereo feature is
introduced. Model weights and all historical attempts remain local.

## Construction and identity

The grid is exactly study 20/study 22: WL, Morgan and Path tokenizers; radius 2 or 3;
ParityFold, b-bit MinHash or controlled CW; delta, union, delta_union or original
roles. There are 3 × 2 × 3 × 4 = 72 constructions. Every vector has 2,048 total bits,
fingerprint seed 1 and bbit 2. Delta_union has 1,024 signed-delta bits followed by 1,024
union bits. Original roles has 1,024 signed-delta bits followed by 1,024 shared bits.
Shared means the token multiset intersection min(R,P), not mapped-atom retention
or a condition-role label. The retired joint sketch from study 27 is excluded.

Complete occurrence counts determine D=P−R, U=P+R and S=min(R,P) before sketching.
The controlled CW comparison uses unit signed/unsigned support after this
algebra. Native count-weighted CW is a different method and is not silently
substituted. Native token labels remain element, aromaticity, charge and hcount
on atoms and order on edges, with native degree contributions unchanged.

All new sources use the frozen strict explicit-side study 20 `yield` parser
profile. Each source already contains the native study's mapped-handle removal,
hydrogen policy and endpoint normalization. No further stereo erasure, aromatic
rewriting or target-conditioned input transformation is applied. Encoding reads
only declared reaction/left/right/eligibility fields, never target values,
condition labels, corpus methods, expert annotations or split/group fields.

## Frozen populations and repeated evaluations

| Task | Data and exact existing evaluations | Logical cells |
|---|---|---:|
| Feasibility | FREA: 200,000 train, 10,000 validation, 10,000 calibration and 751 expert-test rows; optimizer seeds 0–4 on that fixed external test | 360 |
| Reagent sets | USPTO-500: 143,535 source rows, 143,342 frozen eligible rows; 5 native optimizer repeats and 5 grouped folds, one optimizer seed per grouped fold | 720 |
| Condition roles | USPTO-Condition: 680,741 source rows, 660,216 eligible rows; 5 fixed grouped folds × 5 optimizer seeds plus 5 native repeats | 2,160 |
| Barrier, primary | GDB7: 11,926 and Cyclo: 5,269 source rows; 5 original reactant-group holdouts per dataset at full training fraction | 720 |
| Barrier, expanded | Seven existing activation datasets, 5 existing group-CV folds each: SNAr 503; E2 1,264; SN2 2,361; E2SN2 3,625; B97XD3 16,365; RDB7 23,852; RGD1 353,984 | 2,520 |

The full activation scope gives 6,480 logical cells. Primary-only barrier scope
gives 3,960. Exactly 270 completed WL/parity delta, delta_union and original-roles
cells are eligible for reuse: 30 FREA, 60 reagent and 180 condition. Reuse requires
full original-population feature equality, the same native population, selectors,
parameters, training seed and recipe, plus retained-result/artifact verification.
Thus the full scope requires 6,210 fresh fits; primary scope 3,690. All barrier
cells are fresh under their native CUDA regression recipe; historical CPU probes
remain feature anchors and separately reported scientific evidence.

The frozen populations never shrink to make the 72 constructions look better.
Additional encoding failures produce zero vectors with failure flags, and all
original selected rows remain in the main comparison. Original shared-valid
selection from studies 14, 15, 23 and 29 is frozen, not recomputed using new scores or
new fingerprints. FREA's primary evaluation already preserves every row.

The original study 14 barrier partitions are five distinct grouped holdouts,
not five-fold CV. They remain holdouts. Study 15 retains its five group-CV folds
with reverse-direction reaction links. The original reagent groups preserve
isomeric reactant identity, whereas condition groups erase stereo; each native
contract remains unchanged. Native author splits and grouped generalization are
separate reporting groups.

## Models and selection

Reagent and condition use exactly study 23's multilabel MLP: one 512-unit ReLU
hidden layer; Adam learning rate 0.001, weight decay 0.0001, batch 256, max 150 epochs,
patience 15, min_delta 0.0001.
All features use actual-fit-training-only affine moments. Component vocabularies
use training targets only; unknown held-out components remain false negatives.
Epoch selection uses validation BCE. The scalar threshold is selected from
0.05, 0.10, …, 0.95 by validation exact-set accuracy, then micro-F1, then distance to 0.5.

FREA uses exactly study 24: 512-unit ReLU, Adam learning rate 0.001, weight decay 0.0001,
batch 512, max 200 epochs, patience 15, min_delta 0.0001. All factorial vectors are native binary features and
remain unscaled. Epoch selection uses validation BCE; the separate corpus
calibration partition fixes the operating threshold before expert outcomes are
read. Expert annotations and labels never enter the fitting worker.

Barrier uses the corresponding native study 13/study 15 CUDA MLP: 512-unit ReLU,
Adam learning rate 0.001, weight decay 0.0001, batch 256, max 200 epochs, patience 15,
min_delta 0.0001, model seed 42.
Binary inputs remain unscaled; the regression target is standardized on actual
training rows only. Minimum validation MSE selects weights. No learning-curve
fractions, new covariates, Chemprop reruns or stereo tuning are part of this grid.

Model and encoder sources, original plans, tables, selectors, population masks,
native cache anchors and reused result receipts are hash-pinned. Meaningful
label-blind encoder/selector proofs run in the existing `synrfp` environment;
native model equivalence, positive learning, held-out-label rejection and retained
weight replay run with CUDA hidden in `synprop`. No package upgrade is required.
Scientific sealing happens after those gates and before production fitting.

## Reporting and a common representative

Report per-endpoint metrics, uncertainty and paired construction effects first.
Within each condition fold, average over its five optimizer seeds; then calculate
between-fold variation across the five fold means. Report within-fold optimizer
variation separately. Native/FREA five-seed variation concerns one fixed test set.
Do not treat 25 optimizer/fold observations as 25 independent test sets.

Use MCC for classification, AMI for clustering, R² for yield, RMSE for barriers,
AUROC for feasibility and micro-F1 for reagent/condition prediction. Different
units and metrics are never directly averaged. A cross-task diagnostic can
average ranks or normalized within-endpoint regret:aggregate related endpoints
inside source families before equal weighting of the seven tasks. Report
stability, encoding cost and coverage alongside that diagnostic.

GDB7 and RDB7 have related provenance; E2SN2 combines E2 and SN2. Expanded
study 15 Cycloadd G_act is not counted again because study 14 Cyclo is already
represented. These dependencies must remain explicit in all macro summaries.
Reaction enthalpy, rate and conversion targets are supporting property extensions,
not extra activation-barrier datasets or an eighth main task.

This is exploratory construction analysis after earlier test results were
viewed. The analysis can explain tradeoffs and support the user's selection of
one common preset. It cannot turn a test-selected maximum into an unbiased
confirmatory result. Do not select a global default automatically. Separately
freeze the chosen preset and a confirmation protocol before promoting a common
benchmark claim. Validation scores, stability and cost can inform that choice,
but do not remove the historical score exposure.

## Artifacts and resource boundary

Store shared packed 2,048-bit feature pools and shared target/vocabulary/selector
payloads per exact partition, not dense matrices copied per model. Preserve
full-precision inference decisions using compact thresholded bitsets and stable
top-five indices for multilabel tests, validation-grid sufficient statistics,
and a bounded replay sample. No float16 probability rounding is used. Retained
weights and preprocessing reconstruct full probabilities when needed.

The uncompressed 72-feature upper bound is 25.2 GiB for the full raw
inventory; completed native feature aliases and NPZ compression reduce new
storage. Regression/FREA weights add about 14–15 GiB; multilabel weights and compact
test outputs add about 20–25 GiB, depending on vocabulary. Shared target payloads
prevent several additional gigabytes of duplicate labels. A 40 GiB free-space
reserve is enforced before stages and fits. Only newly created, verified,
reconstructible temporary matrices may be removed; model weights, failed
attempts and historical features must never be deleted to satisfy the guard.

The root controller admits CPU stages and leased GPU workers under the combined
75% CPU/RAM limit. Existing global GPU-lock files retain their original inode.
Priority learned-reference fits finish at their native fit boundaries before
devices are handed to this queue, and resume afterwards. No historical scientific
plan or scheduler source is rewritten. Final closure requires every fresh and
reused canonical cell audit, exact expected counts and verified output hashes.
