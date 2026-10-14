# Full-data SynRFP construction factorial

This study supersedes the bounded two-panel pilot as the basis for assessing a
shared or task-specific SynRFP configuration. The user explicitly requested
full dataset rows and removed XGBoost from the new work. Historical weights,
inputs, plans and results remain local and preserved.

## Crossing and coverage

Every endpoint receives all72 configurations: WL/Morgan/Path × radius2/3 ×
Parity/MinHash/controlled-CW × Delta/Union/Delta+Union/Roles. Total width is2048
bits. Composite modes contain two1024-bit channels. Fingerprint seed is1 and
MinHash/CW b-bit width is2. CW uses unit support after full multiset arithmetic;
historical native count-weighted CW is a separate condition. Stereo features
are absent and remain scoped to future barrier work.

| Task | Full populations and evaluation | Cells |
|---|---|---:|
| Reaction classification | All14 original endpoints; first original5 grouped outer folds | 5,040 |
| EHReact activity | All rows of3 eligible original assays; first original5-fold group-CV repeat | 1,080 |
| Yield | BH3955, Suzuki5760, NCS22609; all64 original saved partitions | 4,608 |
| Clustering | All12 original endpoints;5 initialization seeds | 4,320 |
| Total |72 configurations; no new baseline fits |15,048 |

Classification endpoints are Schneider UB/B, TPL UB/B, USPTO50K UB/B, SynTemp
R0/R1/R2, ECReact L1/L2/L3, and Mech L1/L2. Clustering uses the12 endpoints in
the main canonical clustering inventory, excluding SynTemp R1/R2. EHReact is a
separate binary case study and is not pooled into the multiclass average.

Full row count means the exact original benchmark endpoint population, not a
5000- or50000-row cap. TPL endpoints each retain445115 rows; ECReact endpoints
each retain185734. Class/clustering encoding refusals remain zero-vector rows,
with failure coverage recorded. Yield preserves the original seven-fingerprint
shared-valid population: NCS22609 of27497 raw records, including the previously
disclosed exclusions. No new yield exclusions or target clipping occur.

Classification has one5-fold CV, not five repetitions of5-fold CV. Yield uses
BH random10, BH published OOD4, Suzuki random10 and NCS cluster-OOD40 saved
partitions. Those regimes are reported separately. Existing filtering,
balancing and target definitions are inherited, and repeated scores are
correlated.

## Fixed learning and clustering protocols

Classification preserves the original sklearn initialization, random-state
arrays, Adam arithmetic, accuracy stopping rule and best-weight restoration.
It uses one512-unit hidden layer, alpha0.0001, batch256 and at most200 epochs.
Initial model seed follows the original fold index; EHReact uses the original
repeat0 initialization. Bit features are not standardized.

Yield preserves the fixed512-unit ReLU MLP, Adam learning rate0.001,
weight decay0.0001, batch256, at most200 epochs and validation-MSE patience15.
Target statistics use actual fit-training rows only. Responses remain
unclipped, including observed yields above100. Model seed is42.

All new neural fits use matched FP32 CUDA arithmetic, with TF32 disabled. They
form a separate backend series from the pilot CPU scores. No hyperparameter or
representation winner is selected from outer-test results by the controllers.

Clustering uses MiniBatchKMeans consistently across all72 configurations,
with batch8192, n_init3, max_iter100, max_no_improvement10, tol0 and
reassignment_ratio0.01. Its label-blind cluster count is
`min(512,max(2,round(sqrt(n/2))))`, derived solely from full population size.
Labels score AMI/ARI/NMI afterward. Silhouette uses2000 deterministic
identity-selected rows, and retained assignments permit five-seed stability
analysis. This fixed-method factorial remains separate from the main
known-K/KMeans/other-method benchmark arms.

## Validation, provenance and storage

The source and input plan is sealed at
`~/.local/share/synrfp-runs/all_endpoint_factorial_full_v1/plan.json`.
It records original row identities, source positions, split hashes, source
files, parameters and72 exact configuration specifications. Encoders never
receive targets as tokenizer input. GPU workers receive only training and
validation outcomes, plus held-out feature selectors.

Independent fixtures verify exact original initialization/RNG state,
CPU-equivalent shared-input classification predictions/weights/history,
CPU-equivalent yield predictions/history, rejection of corrupted feature pools,
and rejection of held-out outcome injection. A fresh CUDA classification fixture
and an actual full BH yield cell passed the version2 runtime gate.

The first launch exposed a CUDA guard's local-import ordering issue and an
ambiguous clustering-module import. New runtime adapters correct these
boundaries; frozen originals and interrupted attempts remain preserved.
`runtime_execution_v2.json`, `runtime_launch_v2.json` and
`runtime_cuda_READY_v2.json` bind the repaired adapters and successful gate.
Training mathematics, input populations and the sealed plan are unchanged.

Shared packed source caches avoid copying dense training matrices per model.
Compact selectors, initialization states, preprocessing, best weights and
predictions retain replayable training inputs. Models stay outside Git. Dispatch
stops below a35GiB free-space reserve; encoding stops below40GiB.

## Resources, priority and analysis

Both GPUs are allocated to this study after the current GNAN reference fit
completed safely. Additional YieldBERT/GNAN/SynCat reference dispatch is held
until the factor-analysis priority is resolved. XGBoost has no new jobs, and
DRFP is reserved for main benchmarking rather than this factorial.

GPU host controllers use cores24–25/28–29, quota150% each, and16GiB each.
Clustering uses cores0–3/4–7, quota400% and10GiB each. Encoding starts on8–11
with four workers and24GiB; the versioned resource extension expands to8–15
after a completed-source boundary. Heavy-work allocation is approximately19
physical-core equivalents and76GiB in active-fit caps, plus small report
processes, within the authorized24-core/94GiB budget.

The report process publishes task-specific means, descriptive SDs, paired main
effects and all interaction orders, cross-panel consistency and clustering
seed stability. It verifies retained hashes, predictions/outcomes and clustering
center replay before marking completion. Primary classification, EHReact,
yield regimes and clustering stay separate. Test-score ranks are diagnostic;
untouched evaluation is needed for a confirmatory selected-configuration claim.

ETA must be revised from the full-data timings. The old6–10-hour estimate was
for the completed bounded pilot, not this15048-cell full-data study.
