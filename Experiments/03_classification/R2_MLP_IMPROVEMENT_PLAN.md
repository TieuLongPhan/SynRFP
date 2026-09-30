**Plan: improve SynTemp r2 with a fixed SynRFP fingerprint**

Prepared 2026-09-21. This is a proposed experiment plan, informed by a read-only audit of saved results, training code, cached splits, labels and features. No new training was launched while preparing it.

The recommended order is: correct the evaluation, establish a sufficiently trained MLP, test an output head for the large aggregate class, then test residual layers and carefully controlled auxiliary supervision. Beating DRFP is a testable objective, not a guaranteed outcome. The current experiments do not establish the best achievable performance of this fingerprint.

**Objective and scope**

The primary target is higher r2 MCC than DRFP under the same evaluation and model-selection budget. Keep `synrfp-wl-r2-b1024-parity-delta` and its feature files fixed: WL tokenizer, radius 2, 1,024 bits, parity sketch, delta mode, hash seed 1, and the existing atom/bond attributes and agent handling. Change only the MLP architecture, output heads, loss and training configuration. No fingerprint concatenation, additional reaction inputs, changed label definitions or larger fingerprint.

Use r2 MCC as the selection metric. Record macro-F1, balanced accuracy, accuracy, per-class recall and training cost as secondary outcomes. Improving a mean over r0/r1/r2 is not sufficient if r2 gets worse. Auxiliary labels may supervise training heads, but inference must use only the fixed fingerprint.

There are two different comparisons worth reporting: tuned SynRFP versus the existing DRFP baseline, and tuned SynRFP versus equally tuned DRFP. Only the second supports a conclusion about which representation performs better under this MLP search. Preserve the existing common-configuration benchmark alongside the tuning study.

**What the results currently establish**

| r2 model | Mean MCC over 25 existing outer folds | Interpretation |
|---|---:|---|
| Original SynRFP MLP | 0.937371 | Common benchmark configuration |
| Wider SynRFP pilot | 0.938404 | Exploratory tuning result |
| Original DRFP MLP | 0.944750 | Common benchmark configuration |

The observed wider-pilot gain is 0.001033 MCC; its remaining gap to DRFP is 0.006346. It beats original SynRFP on 17/25 folds and DRFP on 4/25. Its gains over SynRFP by repeat are +0.002187, +0.002276, +0.000822, -0.000149 and +0.000028. These are descriptive numbers, not evidence of a reliable improvement from width alone.

SynRFP currently leads DRFP in the original MLP benchmark on 11/14 datasets. The remaining deficits are r2, TPL balanced (0.935065 versus 0.945439), and TPL unbalanced (0.935603 versus 0.957135). Winning r2 would not by itself achieve the broader goal of winning every dataset.

The separate 396-fit benchmark resume completed with zero failures in about 7.6 minutes. It includes Mech31k, the SynTemp seed-42 fixed splits, and the dedicated TPL RXNFP results. Those outputs remain part of the original benchmark, not this tuning study.

**Corrections to earlier interpretations**

1. The original three-head hierarchy pilot and the separate single-task models did not share the same partitions. The hierarchy pilot used r0's split; the r2 model used r2's split. Assignments differ on 18,903 rows, and their validation sets overlap on only 2,536 rows. The earlier mean-MCC comparison cannot establish that multi-task learning is worse.
2. `cv_s0` through `cv_s4` are five rotating folds of one repeat, not five independent random splits. Together their validation sets cover every one of the 43,441 rows. Selecting one configuration on those validation results and evaluating it on all 25 test folds reuses test-row information across folds. Calling the tuned test results untouched was incorrect. Ordinary CV reuse for a fixed model is fine; globally choosing a configuration on these rotating validation sets is the problem.
3. The original GPU MLP uses sklearn-style Adam, batch-normalized L2, batch size 256, accuracy-based stopping and a maximum of 200 epochs. The tuning script uses AdamW, batch size 1,024, MCC-based stopping and a maximum of 50 epochs. Several things changed at once, so the observed difference is a pipeline difference, not an isolated width/dropout effect.
4. The final pilot refit randomly split its inner stopping set. Outer test scaffolds remain separate, but inner training and validation share scaffolds. A grouped inner split better matches the intended scaffold-generalization task.
5. Deep, focal-loss and class-balanced candidates were tested on only one selection fold. They were not all rejected across multiple validation splits. The five-fold r1-auxiliary comparison also used a different learning rate from its single-task comparator and needs a matched zero-auxiliary control.

Retain all existing results as exploratory evidence. A new random seed does not make previously inspected reactions new independent evidence. A nested procedure can prevent further automatic selection leakage; truly independent confirmation would require unused external reactions and compatible labels.

**Why another targeted MLP study is justified**

Training may be incomplete. All five baseline and all five wider-head selection runs hit 50 epochs. Their best epochs were 41–50, and the wider head still gained about 0.004 validation MCC on average after epoch 25. For equal training-set size, the maximum update budget at 50 epochs/batch 1,024 is approximately one sixteenth of 200 epochs/batch 256, although the original models often stopped early. This is evidence to check convergence, not proof that more epochs will win.

The labels have a special structure. There are 680 r2 classes and 43,441 rows; the median class has 18 records. Source label `679` contains 18,721 rows (43.10%) and spans 352 r1 labels and 141 r0 labels. The other r2 labels have a unique r1 parent in the cached data. The large class behaves like an aggregate label, but the source metadata does not document why it was created; rare-class pooling remains an interpretation. Local preprocessing filters rare classes and does not create this aggregate label. Identify it through the source label map: source `679` is encoded as `y=644` here.

A single-parent hierarchy constraint on every r2 label is therefore invalid. A hard r1 decision could exclude the correct r2 output, and a coarse loss could train the shared MLP for a different goal from exact r2 classification. Hierarchy research also reports tradeoffs between exact classification and the severity of mistakes; it does not guarantee better leaf-level accuracy. [Bertinetto et al., CVPR 2020](https://arxiv.org/abs/1912.09393).

Exact-vector collisions do not establish a tight feature ceiling:

| Diagnostic on existing r2 rows | SynRFP | DRFP |
|---|---:|---:|
| Feature width | 1,024 | 2,048 |
| Valid feature rows | 43,439 | 43,441 |
| Unique feature vectors | 35,570 | 40,785 |
| Identical-vector groups with conflicting labels | 32 | 19 |
| Minimum errors forced by those exact conflicts | 41 | 22 |
| Median active bits | 26 | 32 |

The difference in forced errors is just 19 rows, or 0.0437 percentage points of empirical accuracy. This calculation is not an MCC ceiling, an estimate of generalization, or proof of information sufficiency. It does suggest that exact collisions alone are not a persuasive explanation for the current gap. Two invalid SynRFP rows cannot explain the full gap either. Keep all benchmark rows under the existing failure policy and report a sensitivity analysis if needed.

**Step 1 — make the next comparison valid and diagnosable**

Freeze the data, feature hashes, label maps, candidate list and selection rule before starting the next comparison. Archive the present scores as exploratory. Add a separate study directory, with outputs that cannot overwrite the original benchmark.

Keep Track A and Track B distinct. Track A is the existing 5 repeats × 5 scaffold-grouped outer folds; all 25 cached r2 splits were checked and have no scaffold overlap across train, validation and test. Track B remains the requested random 80:10:10 split with partition seed 42. Model initialization seeds are separate from the partition seed. Track B is a reference evaluation, not a substitute for Track A or a fresh holdout after prior inspection.

For each Track A outer fold, combine only that fold's train and validation rows into an 80% development pool. Create three scaffold-grouped inner folds inside that pool, using the r2 labels. Every single-task, multi-task, SynRFP and DRFP run gets the same row assignments. Never use r0's independently generated split to assess r2. Construct any class weights or label relationships from inner training labels or a separately documented pre-existing ontology.

Use inner validation MCC for configuration and epoch selection. After selecting within that outer fold, derive the refit epoch count from its winning inner runs and refit on the full 80% development pool for that fixed count. Evaluate its 20% outer test once. Keep all outer-test results hidden until the complete search procedure has finished. Do not select one global winner using validation results pooled across different outer test folds. This follows the separation between inner selection and outer evaluation described in the [scikit-learn nested-CV documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Save row IDs, source/encoded label maps, split and feature hashes, initialization seeds, model configuration, learning curves, optimizer-update count, stopping reason, checkpoint, predictions and probabilities. Record the feature-invalid mask. Store predictions from the outset: the original worker discarded them, which prevents useful error analysis without rerunning models. Keep probabilities compressed or save top-k plus the probabilities needed for planned diagnostics.

Reproduce the reference-sized MLP inside the new trainer before comparing architectures. Run both fingerprints with it. Keep a separate reproduction of the old trainer so changing Adam/L2 to AdamW or stopping accuracy to MCC cannot be mistaken for a fingerprint effect.

**Step 2 — test convergence before adding complexity**

Start with ordinary cross-entropy and plain 512- and 1,024-unit MLPs. Compare batch sizes 256 and 1,024 and initial learning rates 0.0003 and 0.001. Start with a 200-epoch maximum and patience 25 on inner validation MCC, best-checkpoint restoration, and learning-rate multipliers of 0.5 after epochs 80 and 140. Use the same epoch-based schedule during full-development refits; it requires no validation set. AdamW weight decay 0.0001 is a starting value, not equivalent to the old sklearn `alpha`. Keep the optimizer and scheduler rules identical across fingerprints.

Log training loss and validation loss/MCC against both epochs and optimizer updates. If a candidate reaches the cap while still improving, mark it budget-limited. Do not interpret it as converged or reject its architecture on that basis. Any extension of the cap must use a predeclared rule applied equally to both fingerprints, based only on inner training/validation curves.

This step answers whether the previous pilot simply used too few updates, whether the larger batch harmed validation performance, and whether a wider head adds benefit once training is comparable. If training MCC remains low, investigate optimization/capacity; if training is high but grouped validation stalls, prioritize generalization rather than a larger network.

**Step 3 — target the aggregate class with a soft MLP output head**

First use inner-validation predictions to split errors into: aggregate class predicted as a specific class; specific class predicted as aggregate; and confusion between two specific classes. Compare these error counts between SynRFP and DRFP. Also stratify by class frequency, unseen-in-training labels and whether the same feature vector appears in training. Do not tune from outer-test error slices.

Test a shared MLP with two outputs: a scalar probability `g(x)` for the aggregate class and a 679-way conditional distribution `q(k | x)` for the other classes. Its final 680-class probabilities are:

```text
p(aggregate | x) = g(x)
p(class k | x)   = (1 - g(x)) * q(k | x)
```

Train with the negative log likelihood of the original r2 label and predict by the largest final probability. This uses the same 680 evaluation labels and the same fingerprint. There is no hard routing threshold and no ground-truth parent supplied at inference. Compare it with an ordinary softmax head using the same trunk, optimizer, rows and training budget.

This is a proposed inductive bias, not extra information or a guaranteed stronger model. It is particularly worth testing if errors into/out of the aggregate class dominate the gap. If the gap lies mostly between specific classes, residual capacity becomes the more relevant experiment.

**Step 4 — test modest residual capacity and controlled auxiliary supervision**

Test a two-layer MLP `[1024, 512]` and a compact two-block residual MLP of width 512, with dropout 0 and 0.1. A residual MLP is a reasonable tabular baseline, but published tabular results are not direct evidence on reaction fingerprints. [Gorishniy et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html).

For hierarchy assistance, use one shared MLP and an r2 head plus a separate r1 head. Optimize `CE_r2 + lambda * CE_r1`, with `lambda` in `{0, 0.02, 0.05, 0.10}`. The lambda-zero control must have identical architecture, initialization procedure, learning rate and batch schedule. Select only by r2 MCC. A separate r1 head can use the observed coarse labels without pretending that aggregate r2 label 679 has one parent.

Do not initially force parent probability sums or hard parent gating over the full taxonomy. If a later experiment imposes consistency on the uniquely mapped classes, it must explicitly handle the aggregate class and relationships unseen in inner training. Avoid equal weighting of r0/r1/r2 as the primary objective.

Focal loss, inverse-frequency weights, extensive smoothing and ensembles have lower priority. Their earlier one-fold results do not settle the question, but there is currently stronger evidence for fixing convergence and testing the aggregate-class head. Large class weights may worsen overall r2 MCC even if rare-class recall improves. An ensemble, if eventually tested, must be labeled separately and given a matched DRFP ensemble and inference budget.

**A bounded search and two-GPU schedule**

Use this initial ceiling of 20 configurations per fingerprint. All settings not specified in a row follow the common trainer above. The list is a proposal to freeze before execution, not a claim that these are optimal values.

| Family | Configurations | Count |
|---|---|---:|
| Plain MLP | Width/dropout pairs `(512, 0)` and `(1024, 0.1)` × batch `{256, 1024}` × LR `{0.0003, 0.001}` | 8 |
| Capacity controls | `[1024,512]` and two-block residual width 512 × dropout `{0, 0.1}`; batch 256, LR 0.001 | 4 |
| Soft aggregate head | Width/dropout pairs `(512, 0)` and `(1024, 0.1)` × batch `{256,1024}`; LR 0.001, matched to plain controls | 4 |
| r1 auxiliary head | Lambda `{0,0.02,0.05,0.10}`; width 1024, dropout 0.1, batch 256, LR 0.001 | 4 |

Within each outer fold, screen all 20 on one inner fold with initialization seed 0; advance the top six to the other two inner folds. Take the top two by their three-fold mean and repeat both on all three inner folds with initialization seed 1. Select by the resulting mean r2 MCC, breaking differences below 0.001 in favor of the simpler/faster configuration. This selection rule is identical for both fingerprints. Screening on one fold can miss a strong candidate; document that tradeoff rather than calling it an exhaustive search.

That is at most `20 + 6*2 + 2*3 = 38` inner fits per fingerprint per outer fold. Over 25 outer folds and two fingerprints: **1,900 inner fits**. Three final initialization seeds per fingerprint/fold add **150 refits**, giving **2,050 fits** for the main nested comparison, excluding a small trainer-reproduction check. These three final models estimate seed variability; they are not automatically an ensemble.

Start with one outer fold as a pipeline and timing check (76 inner fits plus six refits), without opening its outer-test scores. Correct only implementation/protocol failures at that stage; changing the candidate list based on its results starts a new versioned exploratory study. Then run the frozen procedure across all folds.

Use one persistent worker per GPU, cache the two feature matrices, and cap CPU threads per worker so the CPU benchmark queue can continue. Alternate fingerprints across GPUs or use a shared work queue; avoid tying one fingerprint permanently to one device. Run matched jobs to comparable completion, without dropping difficult folds.

The old r2 benchmark needed roughly 2.5 seconds of GPU fitting per SynRFP run, but that is not an ETA for the larger search. For planning, a measured average of 10–30 seconds per new fit would imply about **6–17 hours on one GPU or 3–9 hours on two GPUs**, before overhead (`2,050 * seconds_per_fit / number_of_GPUs`). Recompute the ETA from the first pipeline/timing batch and actual device availability. Do not promise immediate completion based on the small pilot timings.

**Decision criteria and reporting**

The first practical target is to close the current 0.0063 MCC gap to the frozen DRFP reference. The stronger success criterion is positive paired r2 MCC against equally tuned DRFP, with repeat-level stability and an uncertainty analysis that respects reused reactions/scaffold groups. A predeclared improvement of 0.002 MCC is a useful practical margin to distinguish a useful gain from a near tie; it is a decision threshold, not a prediction.

Report original-configuration SynRFP/DRFP and nested-tuned SynRFP/DRFP side by side. Include per-fold and per-repeat differences and initialization variability. Report parameter counts and compute cost: identical hidden-layer settings give DRFP more input-layer parameters because its fixed input has twice as many bits. Do not treat the 25 folds or 75 seed/fold evaluations as independent samples. With saved out-of-fold predictions, use scaffold-cluster resampling of paired predictions, keeping repeat/seed predictions for a reaction together; describe its interval as conditional on fitted models and show training variability separately.

Report aggregate-class errors, rare-class recall and macro-F1 even when overall MCC improves. Keep unseen-training-label rows in the benchmark and expose their contribution. Present the seed-42 Track B result separately, without tuning against its test score or pooling it with Track A.

If equally tuned DRFP remains ahead after the bounded search, report that result and use the validation diagnostics to decide whether another MLP hypothesis is justified. Stop repeating searches simply to obtain a favorable test score. If diagnostics point to information lost by the fixed representation, document the constraint rather than silently changing the fingerprint.

After r2, apply the validated training procedure to TPL balanced and unbalanced in a separately budgeted study. Preserve the other 11 original benchmark comparisons; if claiming one universal MLP configuration, select that universal rule inside an appropriate development protocol and test it across all datasets. Dataset-specific selected heads must be described as such.

**Implementation deliverables when this plan is executed**

- A frozen study manifest with candidate definitions, partition/initialization seeds, label maps and source/feature/split hashes.
- A common trainer accepting SynRFP or DRFP, with plain, residual, aggregate and auxiliary MLP heads and consistent logging.
- A nested scheduler that exposes only the current development pool to selection and stores outer predictions separately.
- Validation diagnostics and convergence plots, paired result tables, repeat/seed variability, and an explicit decision for each tested hypothesis.
- Targeted checks for row alignment, inner/outer scaffold separation, probability normalization, aggregate-label encoding and deterministic resumption. Preserve the original benchmark artifacts.

Evidence locations: `results/benchmark/syntemp_r2/`, `results/mlp_model_selection/`, `results/r2_mlp_tuning_finalouter/`, `results/r2_aux_hierarchy/`, `results/multitask_hierarchy/`, `results/mlp_gpu_resume/status.json`, `../work/export/ds_syntemp_r*.csv.gz`, `../work/features/syntemp/`, and `../work/splits/syntemp_r*/`. Protocol code reviewed: `mlp_cuda_worker.py`, `mlp_gpu_queue.py`, `mlp_model_selection.py`, `multitask_hierarchy_pilot.py`, `r2_aux_hierarchy_pilot.py`, `r2_wide_relu_queue.py`, `../synrfp_exp/data.py` and `../synrfp_exp/splits.py`.
