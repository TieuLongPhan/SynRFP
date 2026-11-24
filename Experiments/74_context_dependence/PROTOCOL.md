# When is chemical context worth the bits?

Research and workstation plan, 24 November 2026. This document specifies proposed analyses. Completed evidence is identified below. The next workstation job is the input and split audit.

The protocol lives in SynRFP at `Experiments/74_context_dependence/PROTOCOL.md` on `exp/target-dependence-plan`, based on the E3 release `b1f51d16`. The input catalogue inspected for the plan is in SynRXN at `926d6c2`, with the local file hashes below. To obtain the protocol on a workstation with a SynRFP clone, run the following from the clone.

```sh
git fetch origin exp/target-dependence-plan
git worktree add --detach ../SynRFP-target-dependence origin/exp/target-dependence-plan
```

## Research question

**When does retaining the molecular environments shared across a reaction improve prediction enough to justify spending fewer bits on the transformation? Can we predict which of two targets will benefit more before seeing their test outcomes?**

The first question applies across the seven existing tasks. The second gives the study a falsifiable extension. Change the target on the **same reaction records** while fixing the learner, split and bit budget. A result on two computed properties tests dependence on the target within regression. A replication using two classification labels tests whether the observation carries to a second kind of problem.

## What the paper can say today

| Evidence | Inference it supports |
| --- | --- |
| Full signed counts and a shared recipe are identical for some distinct precursors. | Every deterministic model with those inputs must give the pair the same prediction. This statement holds before hashing. |
| Under nested parity folding, the SI decomposes the population squared-error gain into the benefit of adding context at narrow delta width minus the cost of narrowing delta. | Either term can dominate. This algebra does not determine what a fitted model will learn. |
| Across 72 constructions, side combination explained the largest share of between-construction variance in six of seven tasks. On TPL B, assigning half the bits to pooled context raised classification MCC by 0.033 and lowered clustering AMI by 0.093. | The same reallocation had different consequences on the same source under different tasks and protocols. Those comparisons also change the learner or evaluation. |
| Nine joint Buchwald–Hartwig holdouts covered 2,368 recipe-matched gaps in three exact-count classes. Equal-class mean squared gap error fell 34.4%, with two classes improving. The separate 720-fit sensitivity study missed its frozen whole-screen criterion. | The loss of shared environments can matter at equal width and recipe access. Its magnitude and prevalence on new chemistry remain open. |

MatText varied how strongly a target depended on two information sources and checked the consequences on scientific properties. SynRFP can use that experimental logic and add an exact check of which inputs are indistinguishable. The biological and physical meanings of the targets come from their own sources. [MatText](https://www.nature.com/articles/s43588-026-01045-2)

## 1. Assemble the cohorts and protect the test outcomes

| Role | Data and targets | Use |
| --- | --- | --- |
| Regression development | SynRXN `Data/tables/canonical/property/b97xd3.csv.gz`, 16,365 rows with `ea` and `dh`. `cycloadd.csv.gz`, 5,269 with `G_act` and `G_r`. | Debug the two-target pipeline, overlap checks, splits and compute budget. Their barrier results already feature in the benchmark. |
| Primary non-yield test | QuantumPioneer peroxyl hydrogen-atom transfer. SynRXN `Data/tables/coverage/physics_rates/quantumpioneer_kinetics.csv`, 166,158 rows. Use exactly `barrier` and `deltaHrxn`, the forward barrier and reaction enthalpy with the source's zero-point correction, both in kcal/mol. | Two computed quantities on the same structures. Use only reaction-side graphs as input. The final test measures performance on held-out chemistry after fitting on this source's training partition. [Source data](https://zenodo.org/records/20060606) · [Schema](https://github.com/QuantumPioneer/database) |
| Classification development | SynRXN `Data/tables/canonical/classification/ecreact.csv.gz`, 185,734 rows with EC hierarchy levels. | Exercise multiclass scoring and audit overlap. Hierarchy levels share an annotation; the table lacks an independent transformation label. |
| Classification replication | EnzymeMap v2 `enzymemap_v2_brenda2023.csv.gz`. The original file has 349,458 rows. Selecting `source == direct` and `steps == single` leaves 231,492 rows and 32,920 distinct literal unmapped strings before canonicalization. | Pair the first two EC fields (e.g. EC 1.1) with an EC-blind bond-edit label on identical rows. The local mapping derivative lacks the EC field. Retrieve the original v2 file with its attribution and provenance. [Versioned release](https://zenodo.org/records/8254726) · [Source methods](https://github.com/hesther/enzymemap) |

Input SHA-256 digests, in the table order for B97XD3, Cyclo, QuantumPioneer and the original EnzymeMap file:

```text
33f881612842378a6cff97b2db0c99b1c4877a7e81a318129a26dd58e9702d73  b97xd3.csv.gz
afd03a88ef6268350f2c176b70a67396ff1902b85e6742bb2ac3ef0d9ca84004  cycloadd.csv.gz
e81aa5f9158564e64356a524c046469e2a13e471469ff641713ded3c37ff1520  quantumpioneer_kinetics.csv
0a70759a7cec5720b6aeb2d63c887a670b9ac2a6ada7c0a1ef3c1740ee3cc2cf  enzymemap_v2_brenda2023.csv.gz
```

**Audit before fitting.** Pin the local adapter and source revisions. Write row IDs, graph hashes, parser failures, units and both-target availability. Deduplicate direction-sensitive canonical reaction graphs. Use a second, direction-insensitive key to group reversals and to audit overlap with development and previously scored benchmark sources. Preserve genuine annotation conflicts in the audit. Record every exclusion and coverage by source. Keep identifiers, atom maps, enzyme names, source IDs, rate coefficients and other target columns out of the features. Check explicit hydrogens, radicals and charge in QuantumPioneer, and tally which distinctions the standard graph profile drops.

**QuantumPioneer split.** The released projection gives 133,299 training, 16,530 validation and 16,329 test rows across 109,093, 13,551 and 13,349 disjoint reactant-pair groups. Verify these counts before exclusions. Freeze the common eligible row set for both targets and all arms. Where a canonical duplicate or reversal crosses a boundary, drop its entire connected group and record the loss. Remove prior development/benchmark overlaps from the external test. Report recurring individual partners and shared scaffolds separately. The test is a reactant-pair holdout in one reaction family.

**EnzymeMap split.** Collapse organism/protein copies, sided duplicates and reverse pairs. Record how many canonical reactions have several EC labels. Use unambiguous two-field EC assignments for the single-label primary replication and report discarded coverage. Derive the structural target from a frozen EC-blind mapper on the unmapped reaction and a canonical multiset of changed bonds with endpoint elements and before/after bond orders. Define the no-edit class and record mapping failures. The published rule ID is a sensitivity analysis because source curation can use the EC assignment. Remove known ECReact and other development duplicates/reversals. For the remaining records, group by product scaffold with canonical product identity for scaffold-free molecules, then merge groups linked by duplicate/reverse keys. Assign groups deterministically by hash to 80/10/10 train/validation/test; archive the salt and counts. Fit the label vocabularies on training only. Count unseen test labels as errors. If the EC-blind mapping, label support or overlap audit fails, publish the feasibility result and leave independent classification replication open.

## 2. Fix the fingerprint comparison

Use the radius-2 WL feature profile and parity folding from the pinned attribution implementation. Reuse its independent delta hash and yield E1/E3 code where applicable. Use the same reaction role parsing, training rows, validation rows, learner settings and input order in every arm. Verify exact count equality with canonical token/count tuples behind any digest.

| Arm | Input | Purpose |
| --- | --- | --- |
| Narrow delta | 1,024 delta bits | Measure the cost of narrowing the change channel. |
| Standard delta | 2,048 delta bits | Primary practical comparator. |
| Pooled context | 1,024 delta + 1,024 pooled bits | Intervention. |
| Rehashed delta | 1,024 delta + 1,024 delta bits from an independent seed | Check whether a second encoding of change explains the gain. |
| Reactant context | 1,024 delta + 1,024 bits from pooled reactant-side fragments | Test whether a simpler description of the starting materials suffices. |

Use separate single-output, 512-ReLU-unit MLPs for each target, Adam with learning rate 0.001, batches of 256, at most 200 epochs and patience 15. Minimize validation MSE for regression and cross-entropy for classification, keeping the best checkpoint. Start with the E1 optimizer/regularization implementation and pin its complete settings in the run manifest. Standardize each regression target using its training mean and standard deviation. Pair seeds 17, 29 and 43 across arms. Report mean losses across seeds; treat seeds as sensitivity to fitting. No target-specific tuning on the external data. Fit only after checking that row alignment, metrics and grouping survive a small development smoke test.

## 3. Test the mechanism under known target dependence

On development hardware, construct a balanced factorial of abstract reaction graph pairs. Cross at least 32 transformation motifs with 32 disconnected shared components. Include multiple graph instances per factor, use generator attributes to define the labels, and admit graph pairs only after verifying that varying the shared component leaves the full signed counts identical. Check how often the pooled fingerprint actually distinguishes component values. Duplicate map numbering or row order must not distinguish examples.

Give change and context independent, balanced numerical attributes. Define the synthetic response as `sqrt(1-a) × change + sqrt(a) × context` for `a = 0, .25, .5, .75, 1`. Standardize attributes on training rows and score regression. For classification use the sign of the same response with a fixed zero threshold and a declared tie rule. Hold out transformation–component combinations while keeping each factor represented in training. Repeat three independently generated grids. Use a shuffled context block as a negative control, permuted independently within split and transformation. Check that an oracle using the two generating attributes reproduces the target. This is a probe of information access and learning on known factors.

Report the full curve for the standard, narrow, pooled and rehashed arms. The prespecified contrast is the increase in the pooled-versus-standard gain between the pure-change and pure-context endpoints. Report the two endpoint gains separately. At pure context a delta-only learner cannot distinguish component values within a transformation. A new structure or token leak, an unbalanced held-out grid, or failure of the oracle calls for debugging on synthetic data before scoring real holdouts. A flat or reversed curve with the controls passing is a scientific outcome.

## 4. Make a prediction using training data

Compute a deliberately simple diagnostic on each real source's training partition. Split its independent reaction groups into five folds. For each query, find records in other folds with identical full signed counts and different canonical chemistry. Remove duplicates and reversals. Give each reference group one vote. Keep queries with at least three reference groups and at least two distinct pooled bit vectors among the query and references. Use the same eligible queries for both targets.

Predict each target twice. The delta predictor uses every reference group, weighting the rows within each group equally and giving each group the same total weight. The context predictor uses the three closest reference groups in pooled-bit Jaccard distance. Represent each group by its closest reference, averaging tied references within the group. At the third-neighbour boundary give tied groups equal fractional weights summing to the remaining places. Take the weighted mean for regression or the weighted majority for classification. Break tied class votes by training-fold frequency, then by sorted label. Define the diagnostic as the reduction in squared error divided by training-fold target variance, or the reduction in single-label error rate. Average first within query group, then equally across groups. Log eligible coverage, context-bit ties, independent groups and exact-delta classes. This measures whether context can separate outcomes inside a repeated count class. Finite bit collisions elsewhere may affect the eventual model gain.

**Prediction rule.** Name the target with the larger diagnostic as the one expected to gain more from pooled context. Record a tie as an abstention. As proposed feasibility thresholds, require at least 5% of eligible-source training rows, 30 query groups and 10 repeated exact-delta classes contributing to the diagnostic. Record these choices before using either external target pair. Run a development-only within-delta context permutation to check for spurious predictive power. For an external abstention, release the arm scores as descriptive and count the abstention when reporting coverage. Changing a diagnostic after unblinding creates a new hypothesis for a fresh source.

## 5. Score the untouched test once

For each target, compute the test loss of standard delta minus the test loss of pooled context. Call this gain **G**. Average each loss first within the held-out dependence group, then equally over groups. For regression divide squared error by that target's training variance, making gains comparable across the two outcomes. Also report RMSE and MAE in kcal/mol for QuantumPioneer. For classification the primary loss is the zero–one error; report macro recall, MCC, class frequencies, training-majority baseline and gains relative to its error as sensitivities.

The primary quantity is **G for the predicted higher-gain target minus G for the other target**. Save the predicted ordering, training diagnostics, input/split hashes and code revision before opening any external score. Generate 10,000 paired group-bootstrap draws with a recorded seed, resampling exactly the same groups across both targets and all arms. Average seeds inside each draw. Report 95% percentile intervals, both target gains, and the full paired distribution. Intervals describe variation over test groups conditional on these fits. For regression report the correlation between targets. For classification report their contingency table and normalized mutual information. Estimate likely precision from development-group variation before fitting the external arms, then publish the interval actually achieved.

| Outcome | Consequence for the paper |
| --- | --- |
| The primary interaction interval lies above zero. | The diagnostic correctly predicts relative benefit on this source. Claim an absolute benefit on the higher-ranked target only if its own gain interval also lies above zero. |
| The point ordering agrees and its interval spans zero. | Report the predicted and observed order and the uncertainty. Comparative predictive value remains unresolved. |
| The interval lies below zero, or the higher-ranked target loses to standard delta. | Respectively report a failed ordering or a cost at the fixed bit budget. Use narrow delta, rehash and reactant context to interpret the result. |
| The same ordering test succeeds in EnzymeMap. | Report transfer across these two target pairs and scientific settings. Each source remains one replication. Keep additional seeds, related endpoints and the 18 factorial settings out of the replication count. |
| Coverage fails or a source is exposed before the prediction is recorded. | Report the coverage or exposure. Treat any resulting scores as exploratory. A new independent source is needed to test a revised rule. |

Treat the QuantumPioneer interaction as the single primary test; EnzymeMap is a separately declared replication. Report both outcomes regardless of direction. For control arms report paired estimates and intervals descriptively. A gain over rehash alone does not establish a gain over ordinary delta. If reactant context matches the pooled arm, the evidence supports retaining starting-material information, with the advantage of pooling both sides unresolved.

## Execution order, resources and release

1. **Day 1, inputs and exposure inventory.** Work in `Experiments/74_context_dependence/` on the experiment branch. Pin source files, group logic and software. Write source and split manifests, failure tables and the eligibility flow. Preserve the source notices.
2. **Days 2–3, implementation and development.** Implement the adapters, synthetic generator, grouped diagnostic, training and paired analysis. Check equality and rehash controls, assert forbidden feature columns are absent, hand-check a small paired loss, and reconstruct every summary from row predictions. Use B97XD3/Cyclo and ECReact for development. Record smoke-test memory and seconds per fit before scheduling full jobs.
3. **Next, freeze and run.** Commit the implemented protocol and runner; record any deviations with the reason and whether external scores were accessible. Compute and timestamp the prediction on training only. Score QuantumPioneer, then the independently frozen classification replication. Five arms × two targets × three seeds give 30 model fits per real source (60 across both external sources). Development and synthetic runs add to this budget. Reuse encoded matrices, set worker and numerical thread limits from the smoke test, and checkpoint by source, target, arm, split and seed.
4. **Release.** Return `input_manifest.json`, row membership and exclusion reasons, group and split hashes, feature schema, training-only `prediction.json`, OOF diagnostic rows, held-out predictions for each arm and seed, native and normalized scores, bootstrap settings, environment lock, runtime, runner revision and checksums. Reruns must reproduce predictions or document numerical tolerance. Record group and source overlap, coverage and unscorable records. Publish permitted artifacts and reconstruction instructions in SynRFP. Regenerate paper quantities from the pinned result release.
5. **Revise after results.** Put the prespecified prediction beside the observed gains and their paired interval in the main figure. Keep the seven-task comparison as the evidence for breadth and the matched-yield case as the experimentally measured consequence. Put coverage, all controls and failed predictions in the SI. A synthetic-only success supports the mechanism under controlled conditions. If the external interaction remains unresolved, retain today's bounded claim. Independent-campaign yield replication is a later test of the existing yield result.
