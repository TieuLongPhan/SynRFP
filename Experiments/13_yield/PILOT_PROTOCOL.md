# NCS 2026 Buchwald–Hartwig HTE: separate CUDA pilot

This extension leaves all settings and results in `12_validation` unchanged.
The first comparison is a technical pilot, not the complete published benchmark
or an independently optimized claim of state-of-the-art performance.

## Data and fixed task

- Publication: [Neves et al., Nature Computational Science (2026)](https://www.nature.com/articles/s43588-026-01017-6).
- Release: [bh-hte-ood](https://github.com/schwallergroup/bh-hte-ood),
  `data/BH_HTE_Curated_Dataset_v20260219.csv`, 27,497 original rows.
- CSV SHA256: `247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333`.
- Preserve released `labels` exactly. They are binary reaction success labels.
  There are 28 AZ ELN rows whose labels differ from applying a pooled 10% yield
  cutoff; the audit records these, without relabeling them. The release pools
  measurement methods, so this pilot does not treat all numeric `Yield` values
  as a uniform continuous regression target.
- Use author-released `iteration_0 train_test` entry 0 as the single preselected
  outer test. Confirm disjoint cluster sets using `iteration_0 cluster`.
- Make a stratified 10% validation split **within outer training only**, seed42.
  No outer-test label enters training, stopping, threshold choice, or tuning.
- Retain original row identifiers, source labels, split assignments, a dataset
  byte hash and ordered reaction hashes. Report shared-valid coverage and each
  omitted row. Never substitute malformed reaction strings.

## Input and feature comparison

Include aryl, amine, catalyst, base and solvent on the reactant side, with the
released product on the product side. Use the same chemical information for
both fingerprints. These are condition-aware inputs, separate from the original
classification benchmark input convention.

Valid SMILES dative arrows can contain `>`; parse component columns directly
instead of naively splitting the resulting reaction string on every `>`.
The component adapter must preserve chemistry and pass numerical equivalence
checks against the existing encoders on their supported inputs. Report actual
unsupported molecular inputs explicitly.

- SynRFP: unchanged WL, radius2, 1024bits, parity, delta, fpseed1;
  node attributes element/aromatic/charge/hcount, edge order, bbit2.
- DRFP: official `DrfpEncoder`, radius3, 2048bits, rings=True.
- This is not a reproduction of the authors' precomputed column called `DRFP`;
  that column is abbreviated in the CSV and is not loaded as a usable feature.
- Evaluate both on the intersection of their valid rows with frozen author
  outer assignments. Audit canonical/exact duplicate overlap, without moving
  rows between author partitions.

## Fixed model and evaluation

One controlled CUDA MLP per fingerprint: 512-unit ReLU hidden layer, one binary
logit, Adam learning rate0.001, weight decay0.0001, batch256, at most200epochs,
seed42, unweighted binary cross entropy. Select the saved checkpoint by lowest
validation loss; patience15 with minimum improvement0.0001. No architecture
search. Equal model settings and preprocessing for the paired comparison.
This PyTorch MLP is a new controlled comparison, not the paper's sklearn LBFGS
MLP. All fingerprints stay unchanged.

Report balanced accuracy, AUROC, average precision, F1, MCC, accuracy, Brier
score and log loss, with fixed probability cutoff0.5 and source-specific scores.
Include the outer-training positive-rate baseline. Report test sample counts,
coverage, epochs, validation loss, fit time, CUDA device, and feature/data/code
hashes. A single outer test and seed do not support statistical superiority.

## Execution and local retention

Use physical GPU1 only; GPU0 remains available to the user. Require CUDA and
fail explicitly if unavailable. Acquire the shared `12_validation/runs/gpu_1.lock`
before training. CPU preparation and training support stay on CPUs8–11 with
at most4threads and16GiB RAM. Keep datasets, features, probabilities, split arrays
and trained weights under local-only `data`, `features`, and `runs` directories.
Never delete local weights or add them to Git. Portable summaries and metadata
can be copied to `publication_metadata` after completion.

After the paired pilot, resume the repaired modern encoder on GPU1 through the
same lock. Completed encodings must be reused after strict provenance checks.
Expanding the BH benchmark to all40 author tests/all5fingerprints is a separate
run whose scope and runtime are determined from this pilot.

