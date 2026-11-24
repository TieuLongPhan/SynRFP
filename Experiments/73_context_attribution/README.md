# E1. Fixed-budget attribution of yield differences

This study asks whether the pooled context channel improves yield prediction
when the compared fingerprints use the same number of bits and receive the
same condition identities. It carries forward the feature controls from the
manuscript review. The partitions and encoder below define a new experiment.

## Workstation run

Start from a SynRFP clone. Use a separate worktree for this study.

```bash
git fetch origin exp/context-attribution-e1
git worktree add --detach ../SynRFP-context-e1 origin/exp/context-attribution-e1
cd ../SynRFP-context-e1
python -m pip install -e . -r Experiments/73_context_attribution/requirements.txt
```

Use a Python 3.12 environment. Run the following from the new worktree.
Set `data_dir` to the local SynRXN directory containing both input files.
The example permits eight simultaneous CPU fits with one numerical-library
thread per process. Adjust `--jobs` for the workstation and other jobs.

```bash
set -euo pipefail
data_dir=../SynRXN/Data/tables/studies/classic_yield
run_dir="Experiments/work/context_attribution/E1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"

python Experiments/73_context_attribution/features.py \
  --data-dir "$data_dir" --output-dir "$run_dir/features" --check-only \
  | tee "$run_dir/source-audit.log"

python Experiments/73_context_attribution/features.py \
  --data-dir "$data_dir" --output-dir "$run_dir/features" \
  | tee "$run_dir/encoding.log"

python Experiments/73_context_attribution/run_ablation.py \
  --features-dir "$run_dir/features" --output-dir "$run_dir/results" --check-only \
  | tee "$run_dir/preflight.log"

python Experiments/73_context_attribution/run_ablation.py \
  --features-dir "$run_dir/features" --output-dir "$run_dir/results" --jobs 8 \
  | tee "$run_dir/fitting.log"
```

### Reconstruct missing input tables

SynRXN keeps the two compressed references out of Git. From the experiment
worktree, the command below reconstructs them from three pinned author files,
verifies their exact output hashes and writes into a new local directory.

```bash
run_dir="Experiments/work/context_attribution/E1-$(date -u +%Y%m%dT%H%M%SZ)"
python Experiments/73_context_attribution/prepare_inputs.py --output-dir "$run_dir/inputs"
(cd "$run_dir/inputs" && sha256sum -c SHA256SUMS)
```

Use `data_dir="$run_dir/inputs"` in the source-audit and encoding commands
above, keeping the same `run_dir`. For an offline rebuild, add `--source-dir`
with the directory containing the pinned author files under the names
`bh.xlsx`, `suzuki_source.xlsx` and `suzuki_0.tsv`. The script checks the
author-file hashes before reading them. The two reconstructed tables retain
their local-reference status. [Pinned author repository](https://github.com/rxn4chemistry/rxn_yields/tree/d9e6b87ce1b881978490d68bfc00021e3b48127a)

The checks audit the input tables and feature coverage without fitting. Keep
raw inputs, features, row-level outcomes, split membership and predictions
on the workstation under `Experiments/work/` or in the local data checkout.
The input tables are distributed for local reference. A later public export
can include appropriate aggregate results and provenance.

## Data and grouping

The two tables used to verify this protocol have the following digests.

| Input | Rows | SHA-256 |
| --- | ---: | --- |
| `bh_classic.csv.gz` | 3,955 | `ab5cea6673bb6c1e0dd13f200b25fabcecf8f13ad5471d09f3004f3712fea2f7` |
| `suzuki_native_reference.csv.gz` | 5,760 | `c5a3354079455ad37462253861a76d04e724911b9fc23929f3b10f51ad3a29ce` |

- **Classic Buchwald–Hartwig.** Reconstruct the intended product from the aryl
  halide and constant 4-methylaniline. Use the recorded ligand, additive
  and base together with the author's fixed Pd precatalyst. There are
  15 canonical aryl-halide groups and five distinct intended products.
  The grouped test withholds aryl-halide forms. Different held-out
  halides can lead to a product present in training.
  [Author-data reconstruction](https://github.com/rxn4chemistry/rxn_yields/blob/d9e6b87ce1b881978490d68bfc00021e3b48127a/rxn_yields/data.py)
- **Suzuki–Miyaura.** Use the intended product in the source reaction record
  and identify exactly one electrophile and one organoboron precursor.
  The 15 observed precursor pairs all lead to one intended product, so
  holding out a pair tests precursor-form transfer within one product.
  Individual partners may occur in other pairs. The adapter preserves
  the unspecified Pd bonds written as `~` in the source SMILES.
  [Perera et al.](https://doi.org/10.1126/science.aap9112)

The feature producer canonicalizes molecules and removes atom-map labels.
It uses additive radius-2 WL counts on disjoint components, then applies
the signed and pooled count operations. It checks every row and records
coverage by sample, precursor group, product and condition component.
This adapter preserves the source Pd bonds and restores the fixed BH
precatalyst. These choices differ from the preliminary feature prototype.
The new partitions and parser define a separate experiment from the
historical yield comparisons. SynRXN's
`Data/metadata/reconstruction/case_studies/classic_yield/` includes
reconstructed author outer memberships for a separate replay study.

## Controlled comparison

| Arm | Fingerprint |
| --- | --- |
| `d1024` | 1,024 signed-delta bits with seed 1 |
| `d2048` | 2,048 signed-delta bits with seed 1 |
| `d_pooled` | 1,024 signed-delta bits plus 1,024 pooled bits, seed 1 |
| `d_rehash` | 1,024 signed-delta bits with seed 1 plus 1,024 delta bits with seed 2 |

The 1,024-bit seed-1 delta is the XOR of the two halves of the 2,048-bit
delta. An independent rehash keeps the first delta block and the total
width fixed while using the second block for change.

Each arm is tested with conditions on the reactant side (`R`), on both
sides (`A`), or outside the reaction graphs (`S`). Conditions on both sides
cancel in the signed difference. In each placement, fit once with and once
without an identical component-presence block. Learn the available
component columns from fitting rows only. The block encodes component
identity. Multiplicity and chemical roles can affect the graph counts
independently.

The outer evaluations are ten 80/20 random partitions with seeds 0–9 and
five group folds. The validation partition is common across all arms in an
outer split and group-disjoint for grouped evaluation. Fit one 512-unit
ReLU MLP per cell using Adam and fitting-only target scaling. Checkpoint
by the lowest standardized validation MSE over at most 200 epochs,
with 15-epoch patience and tolerance 0.0001. The runner freezes
optimizer settings, split membership and inputs before fitting.

## Primary analysis and interpretation

The primary contrast is **pooled minus rehashed-delta test R² on the
grouped classic Buchwald–Hartwig evaluation with S placement and the
condition block supplied identically to both arms**. Report pooled
out-of-fold R² and the paired substrate-group bootstrap interval. The
interval is descriptive across held-out groups conditional on the saved
fits. A lower interval bound above zero supports extra predictive
information from the pooled channel on this dataset. An interval
spanning zero leaves that comparison unresolved.

Use `d_pooled` versus `d2048` to report the net shift at fixed width.
Use `d_pooled` versus `d1024` and `d2048` versus `d1024` to report the
effect of adding context at fixed delta width and the effect of changing
delta width. The other placements and Suzuki are secondary contrasts.
A gain concentrated in particular placements is evidence about those
encoding choices. Cross-laboratory transfer and experimental shortlist
quality require separate held-out evaluations.

## Inspect the output

A full run contains 720 cells across two datasets, 15 splits per dataset,
six placement/condition combinations and four encodings.

- `manifest.json` records the feature receipt, code hashes, versions and settings.
- `splits.csv.gz` records fit, validation and test membership by source ID.
- `predictions.csv.gz` records the held-out predictions for each cell.
- `results.csv` records per-cell scores and checkpoint epochs. `paired.csv`
  reports paired changes in R² and RMSE on the same test rows.
- `grouped_oof.csv` contains 48 grouped contrasts with 10,000 paired
  group-bootstrap draws. The primary row is flagged.
- `status.json` records completion and output hashes. Use the results
  once it reports `complete` and 720 cells.

Preserve the run directory, including the feature manifest and prediction
ledger. Use a new output directory for each run.

## Sequence after E1

The [E3 protocol](EXACT_DELTA_PROTOCOL.md) defines a follow-up with
joint holdouts of matched precursor forms and three 2,048-bit encodings.

1. Review paired effects and recipe placement alongside the width and
   rehash controls. Record the primary result before revising the paper.
2. Audit exact signed-delta equivalence classes on these sources. Report
   the number and sizes of multi-reaction groups and whether context
   explains residual differences on held-out groups. Compare with
   replicate variability where replicates exist.
3. Identify and freeze a truly independent campaign split with a source
   overlap audit. Evaluate best-of-three condition regret using only
   information available when selecting the experiment.
4. Add nested preset selection and uncertainty if broader benchmark
   claims depend on the selected settings. Prepare independent retrieval
   or timing evaluations when those specific claims are needed.
