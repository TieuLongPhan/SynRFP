# EnzymeMap coverage audit

This run checks whether the training scaffold groups can support the paired-target diagnostic in the [target-dependence protocol](PROTOCOL.md). The source archive is included with the checkout. The audit canonicalizes reactions, resolves copies, checks overlap, assigns scaffold groups, and encodes the training partition. It uses CPUs and requires neither an atom mapper nor a fitted model.

## Run from the SynRFP checkout

Use the same Python 3.11+ environment as the E1 and QuantumPioneer experiments. For a fresh environment, install the pinned attribution requirements and the current worktree.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Experiments/73_context_attribution/requirements.txt -e .
```

From the repository root, set `--synrxn` to the local SynRXN checkout and `--benchmark-root` to the archive containing the original factorial and full-yield run directories. Choose a fresh output path on repeat. Thread limits avoid multiplying the numerical thread count by the number of workers.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python Experiments/74_context_dependence/precheck_enzymemap.py \
  --synrxn ../SynRXN \
  --benchmark-root "$HOME/.local/share/synrfp-runs" \
  --output Experiments/work/74_context_dependence/enzymemap-precheck-v1 \
  --jobs 8
```

`--synrxn` locates the raw ECReact archive and older canonical property tables. Supply `--ecreact` or `--properties-root` for other locations. The historical source keys and checksums come from the published QuantumPioneer overlap receipt. `--benchmark-root` should contain `all_endpoint_factorial_full_v1`, `cross_task_factorial_v1`, and `yield_fullrow_v1`. With partial archives the audit still runs and identifies the missing inputs. The scope check also verifies the exact ECReact table used in the factorial in addition to the raw ECReact archive. Both EnzymeMap and ECReact draw on BRENDA.

For a relocated archive, pass `--table-overrides paths.json`. The JSON object maps names from the historical receipt, such as `all_endpoint_factorial_full_v1/ecreact`, to file paths relative to that JSON file or to absolute paths. The raw archive key is `synrxn_classification_raw/ecreact`. All files still go through checksum verification.

If SynRXN has reorganized its inputs, extract the pinned ECReact and property archives into a scratch directory and use that directory for `--synrxn`.

```sh
synrxn_inputs=$(mktemp -d)
git -C ../SynRXN archive 926d6c2 Data/classification/ecreact.csv.gz Data/property | tar -x -C "$synrxn_inputs"
```

## Return the result

Send `eligibility_precheck.json` and `overlap.json`. Keep `cohort.csv.gz`, `exclusions.csv.gz`, `training_encodings.csv.gz`, and `exact_delta_classes.csv.gz` with the run for reproducibility. The cohort ledger contains no target values. `eligibility_precheck.json` records hashes of those outputs, software versions, and the checkout commit.

| Decision | Action |
| --- | --- |
| `provisional_overlap_incomplete` | Use `overlap.json` to locate any missing, unverified, or unresolved sources. The current bound is advisory. |
| `stop_insufficient_upper_bound` | Even before mapping, fewer than 30 scaffold groups or 10 exact-delta classes can qualify. Record the abstention and stop the study. |
| `proceed_to_label_audit` | Map the structural labels and rerun the diagnostic on finalized training rows. Freeze an ordering only when coverage reaches 5% of training rows, 30 query groups, and 10 exact-delta classes. |

An exact class contributes to the optimistic bound if it spans at least four distinct training scaffold groups and two distinct pooled bit codes. Cross-fold references and successful label mapping can only reduce those counts on the fixed groups. The percentage of training rows is assessed after label validation fixes the denominator. An upper-bound pass schedules the label audit. The final prediction and the 30 fits are a later stage.

[EnzymeMap v2 archive and citation](https://zenodo.org/records/8254726) · [Original archive and notices](inputs/README.md) · [Historical benchmark source checksums](publication_metadata/quantumpioneer_v1/audits/prior_benchmark_overlap.json)

## Run the training label audit

Run this stage after the overlap audit clears the cohort. Use the pinned RXNMapper
v2 workstation environment with Torch and Transformers for mapping, then return
to the E1 environment for the fingerprint diagnostic. The label stage works
only on the frozen training partition.

```sh
study=Experiments/74_context_dependence
runs=Experiments/work/74_context_dependence
assets="$HOME/.local/share/synrfp-runs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false

"$mapper_python" "$study/map_enzymemap_labels.py" \
  --precheck "$runs/enzymemap-precheck-v1" \
  --followup "$runs/enzymemap-refusal-audit-v1" \
  --mapper-source "$assets/rxnmapper_v2_source_bfa575bc" \
  --mapper-deps "$assets/rxnmapper_v2_deps_v1" \
  --output "$runs/enzymemap-labels-train-v1" --jobs 8

"$e1_python" "$study/audit_enzymemap_labels.py" \
  --precheck "$runs/enzymemap-precheck-v1" \
  --followup "$runs/enzymemap-refusal-audit-v1" \
  --mapping "$runs/enzymemap-labels-train-v1" \
  --source "$study/inputs/enzymemap_v2_brenda2023.csv.gz" \
  --output "$runs/enzymemap-diagnostic-v1" --jobs 8
```

The mapper receives only the unmapped reaction SMILES. The primary label
requires a bijective element- and isotope-preserving atom correspondence and
exact recovery of both source sides when the maps are removed. It counts the
multiset of bond changes by endpoint elements and ordered before/after bond
type, including absence and aromatic bonds. The `no_edit` class records an
empty bond multiset. Charge, stereo, and hydrogen-state changes do not add
entries to this label. Record refusals individually and report coverage on
the frozen training rows as well as on the validated paired subset.

The mapping directory stores atomic per-batch checkpoints. Reusing the output
path resumes missing batches only when the manifest matches. The diagnostic
replays every accepted edit and signed-count signature, joins the existing EC
label by verified training row ID, and saves `prediction.json` and its training
OOF ledger. The two targets share GroupKFold partitions and query/reference
eligibility. Keep mapped reactions and the OOF ledger local. The prediction is
an ordering to test in the later held-out replication.
