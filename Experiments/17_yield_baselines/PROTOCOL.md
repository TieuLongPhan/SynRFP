# Yield baseline reproduction and matched evaluation

Prepared 12 October 2026. Sources and local inputs are pinned by SHA256;
models, checkpoints, clone assets and caches remain local in ignored paths.
No historical experiments or manuscripts are changed.

The primary comparison evaluates YieldBERT, YieldBERT-DA and the 2025 Graph
Neural Additive Network (GNAN) on the exact reaction rows, held-out tests and
seven-fingerprint common-valid populations in `13_yield/continuous_allfps_v1`.
The pilot is **9 fits**: three methods on BH `FullCV_01`, Suzuki `random_0`,
and the NCS `i0_f0` cluster partition. The full matched extension is **192 fits**,
three methods on all 64 original split sets. The first nine predefined job IDs
in this single plan are prioritized, followed by the remaining183; pilot
models are trained once and count toward192.

An additional **60-fit source-protocol comparison** uses the twenty classical
random holdouts and all outer-training rows. It uses fixed epochs without
test-driven checkpoint selection. It is kept separate from the matched
validation-held-out analysis. These adapters cannot reproduce historical
SimpleTransformers versions, optimizer numerics and reported numbers exactly.

## Sources and initialization

- `rxn4chemistry/rxn_yields`, commit `d9e6b87ce1b881978490d68bfc00021e3b48127a`:
  regression scripts, chemistry input recipe and SMILES augmentation.
- `seokhokang/interpretable_yield_prediction`, commit
  `070fc4540999265f389ee9c08ae21c5a32057e09`: GNAN model, graph featurizer,
  loss, scheduler, participant roles and published split assets.
- `chinmaymirji/morgan-fp-yield-prediction`, commit
  `60a1f5009d7e842b32ed5ce74270b6cdf0b77468`: context for arXiv:2609.22167.

YieldBERT starts from the **MLM `bert_pretrained` checkpoint**, with a fresh
scalar regression head. The classification-supervised `bert_ft` fingerprint
already in our comparison is a different model; it is not fine-tuned YieldBERT.
The matched adapter uses published pretrained-model hyperparameters: BH and
NCS ten epochs, dropout .7987, learning rate .00009659; Suzuki fifteen epochs,
dropout .5848, learning rate .00005812; batch16, max length300, AdamW with
linear decay and no warmup. Validation MSE selects the checkpoint.

YieldBERT-DA uses the same initialization and model. Each fit-training row
has **15 rotated SMILES variants with shuffled precursor component order**.
No training augmentation reaches validation or outer-test rows. The primary
test prediction averages **ten** deterministic variants; canonical predictions
are retained as a secondary result. These augmentation budgets are explicit
source-derived choices because the 2026 paper does not provide numeric K or
TTA counts. Augmentation preserves stereo, dative bonds and unspecified `~`
bonds. The paper's `~`-to-fragment replacement is not applied to matched inputs.

GNAN reuses the author's architecture and featurizer, with 500 epochs,
batch128, AdamW .001/.05, ten-epoch warmup and cosine decay to .00001.
Its original additive/heteroscedastic loss is preserved. Validation is assessed
every ten epochs with five dropout passes; lowest validation MSE selects the
matched checkpoint. Test predictions average thirty dropout passes. The
adapter transports category-index tensors to the graph device, fixing the
author code's CPU/CUDA tensor-placement boundary without changing values.

The author's processed BH molecules omit some ligand stereo and use a
different palladium coordination drawing. Matched GNAN uses our frozen
parent molecules with the same six participant roles. Every retained component
pool and product is checked against the parent input by canonical isomeric
SMILES. Constant role columns are removed as in the author implementation.
This is a matched adaptation rather than a bit-for-bit reproduction of its
processed input chemistry. A full NCS eligibility audit encoded all7,541 unique
participant/product molecules across22,609 retained reactions without parse
or shape failures. Native vocabulary fallbacks affect12,812 reactions, with
overlapping counts: Si6,510, charge−2 6,796, DATIVE144, Ni27, Ag6 and H1.
The published featurizer maps these unrecognized categories to zero-hot
fields. We retain that behavior and all rows, and disclose its limitation for
the NCS extension. Native zero-node participant graphs are also preserved;
all-empty and mixed-empty graph controls have finite model outputs.

## Information boundaries and outcomes

Target means and scales use actual fit-training rows only. Worker archives
contain `y_train` and `y_validation`; they **do not contain test outcomes**.
The controller evaluates saved predictions afterward, verifies exact row
coverage, and preserves all weights and preprocessing information.

Primary matched predictions are unbounded for all methods. Classical clipped
predictions are also retained for comparison with GNAN's published convention.
NCS predictions and observations are never clipped; values above100 remain.
The NCS common-valid mask retains 22,609/27,497 source records and excludes
all4,312 Dreher–Doyle2018 reactions plus576 Dreher2014 records because crxnfp
does not encode those inputs. Source composition, assay heterogeneity and
missing additives limit interpretation.

The NCS paper [Neves et al., DOI10.1038/s43588-026-01017-6](https://www.nature.com/articles/s43588-026-01017-6)
primarily benchmarks binary success under heterogeneous assays. Its source OOD
benchmark additionally excludes test aryl/amine substrates seen in training.
Our forty cluster partitions are a separate continuous-response extension;
they do not establish continuous-yield SOTA against the paper. The released
binary labels should be reused rather than replacing source-specific labels
with a newly computed universal threshold.

The 2026 Morgan preprint labels its graph baseline GNAN but cites a different
2022 uncertainty GNN paper. It also states BH train2767/test830, whereas the
released population has3955 rows and the actual test size is1188. We use the
verified released splits and identify the graph baseline by its actual source
and 2025 publication, DOI10.1088/2632-2153/addfaa.

## Execution

The isolated environment inherits existing RXNFP packages read-only and adds
only DGL1.1.3+cu121 and psutil to its own directory. Existing GPU jobs are left
running. The controller requires the selected physical GPU's property unit
to be inactive and checks for declared active GPU workers; `--wait` records
a queued state until that prerequisite is met. The dynamic dual controller
uses per-job claims and the existing physical GPU lease files. GPU1 waits for
the verified20-fit SynCat reference and its unit to finish before joining.
Every completed checkpoint is checked against its original worker SHA
attestation before it counts toward the benchmark. The failed provisional
v1 safe-loading preflight is retained separately with zero benchmark fits.
CPU compatibility checks are excluded from all benchmark results.

```bash
Experiments/17_yield_baselines/env/bin/python Experiments/17_yield_baselines/prepare_baselines_v2.py --scope full
Experiments/17_yield_baselines/env/bin/python Experiments/17_yield_baselines/run_baselines_dual.py --plan Experiments/17_yield_baselines/runs/matched_full_v2/plan.json --gpu 0
Experiments/17_yield_baselines/env/bin/python Experiments/17_yield_baselines/run_baselines_dual.py --plan Experiments/17_yield_baselines/runs/matched_full_v2/plan.json --gpu 1 --after-unit synrfp-syncat-gpu1-v1.service --after-complete /home/labhhc4/.local/share/synrfp-runs/syncat_reference_v1/COMPLETE.json
```

Timing and full-suite ETA must be calibrated from completed GPU pilot fits.
Prepared plans and passing CPU checks are not completed model benchmarks.
