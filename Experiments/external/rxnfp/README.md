# RXNFP baseline (separate environment)

RXNFP pins Python 3.8, an old `transformers` and numpy 1.x, so it runs in its
own conda environment. It exchanges data with the main suite only through files
in `Experiments/work/`: it reads `export/*.csv.gz` and writes `.npz` feature files.

## Setup

```bash
conda env create -f Experiments/external/rxnfp/environment.yml
conda activate rxnfp
python -c "import rxnfp, pkg_resources; print(pkg_resources.resource_filename('rxnfp', 'models/transformers'))"
```

The PyPI wheel `rxnfp==0.1.0` (≈75 MB) ships two of the four encoders used
here. The other two must be downloaded manually from
<https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers>
into the folder printed above (exp.md §4 has the full provenance):

| fingerprint | folder | ships in the wheel? | trained on | used by |
| --- | --- | --- | --- | --- |
| `rxnfp_mlm` | `bert_pretrained/` | yes | masked-LM only, no class labels | Track A/B, clustering (label-free) |
| `rxnfp` | `bert_ft/` | yes | `bert_pretrained` fine-tuned on Pistachio reaction classes: the published RXNFP fingerprint | Track A/B |
| `rxnfp_tpl` | `bert_class_1k_tpl/` | **no, download** | `bert_pretrained` fine-tuned on the TPL template classes (our TPL train split) | Track B, tpl_ub/tpl_b only |
| `rxnfp_schneider` | `bert_ft_10k_25s/` | **no, download** | fine-tuned on the 10k Schneider training reactions (our Schneider train split) | Track B, schneider_ub/schneider_b only |

## Run

After `01_data/prepare_data.py` (and `10_enzyme_activity/activity.py
--export-only` for the activity study), from the repository root:

```bash
conda run -n rxnfp python Experiments/external/rxnfp/rxnfp_embed.py
```

Writes `work/features/<rxn_key>/<fingerprint>.npz` (256-d, one row per
reaction) for every fingerprint in the table above; `rxnfp_tpl`/
`rxnfp_schneider` are only ever computed for their own reaction table (never
used outside it: USPTO 1k TPL is a superset of USPTO-50k/Schneider, so their
test reactions may already be in its training data). Runs on CPU; a GPU is
faster. `--fingerprints`/`--datasets` restrict the run; missing encoder
folders are skipped with a warning rather than failing the whole run. Existing
outputs are skipped, so interrupted runs can be restarted.

```bash
conda run -n rxnfp python Experiments/external/rxnfp/rxnfp_embed.py --check-weights
```

Compares `bert_ft` and `bert_class_1k_tpl`'s `embeddings.word_embeddings.weight`
(exp.md §4). Run this once `bert_class_1k_tpl` is downloaded, **before**
trusting `rxnfp` as a fair (non-TPL-trained) row: if the weights are
identical, `bert_ft` was trained on TPL labels too, and `rxnfp_mlm` should be
used instead as the fair label-free comparison.

## Caveat: `rxnfp` (`bert_ft`) is not label-free

`bert_ft` has seen Pistachio (NameRxn) class labels, which overlap the
Schneider, TPL and USPTO label sets. The published RXNFP papers use the same
encoder, so the numbers are comparable, but `rxnfp` is **not label-free**.
Say so in the paper; `rxnfp_mlm` is the label-free control.
