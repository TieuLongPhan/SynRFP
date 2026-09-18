# RXNFP baseline (separate environment)

RXNFP pins Python 3.8, an old `transformers` and numpy 1.x, so it runs in its
own conda environment. It exchanges data with the main suite only through files
in `Experiments/work/`: it reads `export/*.csv.gz` and `splits/*.npy` and writes
`.npz` feature files.

## Setup

```bash
conda env create -f Experiments/external/rxnfp/environment.yml
conda activate rxnfp
python -c "import rxnfp, pkg_resources; print(pkg_resources.resource_filename('rxnfp', 'models/transformers'))"
```

**No separate model download is needed.** The PyPI wheel `rxnfp==0.1.0`
(≈75 MB) ships the BERT weights. The command above prints their location:

| folder | what it is | used by |
| --- | --- | --- |
| `bert_pretrained/` | masked-LM pre-training only, no class labels | start point of `rxnfp_finetune.py`; `rxnfp_embed.py --model bert_pretrained` |
| `bert_ft/` | `bert_pretrained` fine-tuned on Pistachio reaction classes: the published RXNFP fingerprint | `rxnfp_embed.py` (default) |
| `bert_ft_10k_25s/` | a further fine-tuned variant (see the RXNFP repository) | not used |

If the weights are missing (for example, after installing from a source
checkout without the model files), download them from the upstream repository
at <https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers>
and place them in the folder printed above.

## Run

After `01_data/prepare_data.py` (and `10_enzyme_activity/activity.py --export`
for the activity study), from the repository root:

```bash
conda run -n rxnfp python Experiments/external/rxnfp/rxnfp_embed.py              # all reaction tables
conda run -n rxnfp python Experiments/external/rxnfp/rxnfp_finetune.py --schemes published   # GPU
```

- `rxnfp_embed.py` → `work/features/<rxn_key>/rxnfp_pretrained.npz` (256-d, one
  row per reaction). Runs on CPU; a GPU is faster.
- `rxnfp_finetune.py` fine-tunes `bert_pretrained` once per
  (dataset, scheme, seed) on that split's training records only, then writes
  `work/features/rxnfp_ft/<dataset>/<scheme>_s<seed>.npz`. A GPU is strongly
  recommended. Restrict the cost with `--datasets`, `--schemes` and `--seeds`.

Both scripts skip outputs that already exist, so interrupted runs can be restarted.

## Caveat: `rxnfp_pretrained` uses the fine-tuned `bert_ft` by default

By default, `rxnfp_embed.py` uses the published RXNFP encoder, `bert_ft`. That
encoder has seen Pistachio (NameRxn) class labels, which overlap the Schneider,
TPL and USPTO label sets. The published RXNFP papers use the same encoder, so
the numbers are comparable, but they are **not label-free**. Say so in the
paper, or also report `--model bert_pretrained` as a label-free control.
