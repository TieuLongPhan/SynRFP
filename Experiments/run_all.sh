#!/usr/bin/env bash
# Run the full SynRFP experiment suite in order. Every step caches its outputs
# and skips finished runs, so the script can be re-run after an interruption.
#   bash Experiments/run_all.sh [--config path/to/config.yaml]
set -euo pipefail
cd "$(dirname "$0")"
CFG=("$@")
PY=${PYTHON:-python}   # e.g. PYTHON="conda run -n synrfp-bench python"

$PY 01_data/prepare_data.py   "${CFG[@]}"
$PY 01_data/audit_data.py     "${CFG[@]}"
$PY 10_enzyme_activity/activity.py --export-only "${CFG[@]}"

# RXNFP runs in its own environment and reads the CSV exports written above
# (see external/rxnfp/environment.yml). Run it here, e.g. on a GPU machine:
#   conda run -n rxnfp python external/rxnfp/rxnfp_embed.py    "${CFG[@]}"
#   conda run -n rxnfp python external/rxnfp/rxnfp_finetune.py "${CFG[@]}"
# Steps below skip RXNFP rows (with a warning) until the embeddings exist.

$PY 02_featurize/featurize.py --study all "${CFG[@]}"
$PY 03_classification/classify.py --study benchmark "${CFG[@]}"
$PY 03_classification/classify.py --study factorial "${CFG[@]}"
$PY 04_factorial/anova.py "${CFG[@]}"
$PY 05_clustering/cluster.py "${CFG[@]}"
$PY 06_retrieval/retrieval.py "${CFG[@]}"
$PY 07_robustness/robustness.py "${CFG[@]}"
$PY 08_reaction_center/reaction_center.py "${CFG[@]}"
$PY 10_enzyme_activity/activity.py "${CFG[@]}"
$PY 09_report/report.py "${CFG[@]}"
