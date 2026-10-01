#!/usr/bin/env bash
# CPU steps of the validation suite, in order. Restartable: every step skips
# finished work. GPU and RXNFP steps are separate (see RUNBOOK.md, steps 4-5).
#   bash 12_validation/run_cpu.sh            # full run
#   VALIDATION_SMOKE=1 bash 12_validation/run_cpu.sh    # after 00_setup.py --smoke
set -euo pipefail
cd "$(dirname "$0")/.."                      # Experiments/
PY=${PYTHON:-python}
V=12_validation
ROOT=$V; [ -n "${VALIDATION_SMOKE:-}" ] && ROOT=$V/smoke
POLICIES=${POLICIES:-"raw n1 n2"}
DATASETS=${DATASETS:-}                        # empty = all configured datasets
DS_ARGS=(); [ -n "$DATASETS" ] && DS_ARGS=(--datasets $DATASETS)

$PY $V/01_input_audit.py "${DS_ARGS[@]}"
for P in $POLICIES; do
  C=$ROOT/configs/$P.yaml
  $PY 02_featurize/featurize.py --config $C --study benchmark "${DS_ARGS[@]}"
  $PY 03_classification/classify.py --config $C --study benchmark --models knn "${DS_ARGS[@]}"
done
$PY $V/02_attribution.py ${ATTRIBUTION_ARGS:-}
$PY $V/04_locality.py ${LOCALITY_ARGS:-}
$PY $V/03_retrieval_proxy.py --stage fetch
$PY $V/03_retrieval_proxy.py --stage build
$PY $V/03_retrieval_proxy.py --stage featurize
$PY $V/03_retrieval_proxy.py --stage score --part development
$PY $V/06_report.py
