#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/labhhc4/anaconda3/envs/synrfp/bin/python}"
cd "$ROOT"

"$PYTHON" Experiments/11_interpretation/inspect_existing.py
"$PYTHON" Experiments/11_interpretation/classification_audit.py
"$PYTHON" Experiments/11_interpretation/family_sensitivity.py
"$PYTHON" Experiments/11_interpretation/classification_progress.py
"$PYTHON" Experiments/11_interpretation/tpl_tail_eta.py
"$PYTHON" Experiments/11_interpretation/feature_validity_audit.py
"$PYTHON" Experiments/11_interpretation/clustering_stability_audit.py
"$PYTHON" Experiments/11_interpretation/paired_inference.py
"$PYTHON" Experiments/11_interpretation/claim_audit.py
"$PYTHON" Experiments/11_interpretation/representation_controls.py
"$PYTHON" Experiments/11_interpretation/cross_task_audit.py
"$PYTHON" Experiments/11_interpretation/protocol_audit.py
"$PYTHON" Experiments/11_interpretation/handoff_audit.py
"$PYTHON" Experiments/11_interpretation/artifact_integrity.py
"$PYTHON" Experiments/11_interpretation/live_status.py
