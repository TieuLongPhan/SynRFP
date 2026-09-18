#!/usr/bin/env bash
# Repository lint: file-size policy, docstring style and Flake8.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"

if [[ "$#" -gt 0 ]]; then
  paths=("$@")
else
  paths=(synrfp tests scripts Experiments)
fi

"${PYTHON_BIN}" scripts/check_python_file_size.py
"${PYTHON_BIN}" scripts/check_docstring_style.py
"${PYTHON_BIN}" -m flake8 "${paths[@]}" \
  --count \
  --max-complexity=13 \
  --max-line-length=120 \
  --extend-ignore=E203 \
  --per-file-ignores="\
__init__.py:F401,F403,\
Experiments/*/*.py:E402,\
synrfp/tokenizers/map.py:C901" \
  --exclude="\
venv,\
.venv,\
__pycache__,\
.git,\
.pytest_cache,\
doc,\
docs,\
Experiments/work" \
  --statistics
