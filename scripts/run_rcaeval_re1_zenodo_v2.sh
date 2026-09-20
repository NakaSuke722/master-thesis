#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd
)"

PROJECT_ROOT="$(
    cd "${SCRIPT_DIR}/.."
    pwd
)"

cd "${PROJECT_ROOT}"

CONFIG="configs/main/rcaeval_re1_zenodo_v2.yaml"

if [[ -n "${AMBER_PYTHON:-}" ]]; then
    PYTHON_BIN="${AMBER_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/download_rcaeval_re1.py

"${PYTHON_BIN}" src/prepare_rcaeval_re1.py \
    --config "${CONFIG}"

"${PYTHON_BIN}" -m pytest -q

bash scripts/run_main.sh \
    service \
    "${CONFIG}"
