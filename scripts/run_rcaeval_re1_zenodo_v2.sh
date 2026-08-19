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

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 scripts/download_rcaeval_re1.py

python3 src/prepare_rcaeval_re1.py \
    --config "${CONFIG}"

python3 -m pytest -q

bash scripts/run_main.sh \
    service \
    "${CONFIG}"