#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CONFIGS=(
    "configs/ablation/no_ar.yaml"
    "configs/ablation/no_bayes.yaml"
    "configs/ablation/no_ar_no_bayes.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo
    echo "========================================"
    echo "Running ablation: ${CONFIG}"
    echo "========================================"

    zsh scripts/run_main.sh service "${CONFIG}"
    zsh scripts/run_main.sh metric "${CONFIG}"
done