#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${AMBER_PYTHON:-}" ]]; then
    PYTHON_BIN="${AMBER_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

SELECTOR="${1:---all}"
WORKERS="${2:-${AMBER_WORKERS:-1}}"

if [[ ! "${WORKERS}" =~ '^[1-9][0-9]*$' ]]; then
    echo "workers must be a positive integer: ${WORKERS}" >&2
    exit 2
fi

case "${SELECTOR}" in
    --all)
        CONFIGS=(
            configs/baselines/epsilon_diagnosis.yaml
            configs/baselines/rcd.yaml
            configs/baselines/circa.yaml
            configs/baselines/run.yaml
        )
        ;;
    --epsilon-diagnosis)
        CONFIGS=(configs/baselines/epsilon_diagnosis.yaml)
        ;;
    --rcd)
        CONFIGS=(configs/baselines/rcd.yaml)
        ;;
    --circa)
        CONFIGS=(configs/baselines/circa.yaml)
        ;;
    --run)
        CONFIGS=(configs/baselines/run.yaml)
        ;;
    --baro)
        CONFIGS=(configs/baselines/baro.yaml)
        ;;
    *)
        echo "Usage: $0 [--all|--epsilon-diagnosis|--rcd|--circa|--run|--baro] [workers]" >&2
        exit 2
        ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for CONFIG in "${CONFIGS[@]}"; do
    echo
    echo "========================================"
    echo "Running baseline: ${CONFIG}"
    echo "========================================"
    START_TIME=$(date +%s)
    "${PYTHON_BIN}" src/runner.py \
        --config "${CONFIG}" \
        --workers "${WORKERS}" \
        --resume \
        --defer-success-notification
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    "${PYTHON_BIN}" src/aggregate_results.py \
        --config "${CONFIG}" \
        --total-time "${ELAPSED_TIME}"
done
