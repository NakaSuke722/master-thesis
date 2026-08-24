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

GRANULARITY_ARG="${1:-}"

CONFIG="${2:-configs/main/rcaeval_re1_zenodo_v2.yaml}"
WORKERS="${3:-${AMBER_WORKERS:-1}}"

if [[ ! "${WORKERS}" =~ '^[1-9][0-9]*$' ]]; then
    echo "workers must be a positive integer: ${WORKERS}" >&2
    exit 2
fi

case "${GRANULARITY_ARG}" in
    "")
        GRANULARITY=""
        ;;
    service|-service|--service)
        GRANULARITY="service"
        ;;
    metric|-metric|--metric)
        GRANULARITY="metric"
        ;;
    *)
        echo \
            "Usage: $0 [service|metric] [config] [workers]" \
            >&2
        exit 2
        ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

START_TIME=$(date +%s)

if [[ -n "${GRANULARITY}" ]]; then
    "${PYTHON_BIN}" src/runner.py \
        --config "${CONFIG}" \
        --granularity "${GRANULARITY}" \
        --workers "${WORKERS}" \
        --defer-success-notification
else
    "${PYTHON_BIN}" src/runner.py \
        --config "${CONFIG}" \
        --workers "${WORKERS}" \
        --defer-success-notification
fi

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

if [[ -n "${GRANULARITY}" ]]; then
    "${PYTHON_BIN}" src/aggregate_results.py \
        --config "${CONFIG}" \
        --granularity "${GRANULARITY}" \
        --total-time "${ELAPSED_TIME}"
else
    "${PYTHON_BIN}" src/aggregate_results.py \
        --config "${CONFIG}" \
        --total-time "${ELAPSED_TIME}"
fi
