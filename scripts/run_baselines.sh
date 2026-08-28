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
CONFIG="${2:-configs/baselines/baro.yaml}"
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
        echo "Usage: $0 [service|metric] [config] [workers]" >&2
        exit 2
        ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

START_TIME=$(date +%s)

RUN_ARGS=(--config "${CONFIG}" --workers "${WORKERS}" --defer-success-notification)
AGGREGATE_ARGS=(--config "${CONFIG}")
if [[ -n "${GRANULARITY}" ]]; then
    RUN_ARGS+=(--granularity "${GRANULARITY}")
    AGGREGATE_ARGS+=(--granularity "${GRANULARITY}")
fi

"${PYTHON_BIN}" src/runner.py "${RUN_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

"${PYTHON_BIN}" src/aggregate_results.py \
    "${AGGREGATE_ARGS[@]}" \
    --total-time "${ELAPSED_TIME}"
