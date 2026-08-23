#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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
    python3 src/runner.py \
        --config "${CONFIG}" \
        --granularity "${GRANULARITY}" \
        --workers "${WORKERS}" \
        --defer-success-notification
else
    python3 src/runner.py \
        --config "${CONFIG}" \
        --workers "${WORKERS}" \
        --defer-success-notification
fi

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

if [[ -n "${GRANULARITY}" ]]; then
    python3 src/aggregate_results.py \
        --config "${CONFIG}" \
        --granularity "${GRANULARITY}" \
        --total-time "${ELAPSED_TIME}"
else
    python3 src/aggregate_results.py \
        --config "${CONFIG}" \
        --total-time "${ELAPSED_TIME}"
fi
