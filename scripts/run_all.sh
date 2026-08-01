#!/bin/zsh
# scripts/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GRANULARITY="${1:-}"

if [[ -z "${GRANULARITY}" ]]; then
  GRANULARITY="$(
    python3 - <<'PY'
import yaml
with open("configs/default_params.yaml", "r") as f:
    config = yaml.safe_load(f)
print(config.get("evaluation", {}).get("granularity", "service"))
PY
  )"
fi

if [[ "${GRANULARITY}" != "service" && "${GRANULARITY}" != "metric" ]]; then
  echo "Usage: ./scripts/run_all.sh [service|metric]" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RCA_GRANULARITY="${GRANULARITY}"
export SIMULATION_COMMAND="./scripts/run_all.sh ${GRANULARITY}"

echo "=== Granularity: ${RCA_GRANULARITY} ==="

START_TIME=$(date +%s)
python3 src/runner.py

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

python3 src/aggregate_results.py   --total-time "${ELAPSED_TIME}"