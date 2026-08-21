#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

case "${1:-}" in
    ""|--rcaeval)
        # RCAEval RE1 Zenodo v2正式アブレーション（375ケース × 3 variants）。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/no_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/no_bayes.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/no_ar_no_bayes.yaml"
        )
        GRANULARITY="service"
        ;;
    --baro)
        # 既存BARO pilotの再現用設定は専用ディレクトリから実行する。
        CONFIGS=(
            "configs/ablation/baro/no_ar.yaml"
            "configs/ablation/baro/no_bayes.yaml"
            "configs/ablation/baro/no_ar_no_bayes.yaml"
        )
        GRANULARITY="metric"
        ;;
    *)
        echo "Usage: $0 [--rcaeval|--baro]" >&2
        exit 2
        ;;
esac

for CONFIG in "${CONFIGS[@]}"; do
    if [[ ! -f "${CONFIG}" ]]; then
        echo "Config file not found: ${CONFIG}" >&2
        exit 1
    fi
done

for CONFIG in "${CONFIGS[@]}"; do
    echo
    echo "========================================"
    echo "Running ablation: ${CONFIG}"
    echo "========================================"

    zsh scripts/run_main.sh "${GRANULARITY}" "${CONFIG}"
done

python3 src/aggregate_results.py \
    --configs "${CONFIGS[@]}" \
    --granularity "${GRANULARITY}"
