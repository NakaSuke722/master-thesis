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

WORKERS="${AMBER_WORKERS:-1}"

case "${1:-}" in
    ""|--rcaeval)
        # RCAEval RE1 Zenodo v2正式アブレーション（375ケース × 12 variants）。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/no_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/no_bayes.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/no_ar_no_bayes.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/counterfactual_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_uncertainty.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_full_covariance.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/direct_ar_bayes_factor.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/intercept_shift_ar_bayes_factor.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_ar_bayes_factor.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/bsrc_ar_bayes_factor.yaml"
        )
        GRANULARITY="service"
        ;;
    --counterfactual-ar)
        # 新variantだけを実行し、取得済みの既存3 variantは再実行しない。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/counterfactual_ar.yaml"
        )
        GRANULARITY="service"
        ;;
    --ar-redesign)
        # 定常化、clipなし再帰、対角補正、完全共分散を順番に比較する。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_uncertainty.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_full_covariance.yaml"
        )
        GRANULARITY="service"
        ;;
    --full-covariance-ar)
        # 既存variantを再実行せず、完全forecast-error covarianceだけを実行する。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/stationary_counterfactual_ar_full_covariance.yaml"
        )
        GRANULARITY="service"
        ;;
    --direct-ar-bayes-factor)
        # shared AR対separate ARの直接Bayes Factorだけを実行する。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/direct_ar_bayes_factor.yaml"
        )
        GRANULARITY="service"
        ;;
    --intercept-shift-ar-bayes-factor)
        # AR係数と分散を共有し、intercept変化だけを検出する。
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/intercept_shift_ar_bayes_factor.yaml"
        )
        GRANULARITY="service"
        ;;
    --adaptive-direct-ar-bayes-factor)
        # 応答形状・遅延周辺化とnormal-only校正をまとめて検証する。
        WORKERS="${AMBER_WORKERS:-4}"
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_ar_bayes_factor.yaml"
        )
        GRANULARITY="service"
        ;;
    --adaptive-direct-rollback)
        # 統合candidateから1要素ずつ外すロールバック・アブレーション。
        WORKERS="${AMBER_WORKERS:-4}"
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_no_null_calibration.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_fixed_onset.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_step_only.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_no_step_ramp.yaml"
            "configs/ablation/rcaeval_re1_zenodo_v2/adaptive_direct_no_per_row_normalization.yaml"
        )
        GRANULARITY="service"
        ;;
    --bsrc-ar)
        # 正常posterior predictive対sparse parameter regime change。
        WORKERS="${AMBER_WORKERS:-4}"
        CONFIGS=(
            "configs/ablation/rcaeval_re1_zenodo_v2/bsrc_ar_bayes_factor.yaml"
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
        echo "Usage: $0 [--rcaeval|--counterfactual-ar|--ar-redesign|--full-covariance-ar|--direct-ar-bayes-factor|--intercept-shift-ar-bayes-factor|--adaptive-direct-ar-bayes-factor|--adaptive-direct-rollback|--bsrc-ar|--baro]" >&2
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

    zsh scripts/run_main.sh "${GRANULARITY}" "${CONFIG}" "${WORKERS}"
done

"${PYTHON_BIN}" src/aggregate_results.py \
    --configs "${CONFIGS[@]}" \
    --granularity "${GRANULARITY}"
