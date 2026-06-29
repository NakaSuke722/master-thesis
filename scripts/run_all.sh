#!/bin/zsh
# scripts/run_all.sh

export PYTHONPATH="${PYTHONPATH}:./src"

START_TIME=$(date +%s)

# 1. Python内でループを実行（ここで高速に全ケースが処理される）
python3 src/runner.py

# 2. 経過時間の計算と集計スクリプトへの引き渡し
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

python3 src/aggregate_results.py --total-time ${ELAPSED_TIME}