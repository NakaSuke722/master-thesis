# src/main.py
import argparse
import json
import os
import shlex
import sys
import time
from functools import lru_cache

import yaml

from data_loader import load_timeseries_data
from evaluation import evaluate_ranking
from utils.slack_notify import maybe_notify_slack


@lru_cache(maxsize=None)
def get_dataset_progress_info(dataset):
    """データセット内の総実行数と各ケースの進捗番号を返す"""
    dataset_dir = os.path.join("data", "raw", dataset)
    if not os.path.isdir(dataset_dir):
        return {}, 0

    runs = [1, 2, 3, 4, 5]
    progress_map = {}
    progress = 0

    for fault_dir in sorted(os.listdir(dataset_dir)):
        fault_path = os.path.join(dataset_dir, fault_dir)
        if not os.path.isdir(fault_path):
            continue

        for run in runs:
            file_path = os.path.join(fault_path, str(run), "simple_data.csv")
            if os.path.isfile(file_path):
                progress += 1
                progress_map[(fault_dir, run)] = progress

    return progress_map, progress


def run_experiment(dataset, fault, run, batch=False, progress=None, total_progress=None):
    """1つのデータセット・障害ケースに対する推論と評価を実行する"""
    start_time = time.time()

    with open("configs/default_params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 重複していた設定読み込みを整理
    k_values = config["evaluation"]["k_values"]
    target_model = config["model"]["target"]

    # 1. YAMLから戦略名を取得（指定がない場合は "default" とする）
    strategy = config["model"].get("preprocess_strategy", "default")

    # 2. データローダーに戦略を渡す
    # graph_info は次に実装するモデル（LiNGAMなど）で異常発生時刻(t_F)を参照するために使用する
    df, ground_truth, graph_info = load_timeseries_data(dataset, fault, run, strategy)

    # 3. モデルの動的切り替え（条件分岐）
    current_seed = 42 + run

    if target_model == "dummy":
        from models.dummy import run_random_rca

        variables = list(df.columns)
        predicted_ranking = run_random_rca(variables, seed=current_seed)

    elif target_model == "nonlinear_anm":
        # 提案モデルの実行
        from models.inference import RCAInference
        from models.trainer import Phase1Trainer

        # ターゲットとなっている実験ディレクトリの特定
        target_dir = os.path.join("data/processed", strategy, dataset, fault, str(run))

        # A. フェーズ1: 正常データによる学習の実行
        trainer = Phase1Trainer(data_dir=target_dir, epochs=200, lr=1e-3)
        trained_system = trainer.train()

        # B. フェーズ2 & 3: 推論と不確実性ペナルティ付きスコアリング
        # gammaの不確実性割引強度は必要に応じて引数やYAMLから管理可能
        inference = RCAInference(processed_dir=target_dir, gamma=1.0, mc_samples=30)
        predicted_ranking = inference.compute_rca_scores(trained_system)

    elif target_model == "data_driven_rca":
        import pandas as pd

        from models.data_driven_rca import DataDrivenRCA

        target_dir = os.path.join("data/processed", strategy, dataset, fault, str(run))

        # グラフメタデータ(graph_info.json)は読み込まず、データのみを抽出
        df_normal = pd.read_csv(os.path.join(target_dir, "normal_data.csv"))
        df_abnormal = pd.read_csv(os.path.join(target_dir, "abnormal_data.csv"))

        # モデルの初期化と実行 (lambda_reg や epochs は要調整パラメータ)
        rca_model = DataDrivenRCA(lambda_reg=0.1, epochs=300, lr=0.01)
        predicted_ranking = rca_model.fit_predict(df_normal, df_abnormal)

    else:
        raise ValueError(f"Unknown model target in config: {target_model}")

    metrics = evaluate_ranking(predicted_ranking, ground_truth, k_values)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    if progress is None or total_progress is None:
        progress_map, total_progress = get_dataset_progress_info(dataset)
        progress = progress_map.get((fault, run))

    if progress is not None and total_progress:
        print(f" {dataset} - {fault} (Run {run}, Progress {progress}/{total_progress}) : {execution_time} sec")
    else:
        print(f" {dataset} - {fault} (Run {run}) : {execution_time} sec")

    if not batch:
        print(f"\n=== Evaluation Summary (Metrics for {dataset} | Model: {target_model}) ===")
        for k in k_values:
            print(f"    AC@{k}: {metrics[f'AC@{k}']}, Avg@{k}: {metrics[f'Avg@{k}']:.4f}")

    results = {
        "dataset": dataset,
        "fault_type": fault,
        "run_id": run,
        "model_used": target_model,  # どのモデルの結果かをJSONに記録
        "execution_time_sec": execution_time,
        "metrics": metrics,
        "predicted_top_5": predicted_ranking[:5],
        "ground_truth": ground_truth,
    }

    output_dir = os.path.join(config["paths"]["results_dir"], dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{fault}_run{run}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results, output_file


def main():
    """コマンドラインから単一実行された場合の入り口"""
    parser = argparse.ArgumentParser(description="Run RCA Evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fault", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--batch", action="store_true", help="Suppress detailed metrics output")
    args = parser.parse_args()

    command_label = " ".join(
        [shlex.quote(sys.executable), shlex.quote("src/main.py"), *(shlex.quote(arg) for arg in sys.argv[1:])]
    )
    start_epoch = time.time()
    status = "completed"
    reason = ""
    result_file = ""

    try:
        _, result_file = run_experiment(args.dataset, args.fault, args.run, args.batch)
    except KeyboardInterrupt:
        status = "interrupted"
        reason = "Interrupted by user (SIGINT)"
        raise
    except Exception as exc:
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end_epoch = time.time()
        maybe_notify_slack(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
            mention_user_id=os.environ.get("SLACK_MENTION_USER_ID", ""),
            command=command_label,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            exit_code=0 if status == "completed" else (130 if status == "interrupted" else 1),
            status=status,
            reason=reason,
            result_files=[result_file] if result_file else [],
        )


if __name__ == "__main__":
    main()

# online_boutiqueのcartservice_cpu (Run 1) のみを検証する場合
# python3 src/main.py --dataset online_boutique --fault cartservice_cpu --run 1