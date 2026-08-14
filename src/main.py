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


FAULT_SUFFIXES = ("cpu", "mem", "memory", "loss", "delay")


def split_fault_label(fault: str) -> tuple[str, str]:
    """例: catalogue_delay -> (catalogue, delay)。"""
    for suffix in FAULT_SUFFIXES:
        token = f"_{suffix}"
        if fault.endswith(token):
            return fault[:-len(token)], suffix
    raise ValueError(
        f"Cannot infer fault type from '{fault}'. "
        f"Expected one of: {FAULT_SUFFIXES}"
    )


def canonical_metric_name(metric: str) -> str:
    """パーセンタイル等を除き、service_metric_type へ正規化する。"""
    latency_suffixes = (
        "_latency-50", "_latency-90", "_latency-95", "_latency-99",
        "_latency",
    )
    for suffix in latency_suffixes:
        if metric.endswith(suffix):
            return metric[:-len(suffix)] + "_latency"

    if metric.endswith("_memory"):
        return metric[:-len("_memory")] + "_mem"

    return metric


def aggregate_canonical_metrics(
    metric_result,
    method: str = "max",
) -> list[str]:
    """raw metric scoresをcanonical metric単位に集約してランキングする。"""
    import numpy as np
    import pandas as pd

    if "metric" not in metric_result or "score" not in metric_result:
        raise ValueError("metric_result must contain metric and score columns")

    work = metric_result[["metric", "score"]].copy()
    work["canonical_metric"] = work["metric"].map(canonical_metric_name)

    rows = []
    for name, group in work.groupby("canonical_metric", sort=False):
        values = group["score"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            score = float("-inf")
        elif method == "max":
            score = float(np.max(values))
        elif method == "mean":
            score = float(np.mean(values))
        elif method == "logsumexp":
            m = float(np.max(values))
            score = m + float(np.log(np.sum(np.exp(values - m))))
        else:
            raise ValueError(f"Unknown metric aggregation method: {method}")
        rows.append((name, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in rows]


def make_evaluation_ground_truth(
    fault: str,
    granularity: str,
    mapping: dict,
) -> str | list[str]:
    service, fault_type = split_fault_label(fault)

    if granularity == "service":
        return service

    if granularity != "metric":
        raise ValueError(
            f"evaluation.granularity must be service or metric, got {granularity}"
        )

    metric_types = mapping.get(fault_type)
    if not metric_types:
        raise ValueError(
            f"No fine-grained mapping for fault type '{fault_type}'"
        )
    if isinstance(metric_types, str):
        metric_types = [metric_types]

    return [
        canonical_metric_name(f"{service}_{metric_type}")
        for metric_type in metric_types
    ]


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

    # default_params.yaml から評価設定とモデル名を取得
    k_values = config["evaluation"]["k_values"]
    target_model = config["model"]["target"]

    # run_all.sh の第1引数は RCA_GRANULARITY 経由で上書きする。
    granularity = os.environ.get(
        "RCA_GRANULARITY",
        config["evaluation"].get("granularity", "service"),
    ).lower()
    if granularity not in {"service", "metric"}:
        raise ValueError(
            f"RCA_GRANULARITY must be service or metric, got {granularity}"
        )

    # 1. YAMLから前処理の方法を取得（指定がない場合は "default" とする）
    strategy = config["model"].get("preprocess_strategy", "default")

    # 2. データローダーに戦略を渡す
    # graph_info は次に実装するモデル（LiNGAMなど）で異常発生時刻(t_F)を参照するために使用する
    df, ground_truth, graph_info = load_timeseries_data(dataset, fault, run, strategy)

    # 3. モデルの動的切り替え（条件分岐）
    current_seed = 42 + run
    evaluation_ground_truth = ground_truth

    if target_model == "dummy":
        from models.dummy import run_random_rca

        variables = list(df.columns)
        predicted_ranking = run_random_rca(variables, seed=current_seed)

    elif target_model == "nonlinear_anm":
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
        predicted_ranking = rca_model.fit_predict(df_normal, df_abnormal, dataset_name=dataset)
    
    elif target_model == "bayesian_rca":
        from models.bayesian_rca import BayesianRCA
        import pandas as pd
        
        target_dir = os.path.join("data/processed", strategy, dataset, fault, str(run))
        df_normal = pd.read_csv(os.path.join(target_dir, "normal_data.csv"))
        df_abnormal = pd.read_csv(os.path.join(target_dir, "abnormal_data.csv"))
        
        # モデルの初期化と実行
        # 新理論（ARモデル＋経験ベイズ法）のハイパーパラメータを指定
        rca_model = BayesianRCA(ar_lags=3, init_window=5, eta=5.0)
        predicted_ranking = rca_model.fit_predict(df_normal, df_abnormal, dataset_name=dataset) 

    elif target_model == "amber":
        import pandas as pd

        from models.amber import AMBER

        target_dir = os.path.join(
            "data", "processed", strategy, dataset, fault, str(run)
        )
        df_normal = pd.read_csv(
            os.path.join(target_dir, "normal_data.csv")
        )
        df_abnormal = pd.read_csv(
            os.path.join(target_dir, "abnormal_data.csv")
        )

        service_method = config["evaluation"].get(
            "service_aggregation", {}
        ).get("method", "mean_top3")

        # metricモードでも、まずraw metricのスコア表を作る。
        # serviceモードのみモデル内部でサービス集約する。
        model_aggregate = (
            "service" if granularity == "service" else "metric"
        )
        rca_model = AMBER(
            ar_order=3,
            winsor_quantile=None,
            aggregate=model_aggregate,
            service_aggregation=service_method,
        )

        if granularity == "service":
            predicted_ranking = rca_model.predict(
                df_normal, df_abnormal
            )
        else:
            metric_result = rca_model.fit_predict(
                df_normal, df_abnormal
            )
            metric_method = config["evaluation"].get(
                "metric_aggregation", {}
            ).get("method", "max")
            predicted_ranking = aggregate_canonical_metrics(
                metric_result,
                method=metric_method,
            )

        evaluation_ground_truth = make_evaluation_ground_truth(
            ground_truth,
            granularity,
            config["evaluation"].get(
                "fine_grained_fault_to_metric", {}
            ),
        )

    else:
        raise ValueError(f"Unknown model target in config: {target_model}")


    metrics = evaluate_ranking(
        predicted_ranking,
        evaluation_ground_truth,
        k_values,
    )

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
        print(f"\n===== Evaluation Summary (Metrics for {dataset} | Model: {target_model}) =====")
        for k in k_values:
            print(f"    AC@{k}: {metrics[f'AC@{k}']}, Avg@{k}: {metrics[f'Avg@{k}']:.4f}")

    results = {
        "dataset": dataset,
        "fault_type": fault,
        "run_id": run,
        "model_used": target_model,
        "evaluation_granularity": granularity,
        "execution_time_sec": execution_time,
        "metrics": metrics,
        "predicted_top_5": predicted_ranking[:5],
        "ground_truth": ground_truth,
        "evaluation_ground_truth": evaluation_ground_truth,
    }
    
    output_dir = os.path.join(
        config["paths"]["results_dir"],
        target_model,
        granularity,
        dataset,
    )
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