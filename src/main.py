# src/main.py
import argparse
import json
import os
import shlex
import sys
import time
from functools import lru_cache
from experiments.config import load_config, resolve_granularity
from experiments.paths import case_result_dir
from benchmarks.base import BenchmarkCase
from data_loader import (
    get_processed_case_dir,
    load_processed_case,
    load_timeseries_data,
    load_benchmark_processed_case,
)
from evaluation import (
    aggregate_canonical_metrics,
    evaluate_ranking,
    make_evaluation_ground_truth,
)
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


def run_experiment(
    dataset: str,
    fault: str,
    run: int,
    *,
    config: dict,
    config_path: str,
    granularity: str,
    benchmark_case: BenchmarkCase | None = None,
    batch: bool = False,
    progress: int | None = None,
    total_progress: int | None = None,
    ):    
    """1つのデータセット・障害ケースに対する推論と評価を実行する"""

    start_time = time.time()

    k_values = config["evaluation"]["k_values"]
    target_model = config["model"]["target"]

    # 1. YAMLから前処理の方法を取得（指定がない場合は "default" とする）
    strategy = config["model"].get(
        "preprocess_strategy",
        "default",
    )

    processed_root = config["paths"].get(
        "processed_data_dir",
        "data/processed",
    )

    ground_truth = (
        benchmark_case.root_cause_service
        if benchmark_case is not None
        else fault
    )

    # 3. モデルの動的切り替え（条件分岐）
    current_seed = 42 + run
    evaluation_ground_truth = ground_truth

    if target_model == "dummy":
        from models.dummy import run_random_rca

        df, _, _ = load_timeseries_data(
            dataset=dataset,
            fault_type=fault,
            run_id=run,
            strategy=strategy,
            processed_root=processed_root,
        )

        variables = list(df.columns)
        predicted_ranking = run_random_rca(
            variables,
            seed=current_seed,
        )

    elif target_model == "nonlinear_anm":
        from models.inference import RCAInference
        from models.trainer import Phase1Trainer

        target_dir = get_processed_case_dir(dataset=dataset, fault_type=fault, run_id=run, strategy=strategy, processed_root=processed_root,)

        trainer = Phase1Trainer(data_dir=str(target_dir), epochs=200, lr=1e-3,)
        trained_system = trainer.train()

        inference = RCAInference(processed_dir=str(target_dir), gamma=1.0, mc_samples=30,)
        predicted_ranking = inference.compute_rca_scores(trained_system)

    elif target_model == "data_driven_rca":
        from models.data_driven_rca import DataDrivenRCA

        df_normal, df_abnormal, _ = load_processed_case(dataset=dataset, fault_type=fault, run_id=run, strategy=strategy, processed_root=processed_root, load_graph_info=False,)

        rca_model = DataDrivenRCA(lambda_reg=0.1, epochs=300, lr=0.01,)

        predicted_ranking = rca_model.fit_predict(df_normal, df_abnormal, dataset_name=dataset,)
    
    elif target_model == "bayesian_rca":
        from models.bayesian_rca import BayesianRCA

        df_normal, df_abnormal, _ = load_processed_case(dataset=dataset, fault_type=fault, run_id=run, strategy=strategy, processed_root=processed_root, load_graph_info=False,)

        rca_model = BayesianRCA(ar_lags=3, init_window=5, eta=5.0,)

        predicted_ranking = rca_model.fit_predict(df_normal, df_abnormal, dataset_name=dataset,)

    elif target_model == "amber":
        import pandas as pd

        from models.amber import AMBER, NIG

        if benchmark_case is not None:
            df_normal, df_abnormal, _ = (
                load_benchmark_processed_case(
                    benchmark=(
                        benchmark_case.benchmark
                    ),
                    dataset=(
                        benchmark_case.dataset
                    ),
                    case_id=(
                        benchmark_case.case_id
                    ),
                    strategy=strategy,
                    processed_root=processed_root,
                )
            )
        else:
            df_normal, df_abnormal, _ = (
                load_processed_case(
                    dataset=dataset,
                    fault_type=fault,
                    run_id=run,
                    strategy=strategy,
                    processed_root=processed_root,
                    load_graph_info=False,
                )
            )

        service_method = config["evaluation"].get(
            "service_aggregation", {}
        ).get("method", "mean_top3")

        # metricモードでも、まずraw metricのスコア表を作る。
        # serviceモードのみモデル内部でサービス集約する。
        model_aggregate = (
            "service" if granularity == "service" else "metric"
        )
        params = config["model"].get("params", {})
        prior_params = params.get("prior", {})

        prior = NIG(
            m=float(prior_params.get("m", 0.0)),
            kappa=float(prior_params.get("kappa", 1e-3)),
            alpha=float(prior_params.get("alpha", 2.0)),
            beta=float(prior_params.get("beta", 1.0)),
        )

        rca_model = AMBER(
            ar_order=int(params.get("ar_order", 3)),
            ridge=float(params.get("ridge", 1e-3)),
            min_scale=float(params.get("min_scale", 1e-6)),
            relative_scale_floor=float(
                params.get("relative_scale_floor", 1e-3)
            ),
            winsor_quantile=params.get("winsor_quantile"),
            prior=prior,
            aggregate=model_aggregate,
            service_aggregation=service_method,
            residualization=params.get(
                "residualization",
                "ar",
            ),
            scoring=params.get(
                "scoring",
                "bayes_factor",
            ),
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

        if benchmark_case is not None:

            if granularity == "service":
                evaluation_ground_truth = (
                    benchmark_case.root_cause_service
                )

            elif (
                benchmark_case.root_cause_metrics
                is not None
            ):
                evaluation_ground_truth = list(
                    benchmark_case.root_cause_metrics
                )

            else:
                raise ValueError(
                    "Metric-level ground truth "
                    "is unavailable for "
                    f"{benchmark_case.benchmark}: "
                    f"{benchmark_case.case_id}"
                )

        else:
            evaluation_ground_truth = (
                make_evaluation_ground_truth(
                    ground_truth,
                    granularity,
                    config[
                        "evaluation"
                    ].get(
                        "fine_grained_fault_to_metric",
                        {},
                    ),
                )
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

    experiment = config.get("experiment", {})

    results = {
        "dataset": dataset,
        "fault_type": (
            benchmark_case.fault_type
            if benchmark_case is not None
            else fault
        ),

        "run_id": (
            benchmark_case.repetition
            if benchmark_case is not None
            else run
        ),
        "experiment_category": experiment.get("category", "main"),
        "experiment_name": experiment.get("name", target_model),

        "model_used": target_model,
        "model_parameters": config["model"].get("params", {}),

        "config_path": config_path,
        "evaluation_granularity": granularity,

        "execution_time_sec": execution_time,
        "metrics": metrics,
        "predicted_top_5": predicted_ranking[:5],
        "ground_truth": ground_truth,
        "evaluation_ground_truth": evaluation_ground_truth,
        "benchmark": (
            benchmark_case.benchmark
            if benchmark_case is not None
            else "baro_pilot"
        ),

        "case_id": (
            benchmark_case.case_id
            if benchmark_case is not None
            else None
        ),
    }

    # This is observational output only: AMBER's scoring and ranking are
    # completed above before diagnostics are copied into the result artifact.
    if target_model == "amber":
        results["amber_diagnostics"] = rca_model.diagnostics_

    output_dir = case_result_dir(
        config,
        granularity,
        dataset,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if benchmark_case is not None:
        output_file = (
            output_dir
            / f"{benchmark_case.case_id}.json"
        )
    else:
        output_file = (
            output_dir
            / f"{fault}_run{run}.json"
        )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, )

    return results, str(output_file)

def main():
    """コマンドラインから単一実行された場合の入り口"""
    parser = argparse.ArgumentParser(description="Run RCA Evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fault", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--batch", action="store_true", help="Suppress detailed metrics output")
    parser.add_argument("--config", default="configs/amber.yaml",)
    parser.add_argument("--granularity", choices=["service", "metric"], default=None,)
    args = parser.parse_args()

    config = load_config(args.config)
    granularity = resolve_granularity(
        config,
        args.granularity,
    )

    command_label = " ".join(
        [shlex.quote(sys.executable), shlex.quote("src/main.py"), *(shlex.quote(arg) for arg in sys.argv[1:])]
    )
    start_epoch = time.time()
    status = "completed"
    reason = ""
    result_file = ""

    try:
        _, result_file = run_experiment(
            args.dataset,
            args.fault,
            args.run,
            config=config,
            config_path=args.config,
            granularity=granularity,
            batch=args.batch,
        )    
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
