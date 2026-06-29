# src/main.py
import argparse
import yaml
import json
import os
import time
from data_loader import load_timeseries_data
from evaluation import evaluate_ranking
from models.dummy import run_random_rca

def run_experiment(dataset, fault, run, batch=False):
    """1つのデータセット・障害ケースに対する推論と評価を実行する"""
    start_time = time.time()
    
    with open("configs/default_params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    k_values = config["evaluation"]["k_values"]
    raw_path = config["paths"]["raw_data_dir"]

    df, ground_truth = load_timeseries_data(dataset, fault, run, raw_path)

    variables = list(df.columns)
    current_seed = 42 + run 
    predicted_ranking = run_random_rca(variables, seed=current_seed)

    metrics = evaluate_ranking(predicted_ranking, ground_truth, k_values)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    # ターミナル出力（指定のフォーマット）
    print(f" - {fault} (Run {run}) : {execution_time} sec")
    
    # 単一実行時のみ詳細を表示
    if not batch:
        print(f"  [Metrics for {dataset}]")
        for k in k_values:
            print(f"    AC@{k}: {metrics[f'AC@{k}']}, Avg@{k}: {metrics[f'Avg@{k}']:.4f}")

    results = {
        "dataset": dataset,
        "fault_type": fault,
        "run_id": run,
        "execution_time_sec": execution_time,
        "metrics": metrics,
        "predicted_top_5": predicted_ranking[:5],
        "ground_truth": ground_truth
    }
    
    output_dir = os.path.join(config["paths"]["results_dir"], dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{fault}_run{run}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def main():
    """コマンドラインから単一実行された場合の入り口"""
    parser = argparse.ArgumentParser(description="Run RCA Evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fault", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--batch", action="store_true", help="Suppress detailed metrics output")
    args = parser.parse_args()

    run_experiment(args.dataset, args.fault, args.run, args.batch)

if __name__ == "__main__":
    main()

# online_boutiqueのcartservice_cpu (Run 1) のみを検証する場合
# python3 src/main.py --dataset online_boutique --fault cartservice_cpu --run 1