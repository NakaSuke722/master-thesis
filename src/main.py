# src/main.py
import argparse
import yaml
import json
import os
import time
from data_loader import load_timeseries_data
from evaluation import evaluate_ranking

# 比較するすべてのモデルをインポートしておく
from models.dummy import run_random_rca
# 将来的に新しいモデルを追加した場合はここに追記する
# from models.lingam_model import run_lingam_rca

def run_experiment(dataset, fault, run, batch=False):
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
        variables = list(df.columns)
        predicted_ranking = run_random_rca(variables, seed=current_seed)
        
    elif target_model == "lingam":
        # 今後実装する際、以下のように graph_info から境界線のインデックスを取り出してモデルに渡す
        # t_f_index = graph_info["t_f_index"]
        # predicted_ranking = run_lingam_rca(df, t_f_index, seed=current_seed)
        pass # 現段階では未実装のため仮置き
        
    else:
        # YAMLに不正な値が入力された場合はエラーで停止させる
        raise ValueError(f"Unknown model target in config: {target_model}")

    metrics = evaluate_ranking(predicted_ranking, ground_truth, k_values)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    print(f" - {fault} (Run {run}) : {execution_time} sec")
    
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