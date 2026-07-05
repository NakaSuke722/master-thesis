# src/data_loader.py
import os
import json
import pandas as pd

def load_timeseries_data(
        dataset: str,
        fault_type: str, 
        run_id: int, 
        strategy: str = "default"):
    """
    指定された前処理戦略(strategy)に従ってデータを読み込む。
    """
    # 提案された階層構造に合わせてパスを結合
    base_path = os.path.join("data", "processed", strategy)
    target_dir = os.path.join(base_path, dataset, fault_type, str(run_id))
    
    normal_path = os.path.join(target_dir, "normal_data.csv")
    abnormal_path = os.path.join(target_dir, "abnormal_data.csv")
    info_path = os.path.join(target_dir, "graph_info.json")
    
    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"Processed data not found for strategy '{strategy}' at: {target_dir}")
        
    df_normal = pd.read_csv(normal_path)
    df_abnormal = pd.read_csv(abnormal_path)
    
    with open(info_path, "r") as f:
        graph_info = json.load(f)
        
    df_full = pd.concat([df_normal, df_abnormal], ignore_index=True)
    ground_truth = fault_type
    
    return df_full, ground_truth, graph_info