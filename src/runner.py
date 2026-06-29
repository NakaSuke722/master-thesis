# src/runner.py
import os
from main import run_experiment

def run_all():
    base_data_dir = "data/raw"
    datasets = ["online_boutique", "sock_shop", "train_ticket"]
    runs = [1, 2, 3, 4, 5]

    for dataset in datasets:
        dataset_path = os.path.join(base_data_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        for fault_dir in os.listdir(dataset_path):
            fault_path = os.path.join(dataset_path, fault_dir)
            if not os.path.isdir(fault_path):
                continue
            
            fault_type = fault_dir
            for run in runs:
                file_path = os.path.join(fault_path, str(run), "simple_data.csv")
                if os.path.isfile(file_path):
                    # main.pyの関数を直接呼び出し、一括実行フラグ(batch=True)を渡す
                    run_experiment(dataset, fault_type, run, batch=True)

if __name__ == "__main__":
    run_all()