# src/runner.py
import os
import yaml
import shlex
import sys
import time
from main import run_experiment
from utils.slack_notify import maybe_notify_slack

def run_all():
    # 設定ファイルからモデル名を読み込む
    with open("configs/default_params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    target_model = config["model"]["target"]
    
    # ターミナル出力の先頭にモデル名を記載
    print(f"=== Running experiments with model: {target_model} ===")

    base_data_dir = "data/raw"
    datasets = ["online_boutique", "sock_shop", "train_ticket"]
    runs = [1, 2, 3, 4, 5]
    generated_result_files = []

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
                    _, output_file = run_experiment(dataset, fault_type, run, batch=True)
                    generated_result_files.append(output_file)

    return generated_result_files

if __name__ == "__main__":
    command_label = os.environ.get(
        "SIMULATION_COMMAND",
        " ".join([shlex.quote(sys.executable), shlex.quote("src/runner.py")]),
    )
    start_epoch = time.time()
    status = "completed"
    reason = ""
    generated_result_files = []

    try:
        generated_result_files = run_all()
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
            result_files=generated_result_files,
        )