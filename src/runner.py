# src/runner.py

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time

from experiments.config import load_config, resolve_granularity
from main import run_experiment
from utils.slack_notify import maybe_notify_slack


def run_all(
    config: dict,
    config_path: str,
    granularity: str,
) -> list[str]:
    """設定された全データセット・障害ケースを実行する。"""

    target_model = config["model"]["target"]
    datasets = config.get("datasets", [])

    print(
        f"=== Running experiments with model: "
        f"{target_model} ==="
    )

    base_data_dir = config["paths"].get(
        "raw_data_dir",
        "data/raw",
    )

    runs = [1, 2, 3, 4, 5]
    generated_result_files: list[str] = []

    for dataset in datasets:
        dataset_path = os.path.join(
            base_data_dir,
            dataset,
        )

        if not os.path.isdir(dataset_path):
            print(
                f"Warning: Dataset directory not found: "
                f"{dataset_path}"
            )
            continue

        cases: list[tuple[str, int]] = []

        for fault_dir in sorted(os.listdir(dataset_path)):
            fault_path = os.path.join(
                dataset_path,
                fault_dir,
            )

            if not os.path.isdir(fault_path):
                continue

            for run in runs:
                file_path = os.path.join(
                    fault_path,
                    str(run),
                    "simple_data.csv",
                )

                if os.path.isfile(file_path):
                    cases.append((fault_dir, run))

        total = len(cases)

        for progress, (fault_type, run) in enumerate(
            cases,
            start=1,
        ):
            _, output_file = run_experiment(
                dataset,
                fault_type,
                run,
                config=config,
                config_path=config_path,
                granularity=granularity,
                batch=True,
                progress=progress,
                total_progress=total,
            )

            generated_result_files.append(output_file)

    return generated_result_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all RCA experiments"
    )

    parser.add_argument(
        "--config",
        default="configs/amber.yaml",
    )

    parser.add_argument(
        "--granularity",
        choices=["service", "metric"],
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    granularity = resolve_granularity(
        config,
        args.granularity,
    )

    command_label = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote("src/runner.py"),
            *(
                shlex.quote(arg)
                for arg in sys.argv[1:]
            ),
        ]
    )

    start_epoch = time.time()

    status = "completed"
    reason = ""
    generated_result_files: list[str] = []

    try:
        generated_result_files = run_all(
            config=config,
            config_path=args.config,
            granularity=granularity,
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
            webhook_url=os.environ.get(
                "SLACK_WEBHOOK_URL",
                "",
            ),
            mention_user_id=os.environ.get(
                "SLACK_MENTION_USER_ID",
                "",
            ),
            command=command_label,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            exit_code=(
                0
                if status == "completed"
                else (
                    130
                    if status == "interrupted"
                    else 1
                )
            ),
            status=status,
            reason=reason,
            result_files=generated_result_files,
        )


if __name__ == "__main__":
    main()