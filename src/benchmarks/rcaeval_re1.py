from __future__ import annotations

from pathlib import Path

from benchmarks.base import BenchmarkCase


BENCHMARK_NAME = "rcaeval_re1"

DATASET_DIRECTORIES = {
    "re1_ob": "RE1-OB",
    "re1_ss": "RE1-SS",
    "re1_tt": "RE1-TT",
}

EXPECTED_FAULTS = {
    "cpu",
    "mem",
    "disk",
    "delay",
    "loss",
}


def load_inject_time(
    source_path: Path,
) -> int:
    """case-localなinject_time.txtを読み込む。"""

    inject_path = (
        source_path / "inject_time.txt"
    )

    if not inject_path.is_file():
        raise FileNotFoundError(
            "inject_time.txt not found: "
            f"{inject_path}"
        )

    try:
        return int(
            inject_path
            .read_text(encoding="utf-8")
            .strip()
        )
    except ValueError as exc:
        raise ValueError(
            "Invalid inject_time.txt: "
            f"{inject_path}"
        ) from exc


def parse_fault_directory(
    directory_name: str,
) -> tuple[str, str]:
    """service_fault形式を分割する。"""

    if "_" not in directory_name:
        raise ValueError(
            "Invalid RCAEval fault directory: "
            f"{directory_name}"
        )

    root_cause_service, fault_type = (
        directory_name.rsplit("_", 1)
    )

    if not root_cause_service:
        raise ValueError(
            "Root cause service is empty: "
            f"{directory_name}"
        )

    fault_type = fault_type.lower()

    if fault_type not in EXPECTED_FAULTS:
        raise ValueError(
            "Unexpected RE1 fault type: "
            f"{directory_name}"
        )

    return (
        root_cause_service,
        fault_type,
    )


def discover_cases(
    raw_root: str | Path,
    datasets: list[str] | None = None,
) -> list[BenchmarkCase]:
    """Zenodo v2 RCAEval RE1を走査する。"""

    raw_root = Path(raw_root)

    selected_datasets = (
        datasets
        if datasets is not None
        else list(DATASET_DIRECTORIES)
    )

    unknown = (
        set(selected_datasets)
        - set(DATASET_DIRECTORIES)
    )

    if unknown:
        raise ValueError(
            "Unknown RCAEval RE1 datasets: "
            f"{sorted(unknown)}"
        )

    cases: list[BenchmarkCase] = []

    for dataset in selected_datasets:
        dataset_root = (
            raw_root
            / dataset
            / DATASET_DIRECTORIES[dataset]
        )

        if not dataset_root.is_dir():
            raise FileNotFoundError(
                "RCAEval dataset directory "
                f"not found: {dataset_root}"
            )

        for fault_dir in sorted(
            dataset_root.iterdir()
        ):
            if not fault_dir.is_dir():
                continue

            (
                root_cause_service,
                fault_type,
            ) = parse_fault_directory(
                fault_dir.name
            )

            run_dirs = sorted(
                (
                    path
                    for path in fault_dir.iterdir()
                    if (
                        path.is_dir()
                        and path.name.isdigit()
                    )
                ),
                key=lambda path: int(path.name),
            )

            for run_dir in run_dirs:
                data_path = (
                    run_dir / "data.csv"
                )

                if not data_path.is_file():
                    raise FileNotFoundError(
                        "data.csv not found: "
                        f"{data_path}"
                    )

                inject_time = (
                    load_inject_time(run_dir)
                )

                repetition = int(
                    run_dir.name
                )

                # パス区切りを含まない一意なIDにする。
                case_id = (
                    f"{dataset}__"
                    f"{fault_dir.name}__"
                    f"{repetition}"
                )

                cases.append(
                    BenchmarkCase(
                        benchmark=(
                            BENCHMARK_NAME
                        ),
                        dataset=dataset,
                        case_id=case_id,
                        root_cause_service=(
                            root_cause_service
                        ),
                        fault_type=(
                            fault_type
                        ),
                        inject_time=inject_time,
                        repetition=repetition,
                        root_cause_metrics=None,
                        source_path=run_dir,
                    )
                )

    return sorted(
        cases,
        key=lambda case: (
            case.dataset,
            case.case_id,
        ),
    )