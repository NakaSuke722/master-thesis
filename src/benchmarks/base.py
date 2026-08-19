from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    """1つのRCA benchmark caseを表す共通データ構造。"""

    benchmark: str
    dataset: str
    case_id: str

    root_cause_service: str
    fault_type: str | None
    inject_time: int | None

    repetition: int | None = None

    # Fine-grained benchmarkでのみ使用する。
    # RCAEval RE1ではNone。
    root_cause_metrics: tuple[str, ...] | None = None

    # raw dataを読む段階でのみ利用する。
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "root_cause_service": self.root_cause_service,
            "fault_type": self.fault_type,
            "inject_time": self.inject_time,
            "repetition": self.repetition,
            "root_cause_metrics": (
                list(self.root_cause_metrics)
                if self.root_cause_metrics is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "BenchmarkCase":
        root_cause_metrics = data.get(
            "root_cause_metrics"
        )

        return cls(
            benchmark=str(data["benchmark"]),
            dataset=str(data["dataset"]),
            case_id=str(data["case_id"]),
            root_cause_service=str(
                data["root_cause_service"]
            ),
            fault_type=data.get("fault_type"),
            inject_time=(
                int(data["inject_time"])
                if data.get("inject_time") is not None
                else None
            ),
            repetition=(
                int(data["repetition"])
                if data.get("repetition") is not None
                else None
            ),
            root_cause_metrics=(
                tuple(root_cause_metrics)
                if root_cause_metrics
                else None
            ),
            source_path=None,
        )