from __future__ import annotations

from .aggregation import canonical_metric_name


FAULT_SUFFIXES = ("cpu", "mem", "memory", "loss", "delay")


def split_fault_label(fault: str) -> tuple[str, str]:
    """Split labels such as catalogue_delay into (catalogue, delay)."""
    for suffix in FAULT_SUFFIXES:
        token = f"_{suffix}"
        if fault.endswith(token):
            return fault[: -len(token)], suffix

    raise ValueError(
        f"Cannot infer fault type from '{fault}'. "
        f"Expected one of: {FAULT_SUFFIXES}"
    )


def make_evaluation_ground_truth(
    fault: str,
    granularity: str,
    mapping: dict,
) -> str | list[str]:
    """Construct service- or metric-level evaluation ground truth."""
    service, fault_type = split_fault_label(fault)

    if granularity == "service":
        return service

    if granularity != "metric":
        raise ValueError(
            f"granularity must be 'service' or 'metric', got {granularity}"
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
