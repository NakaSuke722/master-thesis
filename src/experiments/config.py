from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/amber.yaml")


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")

    return config


def resolve_granularity(
    config: dict,
    override: str | None = None,
) -> str:
    granularity = (
        override
        or config.get("evaluation", {}).get("granularity", "service")
    ).lower()

    if granularity not in {"service", "metric"}:
        raise ValueError(
            f"granularity must be 'service' or 'metric', got {granularity}"
        )

    return granularity