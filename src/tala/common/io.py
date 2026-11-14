"""Standardized serialization helpers for TALA outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting and UTF-8 encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
    return output_path


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe_csv(dataframe: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame without its implicit index."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path
