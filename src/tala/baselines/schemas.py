"""Data contracts for dynamically derived movement baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SessionMetrics:
    """Descriptive metrics extracted from one movement-analysis session."""

    exercise: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineComparison:
    """Comparison against previously recorded sessions."""

    exercise: str
    status: str
    comparison_method: str
    metrics: dict[str, dict[str, Any]]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)