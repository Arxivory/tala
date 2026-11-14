
"""Historical, recording-aware movement baseline calculations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from datetime import datetime, timezone
from typing import Any

from tala.baselines.schemas import (
    BaselineComparison,
    SessionMetrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "results" / "baselines" / "session_history.jsonl"
)


def _finite_positive(values: list[Any]) -> list[float]:
    result = []

    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number) and number > 0:
            result.append(number)

    return result


def extract_session_metrics(
    movement_report: dict[str, Any],
) -> SessionMetrics:
    """Extract comparable duration metrics from a movement report."""

    exercise = str(movement_report.get("exercise", "")).strip()

    if not exercise:
        raise ValueError("Movement report has no exercise identifier.")

    repetition_data = movement_report.get("repetitions") or {}
    segmentation = movement_report.get("segmentation") or {}

    repetitions = repetition_data.get("repetitions") or []
    segments = segmentation.get("segments") or []

    repetition_durations = _finite_positive([
        item.get("duration_seconds")
        for item in repetitions
        if isinstance(item, dict)
    ])

    ascent_durations = _finite_positive([
        item.get("duration_seconds")
        for item in segments
        if isinstance(item, dict)
        and item.get("phase") == "ascent"
    ])

    descent_durations = _finite_positive([
        item.get("duration_seconds")
        for item in segments
        if isinstance(item, dict)
        and item.get("phase") == "descent"
    ])

    metrics: dict[str, float] = {}

    if repetition_durations:
        metrics["median_repetition_duration_seconds"] = float(
            median(repetition_durations)
        )

    if ascent_durations:
        metrics["median_ascent_duration_seconds"] = float(
            median(ascent_durations)
        )

    if descent_durations:
        metrics["median_descent_duration_seconds"] = float(
            median(descent_durations)
        )

    return SessionMetrics(
        exercise=exercise,
        metrics=metrics,
    )


def _load_history(
    history_path: Path,
) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []

    records = []

    with history_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def compare_with_baseline(
    movement_report: dict[str, Any],
    *,
    history_path: Path = DEFAULT_HISTORY_PATH,
    session_id: str,
    recording_fingerprint: str | None = None,
    baseline_scope: str | None = None,
) -> BaselineComparison:
    """Compare current recording against previous verified recordings.

    Legacy records without recording fingerprints are deliberately excluded.
    The current recording is never included in its own reference baseline.
    """

    current = extract_session_metrics(movement_report)
    history = _load_history(history_path)

    previous_sessions = [
        record
        for record in history
        if record.get("recording_fingerprint")
        and record.get("session_id") != session_id
        and record.get("recording_fingerprint") != recording_fingerprint
        and record.get("exercise") == current.exercise
        and record.get("baseline_scope") == baseline_scope
    ]

    comparisons: dict[str, dict[str, Any]] = {}

    for metric_name, current_value in current.metrics.items():
        reference_values = []

        for record in previous_sessions:
            historical_metrics = record.get("metrics") or {}
            value = historical_metrics.get(metric_name)

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(value) and value > 0:
                reference_values.append(value)

        if not reference_values:
            comparisons[metric_name] = {
                "status": "unavailable",
                "current_value": current_value,
                "reference_median": None,
                "difference": None,
                "difference_percent": None,
                "reference_session_count": 0,
            }
            continue

        reference_median = float(median(reference_values))
        difference = current_value - reference_median

        difference_percent = (
            (difference / reference_median) * 100
            if reference_median != 0
            else None
        )

        comparisons[metric_name] = {
            "status": "available",
            "current_value": current_value,
            "reference_median": reference_median,
            "difference": difference,
            "difference_percent": difference_percent,
            "reference_session_count": len(reference_values),
        }

    available = any(
        item["status"] == "available"
        for item in comparisons.values()
    )

    return BaselineComparison(
        exercise=current.exercise,
        status="available" if available else "unavailable",
        comparison_method="historical_session_median",
        metrics=comparisons,
        limitations=(
            "Comparisons use previously recorded, fingerprinted sessions from the same exercise and baseline scope.",
            "Legacy history entries without recording fingerprints are excluded.",
            "A historical median is descriptive and is not a clinical or population reference range.",
            "Differences do not independently establish improvement, deterioration, fatigue, or injury.",
            "Video quality, reconstruction quality, exercise execution, and recording duration may affect comparability.",
            "No comparison is available until prior compatible recording data exists.",
        ),
    )


def record_session(
    movement_report: dict[str, Any],
    *,
    session_id: str,
    recording_fingerprint: str | None = None,
    history_path: Path = DEFAULT_HISTORY_PATH,
    baseline_scope: str | None = None,
) -> None:
    """Append a recording once, after its baseline comparison."""

    session = extract_session_metrics(movement_report)

    if not session.metrics:
        return

    if not recording_fingerprint:
        raise ValueError(
            "A recording fingerprint is required to record a baseline session."
        )

    history_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_history(history_path)

    # Prevent duplicate processing of the same recording.
    if any(
        item.get("recording_fingerprint") == recording_fingerprint
        and item.get("baseline_scope") == baseline_scope
        and item.get("exercise") == session.exercise
        for item in existing
    ):
        return

    record = {
        "session_id": session_id,
        "recording_fingerprint": recording_fingerprint,
        "baseline_scope": baseline_scope,
        "exercise": session.exercise,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "metrics": session.metrics,
    }

    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
