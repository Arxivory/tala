
"""Schemas and constants for TALA's contextual assessment layer."""

from typing import Any, TypedDict


SORENESS_LOCATIONS = (
    "shoulders",
    "arms",
    "elbows",
    "lower_back",
    "hips",
    "thighs",
    "knees",
    "ankles",
)

SORENESS_LEVELS = {"none", "mild", "moderate", "severe"}
FATIGUE_LEVELS = {"none", "mild", "moderate", "high", "severe"}
SLEEP_QUALITIES = {"poor", "fair", "good"}
HYDRATION_NUTRITION_STATES = {"on_track", "behind"}

# The Phase 3.10 mapped_coordinates.csv stores angular coordinates in radians.
# Convert to degrees only for human-readable assessment output.
MOVEMENT_COORDINATES = {
    "hip": {
        "right": "hip_flexion_r",
        "left": "hip_flexion_l",
    },
    "knee": {
        "right": "knee_angle_r",
        "left": "knee_angle_l",
    },
    "ankle": {
        "right": "ankle_angle_r",
        "left": "ankle_angle_l",
    },
}

ANGLE_UNIT_IN_SOURCE = "radians"
ANGLE_UNIT_IN_OUTPUT = "degrees"

# Explicitly a configurable demonstration heuristic, not a clinical threshold.
DEMO_SHORT_SLEEP_HOURS = 6.0
DEMO_ELEVATED_STRESS_LEVEL = 4


class AssessmentInsight(TypedDict, total=False):
    category: str
    title: str
    observation: str
    evidence: list[dict[str, Any]]
    interpretation: str
    suggested_actions: list[str]
    when_to_seek_support: str
    limitations: list[str]


def normalize_choice(value: Any) -> Any:
    """Normalize UI labels while preserving unanswered values as None."""
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")

    if normalized in {"", "not_reported"}:
        return None

    return normalized


def normalize_daily_checkin(checkin: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize a partial daily check-in.

    Missing fields remain None. This function does not infer that an unanswered
    question means 'none', 'good', or 'on track'.
    """
    checkin = checkin or {}

    sleep = checkin.get("sleep") or {}
    soreness = checkin.get("soreness") or {}

    normalized_soreness = {
        location: normalize_choice(soreness.get(location))
        for location in SORENESS_LOCATIONS
    }

    result = {
        "sleep": {
            "duration_hours": sleep.get("duration_hours"),
            "quality": normalize_choice(sleep.get("quality")),
        },
        "fatigue": normalize_choice(checkin.get("fatigue")),
        "soreness": normalized_soreness,
        "stress_level": checkin.get("stress_level"),
        "hydration": normalize_choice(checkin.get("hydration")),
        "nutrition": normalize_choice(checkin.get("nutrition")),
        "pain_note": checkin.get("pain_note"),
    }

    return result


def validate_daily_checkin(checkin: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize and validate supplied values.

    Partial check-ins are accepted. Invalid supplied values raise ValueError.
    """
    result = normalize_daily_checkin(checkin)

    sleep_hours = result["sleep"]["duration_hours"]
    if sleep_hours is not None:
        try:
            sleep_hours = float(sleep_hours)
        except (TypeError, ValueError):
            raise ValueError("sleep.duration_hours must be numeric.")

        if not 0 <= sleep_hours <= 12:
            raise ValueError("sleep.duration_hours must be between 0 and 12.")

        result["sleep"]["duration_hours"] = sleep_hours

    quality = result["sleep"]["quality"]
    if quality not in SLEEP_QUALITIES and quality is not None:
        raise ValueError(f"Invalid sleep quality: {quality}")

    fatigue = result["fatigue"]
    if fatigue not in FATIGUE_LEVELS and fatigue is not None:
        raise ValueError(f"Invalid fatigue level: {fatigue}")

    for location, severity in result["soreness"].items():
        if severity not in SORENESS_LEVELS and severity is not None:
            raise ValueError(
                f"Invalid soreness level for {location}: {severity}"
            )

    stress = result["stress_level"]
    if stress is not None:
        try:
            stress = int(stress)
        except (TypeError, ValueError):
            raise ValueError("stress_level must be an integer from 1 to 5.")

        if not 1 <= stress <= 5:
            raise ValueError("stress_level must be between 1 and 5.")

        result["stress_level"] = stress

    for field in ("hydration", "nutrition"):
        value = result[field]
        if value not in HYDRATION_NUTRITION_STATES and value is not None:
            raise ValueError(f"Invalid {field} state: {value}")

    pain_note = result["pain_note"]
    if pain_note is not None:
        result["pain_note"] = str(pain_note).strip() or None

    return result
