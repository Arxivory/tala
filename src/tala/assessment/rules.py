
"""Transparent rules for TALA's contextual assessment."""

from __future__ import annotations

from typing import Any
import math

import pandas as pd

from .schemas import (
    DEMO_ELEVATED_STRESS_LEVEL,
    DEMO_SHORT_SLEEP_HOURS,
    MOVEMENT_COORDINATES,
    validate_daily_checkin,
)


RAD_TO_DEG = 180.0 / math.pi


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _walk_report(data: Any, prefix: str = ""):
    """Yield flattened report paths and values for tolerant metric lookup."""
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key, value
            yield from _walk_report(value, path)

    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _walk_report(value, f"{prefix}[{index}]")


def _find_report_value(
    report: dict[str, Any],
    aliases: tuple[str, ...],
) -> tuple[str, Any] | None:
    """Find the first exact matching key in a nested report."""
    alias_set = {alias.lower() for alias in aliases}

    for path, key, value in _walk_report(report):
        if str(key).lower() in alias_set and not isinstance(value, (dict, list)):
            return path, value

    return None

def extract_movement_observations(
    mapped_coordinates: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Extract min/max/excursion and their timestamps for six lower-limb
    coordinates. Source angles are radians; output angles are degrees.

    These are estimated coordinate ranges, not clinically validated ROM.
    """
    if not isinstance(mapped_coordinates, pd.DataFrame):
        raise TypeError("mapped_coordinates must be a pandas DataFrame.")

    observations = []

    if "time" not in mapped_coordinates.columns:
        raise ValueError("mapped_coordinates is missing the 'time' column.")

    times = pd.to_numeric(mapped_coordinates["time"], errors="coerce")

    for joint, sides in MOVEMENT_COORDINATES.items():
        side_results = {}

        for side, column in sides.items():
            if column not in mapped_coordinates.columns:
                side_results[side] = {
                    "available": False,
                    "reason": f"Missing coordinate column: {column}",
                }
                continue

            values = pd.to_numeric(
                mapped_coordinates[column],
                errors="coerce",
            )

            valid = values.notna() & times.notna()
            valid_values = values[valid]
            valid_times = times[valid]

            if valid_values.empty:
                side_results[side] = {
                    "available": False,
                    "reason": "No finite coordinate samples available.",
                }
                continue

            minimum_index = valid_values.idxmin()
            maximum_index = valid_values.idxmax()

            minimum_rad = float(valid_values.loc[minimum_index])
            maximum_rad = float(valid_values.loc[maximum_index])

            minimum_deg = minimum_rad * RAD_TO_DEG
            maximum_deg = maximum_rad * RAD_TO_DEG

            side_results[side] = {
                "available": True,
                "source_column": column,
                "sample_count": int(valid_values.count()),
                "minimum_degrees": round(minimum_deg, 2),
                "maximum_degrees": round(maximum_deg, 2),
                "coordinate_excursion_degrees": round(
                    maximum_deg - minimum_deg, 2
                ),
                "time_at_minimum_seconds": round(
                    float(times.loc[minimum_index]), 3
                ),
                "time_at_maximum_seconds": round(
                    float(times.loc[maximum_index]), 3
                ),
                "source_unit": "radians",
                "display_unit": "degrees",
            }

        available_sides = [
            side for side, result in side_results.items()
            if result["available"]
        ]

        if available_sides:
            observations.append({
                "joint": joint,
                "title": f"Estimated {joint} coordinate movement",
                "available": True,
                "sides": side_results,
                "interpretation": (
                    "The values describe the estimated coordinate range "
                    "in this reconstructed movement. They are not a "
                    "clinical range-of-motion assessment."
                ),
                "limitations": [
                    "The estimates depend on monocular pose estimation "
                    "and approximate OpenSim marker fitting.",
                    "Coordinate excursion does not establish movement "
                    "quality, injury risk, or safety.",
                ],
            })
        else:
            observations.append({
                "joint": joint,
                "title": f"Estimated {joint} coordinate movement",
                "available": False,
                "sides": side_results,
                "interpretation": "No usable coordinate samples were available.",
                "limitations": [
                    "Missing movement data must not be interpreted as "
                    "normal or absent movement."
                ],
            })

    return observations


def extract_reconstruction_quality(
    ik_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Extract available reconstruction-quality information.

    Metric names are searched recursively so the assessment is not coupled
    to one exact nesting arrangement in ik_report.json.
    """
    if not isinstance(ik_report, dict):
        return {
            "status": "unavailable",
            "metrics": {},
            "limitations": ["The inverse-kinematics report was not provided."],
        }

    aliases = {
        "frame_count": ("frame_count",),
        "duration_seconds": ("duration_seconds", "duration"),
        "sampling_rate_hz": ("sampling_rate_hz",),
        "marker_sample_count": ("sample_count",),
        "mean_fitting_error_m": ("mean_error_m", "mean_fitting_error_m"),
        "rms_fitting_error_m": (
            "overall_rms_error_m",
            "rms_error_m",
            "overall_rms",
        ),
        "maximum_fitting_error_m": (
            "maximum_error_m",
            "max_error_m",
            "maximum_fitting_error_m",
        ),
        "observed_marker_samples": ("observed_samples",),
        "interpolated_marker_samples": ("interpolated_samples",),
        "unavailable_marker_samples": ("unavailable_samples",),
    }

    metrics = {}
    source_paths = {}

    for output_name, candidates in aliases.items():
        match = _find_report_value(ik_report, candidates)
        if match is None:
            continue

        path, raw_value = match
        number = _finite_number(raw_value)

        if number is not None:
            metrics[output_name] = number
            source_paths[output_name] = path

    limitations = []

    report_limitations = ik_report.get("limitations")
    if isinstance(report_limitations, list):
        limitations.extend(
            str(item) for item in report_limitations if item
        )

    unavailable = metrics.get("unavailable_marker_samples", 0)
    if unavailable > 0:
        limitations.append(
            f"{int(unavailable)} marker samples were reported unavailable."
        )

    if not metrics:
        status = "unavailable"
        limitations.append(
            "No recognized reconstruction-quality metrics were found."
        )
    elif limitations:
        status = "limited"
    else:
        status = "reported"

    return {
        "status": status,
        "metrics": metrics,
        "source_paths": source_paths,
        "limitations": list(dict.fromkeys(limitations)),
        "interpretation": (
            "Marker-fitting residuals describe agreement between the "
            "transformed pose targets and fitted model markers. They do "
            "not establish anatomical accuracy."
        ),
    }



def extract_gravity_context(gravity_report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract whole-body gravity metrics and individual movement phases."""

    if not isinstance(gravity_report, dict):
        return {
            "status": "unavailable",
            "method": None,
            "vertical_axis": None,
            "metrics": {},
            "environments": {},
            "interpretation": (
                "Gravity analysis data was not available."
            ),
            "limitations": [
                "No gravity analysis report was supplied."
            ],
        }

    metric_keys = [
        "total_body_mass_kg",
        "initial_com_height_m",
        "minimum_com_height_m",
        "maximum_com_height_m",
        "com_height_excursion_m",
    ]

    whole_body_metrics = {
        key: value
        for key in metric_keys
        if (value := _finite_number(gravity_report.get(key))) is not None
    }

    source_environments = gravity_report.get(
        "gravity_environments",
        {},
    )

    environments = {}

    environment_metric_keys = [
        "gravity_m_per_s2",
        "minimum_relative_potential_energy_j",
        "maximum_relative_potential_energy_j",
        "relative_potential_energy_at_lowest_com_j",
        "energy_difference_to_return_to_initial_height_j",
    ]

    phase_numeric_keys = [
        "phase_number",
        "start_frame_index",
        "end_frame_index",
        "start_time_seconds",
        "end_time_seconds",
        "duration_seconds",
        "start_com_height_m",
        "end_com_height_m",
        "minimum_com_height_m",
        "maximum_com_height_m",
        "potential_energy_change_j",
        "potential_energy_range_j",
    ]

    aggregate_numeric_keys = [
        "phase_count",
        "total_potential_energy_change_j",
        "total_potential_energy_range_j",
    ]

    if isinstance(source_environments, dict):
        for environment_name, environment_data in source_environments.items():

            if not isinstance(environment_data, dict):
                continue

            # Environment-level metrics are stored directly
            # inside each environment in the source report.
            environment_metrics = {
                key: value
                for key in environment_metric_keys
                if (
                    value := _finite_number(
                        environment_data.get(key)
                    )
                ) is not None
            }

            # The source report calls this "movement_phases".
            source_phases = environment_data.get(
                "movement_phases",
                environment_data.get("individual_phase_results", []),
            )

            individual_phases = []

            if isinstance(source_phases, list):
                for phase in source_phases:
                    if not isinstance(phase, dict):
                        continue

                    phase_record = {}

                    for key in [
                        "phase_id",
                        "phase",
                    ]:
                        if key in phase:
                            phase_record[key] = phase[key]

                    for key in phase_numeric_keys:
                        value = _finite_number(phase.get(key))

                        if value is not None:
                            phase_record[key] = value

                    individual_phases.append(phase_record)

            # Preserve the actual movement phase aggregates.
            source_aggregates = environment_data.get(
                "movement_phase_aggregates",
                environment_data.get("phase_aggregates", {}),
            )

            phase_aggregates = {}

            if isinstance(source_aggregates, dict):
                for phase_name, aggregate in source_aggregates.items():

                    if not isinstance(aggregate, dict):
                        continue

                    aggregate_record = {
                        key: value
                        for key in aggregate_numeric_keys
                        if (
                            value := _finite_number(
                                aggregate.get(key)
                            )
                        ) is not None
                    }

                    phase_ids = aggregate.get(
                        "individual_phase_ids"
                    )

                    if isinstance(phase_ids, list):
                        aggregate_record["individual_phase_ids"] = [
                            phase_id
                            for phase_id in phase_ids
                            if isinstance(phase_id, str)
                        ]

                    phase_aggregates[phase_name] = aggregate_record

            environments[environment_name] = {
                "metrics": environment_metrics,
                "individual_phases": individual_phases,
                "phase_aggregates": phase_aggregates,
            }

    has_environment_data = any(
        environment.get("metrics")
        for environment in environments.values()
    )

    return {
        "status": (
            "reported"
            if whole_body_metrics or has_environment_data
            else "unavailable"
        ),
        "method": (
            "Whole-body gravitational potential-energy comparison "
            "using the OpenSim model center of mass."
        ),
        "vertical_axis": gravity_report.get(
            "vertical_axis",
            "OpenSim Y",
        ),
        "metrics": whole_body_metrics,
        "environments": environments,
        "interpretation": gravity_report.get(
            "interpretation",
            [
                "The same reconstructed movement is evaluated "
                "under each gravity condition.",
                "Potential-energy changes describe modeled "
                "gravitational effects, not metabolic expenditure "
                "or measured muscle work.",
            ],
        ),
        "limitations": gravity_report.get(
            "limitations",
            [
                "The comparison holds the reconstructed movement "
                "fixed across gravitational environments.",
                "Center-of-mass estimates depend on reconstructed "
                "kinematics and the scaled OpenSim model.",
            ],
        ),
    }


def build_gravity_insights(
    gravity_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe modeled gravitational context without health inference."""
    if gravity_context["status"] == "unavailable":
        return [{
            "category": "gravity_context",
            "title": "Gravity analysis unavailable",
            "observation": (
                "No usable gravity-analysis results were available "
                "for this assessment."
            ),
            "evidence": [],
            "interpretation": (
                "Gravity-related potential-energy context cannot be "
                "reported for this movement."
            ),
            "suggested_actions": [
                "Review the gravity-analysis output if this context "
                "is needed for the session."
            ],
            "when_to_seek_support": (
                "This is a technical data limitation, not a health finding."
            ),
            "limitations": gravity_context["limitations"],
        }]

    metrics = gravity_context["metrics"]
    environments = gravity_context["environments"]

    mass = metrics.get("total_body_mass_kg")
    excursion = metrics.get("com_height_excursion_m")

    observation_parts = []

    if mass is not None:
        observation_parts.append(
            f"The modeled body mass was {mass:.2f} kg"
        )

    if excursion is not None:
        observation_parts.append(
            f"the reconstructed center-of-mass height excursion "
            f"was {excursion:.3f} m"
        )

    observation = (
        "; ".join(observation_parts) + "."
        if observation_parts
        else "Gravity-environment results were available."
    )

    environment_names = list(environments)

    if environment_names:
        observation += (
            " Relative gravitational potential energy was evaluated "
            "under these environments: "
            + ", ".join(environment_names)
            + "."
        )

    return [{
        "category": "gravity_context",
        "title": "Gravity-dependent potential-energy context",
        "observation": observation,
        "evidence": [{
            "whole_body_metrics": metrics,
            "gravity_environments": environments,
        }],
        "interpretation": (
            "The same reconstructed movement trajectory was evaluated "
            "under each reported gravitational acceleration. The resulting "
            "values describe modeled gravitational potential-energy "
            "changes associated with center-of-mass height. They do not "
            "measure muscle work, metabolic energy expenditure, exercise "
            "effort, joint loading, or movement adaptation."
        ),
        "suggested_actions": [
            "Use the environment comparison as physical context when "
            "reviewing the reconstructed movement.",
            "Interpret these values alongside the reconstruction-quality "
            "limitations and the movement observations."
        ],
        "when_to_seek_support": (
            "This is a modeled physics output, not a medical finding "
            "or an indication for medical consultation."
        ),
        "limitations": gravity_context["limitations"],
    }]


def build_wellness_flags(
    daily_checkin: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create transparent flags from reported check-in values."""
    checkin = validate_daily_checkin(daily_checkin)
    flags = []
    missing_fields = []

    sleep = checkin["sleep"]
    sleep_hours = sleep["duration_hours"]
    sleep_quality = sleep["quality"]

    if sleep_hours is None:
        missing_fields.append("sleep.duration_hours")
    elif sleep_hours < DEMO_SHORT_SLEEP_HOURS:
        flags.append({
            "code": "short_sleep_context",
            "category": "sleep",
            "severity": "context",
            "evidence": {
                "duration_hours": sleep_hours,
                "demo_threshold_hours": DEMO_SHORT_SLEEP_HOURS,
            },
            "message": (
                "Reported sleep duration is below the configurable "
                "demonstration reference of 6 hours."
            ),
        })

    if sleep_quality is None or sleep_quality == "not_reported":
        missing_fields.append("sleep.quality")
    elif sleep_quality == "poor":
        flags.append({
            "code": "poor_sleep_quality",
            "category": "sleep",
            "severity": "context",
            "evidence": {"sleep_quality": sleep_quality},
            "message": "You reported poor sleep quality.",
        })

    fatigue = checkin["fatigue"]
    if fatigue is None:
        missing_fields.append("fatigue")
    elif fatigue in {"high", "severe"}:
        flags.append({
            "code": "elevated_fatigue",
            "category": "fatigue",
            "severity": "attention",
            "evidence": {"fatigue": fatigue},
            "message": f"You reported {fatigue} fatigue.",
        })

    for location, severity in checkin["soreness"].items():
        if severity is None:
            missing_fields.append(f"soreness.{location}")
        elif severity in {"moderate", "severe"}:
            flags.append({
                "code": "localized_soreness",
                "category": "soreness",
                "severity": "attention",
                "location": location,
                "evidence": {
                    "body_location": location,
                    "soreness_level": severity,
                },
                "message": (
                    f"You reported {severity} soreness in your "
                    f"{location.replace('_', ' ')}."
                ),
            })

    stress = checkin["stress_level"]
    if stress is None:
        missing_fields.append("stress_level")
    elif stress >= DEMO_ELEVATED_STRESS_LEVEL:
        flags.append({
            "code": "elevated_stress_context",
            "category": "stress",
            "severity": "context",
            "evidence": {
                "stress_level": stress,
                "demo_threshold": DEMO_ELEVATED_STRESS_LEVEL,
            },
            "message": (
                "Your reported stress level is elevated relative to "
                "the demonstration rule."
            ),
        })

    for field in ("hydration", "nutrition"):
        value = checkin[field]
        if value is None:
            missing_fields.append(field)
        elif value == "behind":
            flags.append({
                "code": f"{field}_behind",
                "category": field,
                "severity": "context",
                "evidence": {field: value},
                "message": f"You reported being behind on {field}.",
            })

    pain_note = checkin["pain_note"]
    if pain_note:
        flags.append({
            "code": "pain_or_discomfort_reported",
            "category": "pain",
            "severity": "attention",
            "evidence": {"pain_note": pain_note},
            "message": (
                "You provided an optional note describing pain "
                "or discomfort."
            ),
        })

    return {
        "flags": flags,
        "missing_fields": missing_fields,
        "is_complete": len(missing_fields) == 0,
    }, flags


def build_wellness_insights(
    wellness_flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn reported wellness flags into proportionate guidance."""
    insights = []

    codes = {flag["code"] for flag in wellness_flags}

    recovery_codes = {
        "short_sleep_context",
        "poor_sleep_quality",
        "elevated_fatigue",
    }

    if codes.intersection(recovery_codes):
        evidence = [
            flag for flag in wellness_flags
            if flag["code"] in recovery_codes
        ]

        insights.append({
            "category": "wellness_context",
            "title": "Sleep and fatigue context",
            "observation": "Your check-in includes sleep or fatigue factors "
                           "that may be relevant to today's activity.",
            "evidence": evidence,
            "interpretation": (
                "These are self-reported context indicators. They do not "
                "measure physical readiness or establish a cause."
            ),
            "suggested_actions": [
                "Consider prioritizing your usual recovery routine.",
                "Pay attention to unusual fatigue during routine activity.",
                "Avoid pushing through fatigue that feels unusual or worsens."
            ],
            "when_to_seek_support": (
                "Discuss persistent, worsening, or activity-limiting fatigue "
                "with your CMO."
            ),
            "limitations": [
                "The sleep-duration reference is a configurable demo "
                "heuristic, not a clinical threshold."
            ],
        })

    soreness_flags = [
        flag for flag in wellness_flags
        if flag["code"] == "localized_soreness"
    ]

    if soreness_flags:
        locations = [
            flag["location"].replace("_", " ")
            for flag in soreness_flags
        ]

        insights.append({
            "category": "wellness_context",
            "title": "Reported muscle or joint soreness",
            "observation": (
                "You reported moderate or severe soreness in: "
                + ", ".join(locations)
                + "."
            ),
            "evidence": soreness_flags,
            "interpretation": (
                "This records your reported soreness. It does not identify "
                "the cause or determine whether an exercise is safe."
            ),
            "suggested_actions": [
                "Pay attention to whether discomfort changes during "
                "or after routine activity.",
                "Avoid pushing through movements that aggravate discomfort.",
                "Share persistent or changing symptoms with your CMO."
            ],
            "when_to_seek_support": (
                "Consult your CMO if soreness is persistent, worsening, "
                "or interfering with normal activity."
            ),
            "limitations": [
                "TALA does not diagnose the source of soreness."
            ],
        })

    if "pain_or_discomfort_reported" in codes:
        pain_flag = next(
            flag for flag in wellness_flags
            if flag["code"] == "pain_or_discomfort_reported"
        )

        insights.append({
            "category": "wellness_context",
            "title": "Pain or discomfort note",
            "observation": (
                "You submitted an optional pain or discomfort note."
            ),
            "evidence": [pain_flag],
            "interpretation": (
                "The note is preserved as user-reported context. TALA does "
                "not diagnose or infer a medical condition from free text."
            ),
            "suggested_actions": [
                "Describe when the discomfort occurs and whether it changes "
                "with activity when discussing it with your CMO.",
                "Avoid pushing through activity that worsens the discomfort."
            ],
            "when_to_seek_support": (
                "Discuss new, persistent, or worsening pain with your CMO. "
                "For severe or sudden symptoms, or other emergency warning "
                "signs, follow applicable emergency procedures and seek "
                "immediate medical attention."
            ),
            "limitations": [
                "The free-text note is not automatically classified "
                "as a diagnosis or severity score."
            ],
        })

    for field in ("hydration", "nutrition"):
        code = f"{field}_behind"
        if code in codes:
            flag = next(item for item in wellness_flags if item["code"] == code)

            if field == "hydration":
                actions = [
                    "Follow your established hydration plan and make water "
                    "available as permitted by your operational guidance."
                ]
            else:
                actions = [
                    "Follow your established meal plan and raise any "
                    "persistent difficulty meeting it with your support team."
                ]

            insights.append({
                "category": "wellness_context",
                "title": f"{field.capitalize()} check-in context",
                "observation": flag["message"],
                "evidence": [flag],
                "interpretation": (
                    "This is a self-reported status, not a measurement "
                    "of hydration or nutritional adequacy."
                ),
                "suggested_actions": actions,
                "when_to_seek_support": (
                    "Raise persistent concerns with your CMO or the "
                    "appropriate support personnel."
                ),
                "limitations": [
                    "TALA does not calculate fluid requirements or prescribe "
                    "dietary changes."
                ],
            })

    if "elevated_stress_context" in codes:
        flag = next(
            item for item in wellness_flags
            if item["code"] == "elevated_stress_context"
        )

        insights.append({
            "category": "wellness_context",
            "title": "Reported stress context",
            "observation": "Your reported stress level is elevated "
                           "relative to the demonstration rule.",
            "evidence": [flag],
            "interpretation": (
                "This reflects your check-in response and is not a "
                "psychological assessment."
            ),
            "suggested_actions": [
                "Consider using a familiar, approved stress-management "
                "or relaxation routine."
            ],
            "when_to_seek_support": (
                "Discuss persistent stress or stress that interferes "
                "with daily activities with your CMO or appropriate support."
            ),
            "limitations": [
                "The stress reference is a configurable demonstration "
                "heuristic, not a clinical threshold."
            ],
        })

    return insights


def build_movement_wellness_insights(
    movement_observations: list[dict[str, Any]],
    wellness_flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe overlap between estimated movement and reported soreness."""
    insights = []

    soreness_by_location = {
        flag["location"]: flag
        for flag in wellness_flags
        if flag["code"] == "localized_soreness"
    }

    for joint in ("hip", "knee", "ankle"):
        soreness_location = {
            "hip": "hips",
            "knee": "knees",
            "ankle": "ankles",
        }[joint]

        soreness_flag = soreness_by_location.get(soreness_location)

        if soreness_flag is None:
            continue

        movement = next(
            (
                item for item in movement_observations
                if item["joint"] == joint and item["available"]
            ),
            None,
        )

        if movement is None:
            continue

        insights.append({
            "category": "movement_wellness_context",
            "title": f"{joint.capitalize()} movement and reported soreness",
            "observation": (
                f"Your reconstructed movement contains estimated "
                f"{joint} coordinate changes, and you reported "
                f"{soreness_flag['evidence']['soreness_level']} "
                f"{joint} soreness."
            ),
            "evidence": [
                soreness_flag,
                {
                    "joint": joint,
                    "coordinate_observation": movement,
                },
            ],
            "interpretation": (
                "These findings provide context for reviewing the session "
                "together. They do not establish that the movement caused "
                "the soreness or determine movement safety."
            ),
            "suggested_actions": [
                "Pay attention to whether discomfort changes during "
                "or after similar movements.",
                "Avoid pushing through movements that aggravate discomfort.",
                "Consider reviewing the symptoms and activity with your CMO."
            ],
            "when_to_seek_support": (
                "Consult your CMO if discomfort persists, worsens, or "
                "limits normal activity."
            ),
            "limitations": [
                "Movement coordinates are estimated from monocular video.",
                "The reconstruction does not measure pain or tissue loading."
            ],
        })

    return insights


def build_quality_insights(
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create technical guidance without misclassifying it as a health issue."""
    if quality["status"] == "unavailable":
        return [{
            "category": "reconstruction_quality",
            "title": "Movement reconstruction unavailable",
            "observation": (
                "The available reconstruction report does not provide "
                "enough recognized quality information."
            ),
            "evidence": [],
            "interpretation": (
                "Movement estimates should not be treated as sufficiently "
                "characterized by the current report."
            ),
            "suggested_actions": [
                "Review the source video and reconstruction outputs before "
                "relying on the movement estimates."
            ],
            "when_to_seek_support": (
                "This is a technical data limitation and does not by itself "
                "indicate a need for medical consultation."
            ),
            "limitations": quality["limitations"],
        }]

    if quality["status"] == "limited":
        return [{
            "category": "reconstruction_quality",
            "title": "Reconstruction has known limitations",
            "observation": (
                "The IK report identifies limitations or unavailable "
                "marker samples."
            ),
            "evidence": [
                {"metrics": quality["metrics"]},
                {"limitations": quality["limitations"]},
            ],
            "interpretation": (
                "The movement estimates should be interpreted cautiously, "
                "particularly for body regions affected by missing markers."
            ),
            "suggested_actions": [
                "For a future recording, keep the full body visible, use a "
                "stable camera position, and ensure adequate lighting.",
                "Avoid interpreting unavailable or neutralized coordinates "
                "as observed movement."
            ],
            "when_to_seek_support": (
                "This is a reconstruction limitation, not a medical finding."
            ),
            "limitations": [
                quality["interpretation"]
            ],
        }]

    return []
