
"""Gravity-aware mechanical energy analysis using reconstructed OpenSim motion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import opensim as osim
from tala.movement_analysis.profiles import get_profile
from tala.movement_analysis.segmentation import segment_movement
from tala.movement_analysis.repetitions import count_repetitions
from tala.movement_analysis.completeness import evaluate_completeness


GRAVITY_PRESETS = {
    "microgravity": 0.0,
    "moon": 1.62,
    "mars": 3.71,
    "earth": 9.80665,
}

VERTICAL_AXIS_INDEX = 1


def _vec3_component(vector, index: int) -> float:
    """Read one component from an OpenSim Vec3."""
    try:
        return float(vector.get(index))
    except AttributeError:
        return float(vector[index])


def _get_total_body_mass(model) -> float:
    """Calculate total modeled body mass, excluding the ground."""
    body_set = model.getBodySet()

    total_mass = sum(
        float(body_set.get(index).getMass())
        for index in range(body_set.getSize())
    )

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError(
            f"Invalid total OpenSim body mass: {total_mass}"
        )

    return total_mass


def calculate_center_of_mass_trajectory(
    model_path: str | Path,
    coordinates_path: str | Path,
) -> tuple[pd.DataFrame, float]:
    """
    Evaluate whole-body center of mass for every reconstructed frame.

    Coordinate CSV rotational values are expected in radians and
    translational values in meters, as exported by Phase 3.10.
    """
    model_path = Path(model_path)
    coordinates_path = Path(coordinates_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Scaled OpenSim model not found: {model_path}"
        )

    if not coordinates_path.exists():
        raise FileNotFoundError(
            f"Mapped coordinate CSV not found: {coordinates_path}"
        )

    coordinates_table = pd.read_csv(coordinates_path)

    required_columns = {"frame_index", "time"}

    missing = required_columns - set(coordinates_table.columns)

    if missing:
        raise ValueError(
            f"Coordinate CSV is missing required columns: {sorted(missing)}"
        )

    if coordinates_table.empty:
        raise ValueError("Coordinate CSV contains no frames.")

    times = coordinates_table["time"].to_numpy(dtype=float)

    if not np.isfinite(times).all():
        raise ValueError("Coordinate timestamps contain nonfinite values.")

    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Coordinate timestamps must be strictly increasing.")

    model = osim.Model(str(model_path))
    state = model.initSystem()

    coordinate_set = model.getCoordinateSet()

    coordinate_names = [
        coordinate_set.get(index).getName()
        for index in range(coordinate_set.getSize())
    ]

    missing_coordinates = (
        set(coordinate_names) - set(coordinates_table.columns)
    )

    if missing_coordinates:
        raise ValueError(
            "Mapped coordinate CSV is missing OpenSim coordinates: "
            f"{sorted(missing_coordinates)}"
        )

    total_mass = _get_total_body_mass(model)

    com_x = []
    com_y = []
    com_z = []

    for row_index, row in coordinates_table.iterrows():
        for coordinate_name in coordinate_names:
            value = float(row[coordinate_name])

            if not np.isfinite(value):
                raise ValueError(
                    f"Nonfinite coordinate value at row {row_index}: "
                    f"{coordinate_name}={value}"
                )

            coordinate = coordinate_set.get(coordinate_name)

            # Phase 3.10 exports rotational values in radians.
            # enforceConstraints=False preserves the exported IK values.
            coordinate.setValue(state, value, False)

        model.realizePosition(state)

        center_of_mass = model.calcMassCenterPosition(state)

        com_x.append(_vec3_component(center_of_mass, 0))
        com_y.append(
            _vec3_component(
                center_of_mass,
                VERTICAL_AXIS_INDEX,
            )
        )
        com_z.append(_vec3_component(center_of_mass, 2))

    result = pd.DataFrame({
        "frame_index": coordinates_table["frame_index"].to_numpy(),
        "time": times,
        "com_x_m": com_x,
        "com_y_m": com_y,
        "com_z_m": com_z,
    })

    if not np.isfinite(
        result[["com_x_m", "com_y_m", "com_z_m"]].to_numpy()
    ).all():
        raise ValueError("OpenSim returned nonfinite center-of-mass values.")

    return result, total_mass


def analyze_movement_phases(
    com_trajectory: pd.DataFrame,
    exercise: str = "squat",
    smoothing_window: int = 5,
) -> tuple[list[dict], dict]:
    """Analyze movement phases using the shared movement-analysis module."""

    if len(com_trajectory) < 3:
        raise ValueError(
            "At least three COM samples are required for phase detection."
        )

    required_columns = {"frame_index", "time", "com_y_m"}
    missing = required_columns - set(com_trajectory.columns)

    if missing:
        raise ValueError(
            f"COM trajectory is missing columns: {sorted(missing)}"
        )

    heights = com_trajectory["com_y_m"].to_numpy(dtype=float)

    if not np.isfinite(heights).all():
        raise ValueError("COM heights contain nonfinite values.")

    profile = get_profile(exercise)

    segmentation = segment_movement(
        signal=heights,
        profile=profile,
        frame_indices=com_trajectory["frame_index"].to_numpy(),
        timestamps=com_trajectory["time"].to_numpy(dtype=float),
        smoothing_window=smoothing_window,
        min_phase_frames=3,
    )

    repetitions = count_repetitions(
        segmentation,
        profile,
    )

    completeness = evaluate_completeness(
        heights,
        segmentation,
        profile,
    )

    phases = []

    for segment in segmentation.segments:
        start_index = segment.start_index
        end_index = segment.end_index

        phase_heights = heights[start_index:end_index + 1]

        phases.append({
            "phase": segment.phase,
            "direction": segment.direction,

            "start_row_index": int(start_index),
            "end_row_index": int(end_index),

            "start_frame_index": int(segment.start_frame),
            "end_frame_index": int(segment.end_frame),

            "start_time_seconds": float(segment.start_time),
            "end_time_seconds": float(segment.end_time),

            "duration_seconds": float(segment.duration_seconds),

            "start_com_height_m": float(phase_heights[0]),
            "end_com_height_m": float(phase_heights[-1]),

            "minimum_com_height_m": float(np.min(phase_heights)),
            "maximum_com_height_m": float(np.max(phase_heights)),
        })

    movement_summary = {
        "exercise": profile.name,
        "signal": segmentation.signal_name,
        "turning_points": segmentation.turning_points,
        "repetition_analysis": repetitions,
        "completeness": completeness,
        "segmentation_parameters": {
            "smoothing_window_samples": smoothing_window,
            "minimum_phase_frames": 3,
        },
    }

    return phases, movement_summary


def detect_movement_phases(
    com_trajectory: pd.DataFrame,
    smoothing_window: int = 5,
) -> list[dict]:
    """Compatibility wrapper for existing gravity-analysis callers."""

    phases, _ = analyze_movement_phases(
        com_trajectory,
        exercise="squat",
        smoothing_window=smoothing_window,
    )

    return phases



def calculate_gravity_energy(
    com_trajectory: pd.DataFrame,
    total_mass_kg: float,
    phases: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Calculate whole-sequence and individual movement-phase gravitational energy."""
    if com_trajectory.empty:
        raise ValueError("Center-of-mass trajectory is empty.")

    if not np.isfinite(total_mass_kg) or total_mass_kg <= 0:
        raise ValueError("Total body mass must be finite and positive.")

    heights = com_trajectory["com_y_m"].to_numpy(dtype=float)
    times = com_trajectory["time"].to_numpy(dtype=float)

    if not np.isfinite(heights).all():
        raise ValueError("COM heights contain nonfinite values.")

    if not np.isfinite(times).all():
        raise ValueError("COM timestamps contain nonfinite values.")

    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("COM timestamps must be strictly increasing.")

    initial_height = float(heights[0])
    relative_height = heights - initial_height

    output = com_trajectory.copy()
    output["com_height_change_m"] = relative_height

    lowest_row_index = int(np.argmin(heights))
    highest_row_index = int(np.argmax(heights))

    lowest_height = float(heights[lowest_row_index])
    highest_height = float(heights[highest_row_index])

    if phases is None:
        phases = detect_movement_phases(com_trajectory)

    # Validate phase row indices before calculating energy.
    for phase in phases:
        start_index = int(phase["start_row_index"])
        end_index = int(phase["end_row_index"])

        if not (0 <= start_index <= end_index < len(heights)):
            raise ValueError(
                f"Invalid row indices for movement phase: {phase}"
            )

    environment_results = {}

    for environment, gravity in GRAVITY_PRESETS.items():
        energy = total_mass_kg * gravity * relative_height

        output[f"potential_energy_{environment}_j"] = energy

        bottom_energy = float(energy[lowest_row_index])

        individual_phase_results = []
        aggregate_phase_results = {}

        for phase_number, phase in enumerate(phases, start=1):
            start_index = int(phase["start_row_index"])
            end_index = int(phase["end_row_index"])

            phase_type = str(phase["phase"]).lower()

            phase_heights = heights[start_index:end_index + 1]

            phase_start_height = float(phase_heights[0])
            phase_end_height = float(phase_heights[-1])

            phase_energy_change = (
                total_mass_kg
                * gravity
                * (phase_end_height - phase_start_height)
            )

            phase_energy_range = (
                total_mass_kg
                * gravity
                * (
                    float(np.max(phase_heights))
                    - float(np.min(phase_heights))
                )
            )

            phase_result = {
                "phase_id": f"{phase_type}_{phase_number}",
                "phase": phase_type,
                "phase_number": phase_number,
                "start_frame_index": int(
                    com_trajectory.iloc[start_index]["frame_index"]
                ),
                "end_frame_index": int(
                    com_trajectory.iloc[end_index]["frame_index"]
                ),
                "start_time_seconds": float(times[start_index]),
                "end_time_seconds": float(times[end_index]),
                "duration_seconds": float(
                    times[end_index] - times[start_index]
                ),
                "start_com_height_m": phase_start_height,
                "end_com_height_m": phase_end_height,
                "minimum_com_height_m": float(np.min(phase_heights)),
                "maximum_com_height_m": float(np.max(phase_heights)),
                "potential_energy_change_j": float(
                    phase_energy_change
                ),
                "potential_energy_range_j": float(
                    phase_energy_range
                ),
            }

            # Preserve every individual phase.
            individual_phase_results.append(phase_result)

            # Accumulate statistics by phase type.
            if phase_type not in aggregate_phase_results:
                aggregate_phase_results[phase_type] = {
                    "phase_count": 0,
                    "total_potential_energy_change_j": 0.0,
                    "total_potential_energy_range_j": 0.0,
                    "individual_phase_ids": [],
                }

            aggregate = aggregate_phase_results[phase_type]
            aggregate["phase_count"] += 1
            aggregate["total_potential_energy_change_j"] += (
                phase_energy_change
            )
            aggregate["total_potential_energy_range_j"] += (
                phase_energy_range
            )
            aggregate["individual_phase_ids"].append(
                phase_result["phase_id"]
            )

        environment_results[environment] = {
            "gravity_m_per_s2": gravity,
            "initial_potential_energy_reference_j": 0.0,
            "minimum_relative_potential_energy_j": float(
                np.min(energy)
            ),
            "maximum_relative_potential_energy_j": float(
                np.max(energy)
            ),
            "relative_potential_energy_at_lowest_com_j": bottom_energy,
            "energy_difference_to_return_to_initial_height_j": (
                -bottom_energy
            ),
            "movement_phases": individual_phase_results,
            "movement_phase_aggregates": aggregate_phase_results,
        }

    phase_report = []

    for phase in phases:
        phase_report.append({
            key: value
            for key, value in phase.items()
            if key not in {"start_row_index", "end_row_index"}
        })

    report = {
        "method": (
            "Whole-body gravitational potential-energy comparison "
            "using the OpenSim model center of mass."
        ),
        "total_body_mass_kg": float(total_mass_kg),
        "vertical_axis": "OpenSim Y",
        "initial_com_height_m": initial_height,
        "minimum_com_height_m": lowest_height,
        "maximum_com_height_m": highest_height,
        "com_height_excursion_m": float(
            highest_height - lowest_height
        ),
        "lowest_com_frame_index": int(
            output.iloc[lowest_row_index]["frame_index"]
        ),
        "lowest_com_time_seconds": float(
            output.iloc[lowest_row_index]["time"]
        ),
        "highest_com_frame_index": int(
            output.iloc[highest_row_index]["frame_index"]
        ),
        "highest_com_time_seconds": float(
            output.iloc[highest_row_index]["time"]
        ),
        "phase_detection": {
            "method": "Shared signal-based movement segmentation",
            "signal": "com_y_m",
            "smoothing_used_for_energy_calculation": False,
            "phase_boundaries_used_for_energy_calculation": True,
        },
        "movement_phases": phase_report,
        "gravity_environments": environment_results,
        "interpretation": [
            "The same reconstructed movement is evaluated under each gravity condition.",
            "Relative potential energy is referenced to the first reconstructed frame.",
            "Phase energy changes are calculated from the COM height at each phase's start and end.",
            "Potential-energy range describes the difference between the highest and lowest COM positions within a phase.",
            "Aggregate phase energy changes are sums of the individual phase endpoint changes, not estimates of metabolic expenditure.",
            "Energy differences represent gravitational potential-energy changes, not measured muscle work or metabolic energy expenditure.",
            "The analysis does not predict movement adaptation, muscle activation, joint loading, or exercise resistance.",
        ],
        "limitations": [
            "Center-of-mass estimates depend on the scaled OpenSim model and reconstructed kinematics.",
            "The input movement is a monocular-video-derived kinematic reconstruction, not calibrated motion capture.",
            "The comparison holds the reconstructed movement fixed across gravitational environments.",
            "The calculation assumes the OpenSim Y-axis is vertical.",
            "Phase boundaries are estimated from the smoothed COM signal and may not correspond exactly to independently annotated exercise repetitions.",
        ],
    }

    return output, report

def run_gravity_analysis(
    model_path: str | Path,
    coordinates_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run and save gravity-aware energy analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    com_trajectory, total_mass = calculate_center_of_mass_trajectory(
        model_path,
        coordinates_path,
    )

    phases, movement_summary = analyze_movement_phases(
        com_trajectory,
        exercise="squat",
    )

    timeseries, report = calculate_gravity_energy(
        com_trajectory,
        total_mass,
        phases=phases,
    )

    report["movement_analysis"] = movement_summary

    timeseries_path = output_dir / "gravity_energy_timeseries.csv"
    report_path = output_dir / "gravity_analysis.json"

    timeseries.to_csv(
        timeseries_path,
        index=False,
        float_format="%.10f",
    )

    report["input_model"] = str(Path(model_path))
    report["input_coordinates"] = str(Path(coordinates_path))
    report["output_files"] = {
        "timeseries": str(timeseries_path),
        "report": str(report_path),
    }

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("=" * 72)
    print("GRAVITY-AWARE ENERGY ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Total modeled body mass: {total_mass:.3f} kg")
    print(
        "COM height excursion: "
        f"{report['com_height_excursion_m']:.4f} m"
    )
    print(
        "Lowest COM frame: "
        f"{report['lowest_com_frame_index']}"
    )
    print(
        "Lowest COM time: "
        f"{report['lowest_com_time_seconds']:.3f} s"
    )

    print("\nMOVEMENT PHASES")
    print("-" * 72)

    for phase in report["movement_phases"]:
        print(
            f"{phase['phase'].replace('_', ' ').title():<18} "
            f"Frames {phase['start_frame_index']}–"
            f"{phase['end_frame_index']} | "
            f"{phase['duration_seconds']:.3f} s"
        )

    print("\nEARTH GRAVITY: PHASE ENERGY CHANGES")
    print("-" * 72)

    earth_phases = report["gravity_environments"]["earth"]["movement_phases"]

    for phase in earth_phases:
        phase_name = phase["phase_id"]
        values = phase

        print(
            f"{phase_name:<18} "
            f"Delta U: {values['potential_energy_change_j']:+.3f} J | "
            f"Range: {values['potential_energy_range_j']:.3f} J"
        )

    print(f"Timeseries: {timeseries_path}")
    print(f"Report: {report_path}")

    return report
