
"""Run exercise-aware movement analysis on an existing COM trajectory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the project source directory is importable when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tala.movement_analysis import (
    get_profile,
    segment_movement,
    count_repetitions,
    evaluate_completeness,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "gravity_analysis"
    / "gravity_energy_timeseries.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "movement_analysis"
)

EXERCISE = "squat"


def main() -> None:
    print("=" * 72)
    print("TALA - MOVEMENT ANALYSIS")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"COM trajectory not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "frame_index",
        "time",
        "com_y_m",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.sort_values("frame_index").reset_index(drop=True)

    profile = get_profile(EXERCISE)

    print(f"\nExercise: {profile.display_name}")
    print(f"Input: {INPUT_PATH}")
    print(f"Samples: {len(data)}")

    segmentation = segment_movement(
        signal=data[profile.primary_signal].to_numpy(),
        profile=profile,
        frame_indices=data["frame_index"].to_numpy(),
        timestamps=data["time"].to_numpy(),
        smoothing_window=5,
        min_phase_frames=3,
    )

    repetitions = count_repetitions(
        result=segmentation,
        profile=profile,
    )

    completeness = evaluate_completeness(
        signal=data[profile.primary_signal].to_numpy(),
        result=segmentation,
        profile=profile,
    )

    report = {
        "exercise": profile.name,
        "input_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "sample_count": len(data),
        "segmentation": segmentation.to_dict(),
        "repetitions": repetitions,
        "completeness": completeness,
        "limitations": [
            "Exercise type is supplied by the caller, not recognized automatically.",
            "Phase segmentation uses one numerical movement signal.",
            "Completeness is estimated from relative endpoint positions.",
            "This analysis does not validate the underlying OpenSim reconstruction.",
        ],
    }

    report_path = OUTPUT_DIR / "movement_analysis.json"
    phases_path = OUTPUT_DIR / "movement_phases.csv"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    phase_rows = [
        segment.to_dict()
        for segment in segmentation.segments
    ]

    pd.DataFrame(phase_rows).to_csv(
        phases_path,
        index=False,
    )

    print("\nTURNING POINTS")
    print("-" * 72)

    for point in segmentation.turning_points:
        print(
            f"Frame {point['frame_index']:>4} | "
            f"Time {point['time']:>7.3f}s | "
            f"Value {point['value']:.6f} | "
            f"{point['kind']}"
        )

    print("\nMOVEMENT PHASES")
    print("-" * 72)

    for segment in segmentation.segments:
        print(
            f"{segment.phase:<22} "
            f"Frames {segment.start_frame:>4}-{segment.end_frame:<4} "
            f"Duration {segment.duration_seconds:.3f}s"
        )

    print("\nREPETITION ANALYSIS")
    print("-" * 72)
    print(f"Repetitions: {repetitions['repetition_count']}")

    print("\nCOMPLETENESS")
    print("-" * 72)
    print(f"Status: {completeness['status']}")
    print(f"Reason: {completeness['reason']}")

    print("\nOUTPUTS")
    print("-" * 72)
    print(report_path)
    print(phases_path)

    print("\nMovement analysis complete.")


if __name__ == "__main__":
    main()
