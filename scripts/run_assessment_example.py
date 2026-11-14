
"""
TALA - Contextual Wellness and Movement Assessment
===================================================

Example runner for the assessment module.

Inputs:
    results/kinematic_reconstruction/mapped_coordinates.csv
    results/kinematic_reconstruction/ik_report.json

Output:
    results/assessment/assessment_output.json

Run from the project root:

    python scripts/run_assessment_example.py
"""

import json
from pathlib import Path

import pandas as pd

from tala.assessment import generate_assessment


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECONSTRUCTION_DIR = (
    PROJECT_ROOT / "results" / "kinematic_reconstruction"
)

GRAVITY_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "gravity_analysis"
)

ASSESSMENT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "assessment"
)

MAPPED_COORDINATES_PATH = (
    RECONSTRUCTION_DIR / "mapped_coordinates.csv"
)

IK_REPORT_PATH = (
    RECONSTRUCTION_DIR / "ik_report.json"
)

GRAVITY_REPORT_PATH = (
    GRAVITY_OUTPUT_DIR / "gravity_analysis.json"
)

ASSESSMENT_OUTPUT_PATH = (
    ASSESSMENT_OUTPUT_DIR / "assessment_output.json"
)


# ============================================================
# DAILY WELLNESS CHECK-IN
# ============================================================

def get_example_daily_checkin() -> dict:
    """
    Return an illustrative daily wellness check-in.

    These values are demonstration inputs only.
    They are not defaults for actual users.
    """
    return {
        "sleep": {
            "duration_hours": 7.0,
            "quality": "fair",
        },
        "fatigue": "moderate",
        "soreness": {
            "shoulders": "none",
            "arms": "none",
            "elbows": "none",
            "lower_back": "none",
            "hips": "none",
            "thighs": "none",
            "knees": "moderate",
            "ankles": "none",
        },
        "stress_level": 3,
        "hydration": "on_track",
        "nutrition": "on_track",
        "pain_note": None,
    }


# ============================================================
# INPUT LOADING
# ============================================================

def load_assessment_inputs():
    """Load reconstruction and gravity analysis outputs."""

    required_files = [
        MAPPED_COORDINATES_PATH,
        IK_REPORT_PATH,
        GRAVITY_REPORT_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required assessment input not found:\n{path}"
            )

    mapped_coordinates = pd.read_csv(
        MAPPED_COORDINATES_PATH
    )

    with IK_REPORT_PATH.open("r", encoding="utf-8") as file:
        ik_report = json.load(file)

    with GRAVITY_REPORT_PATH.open("r", encoding="utf-8") as file:
        gravity_report = json.load(file)

    return mapped_coordinates, ik_report, gravity_report


# ============================================================
# ASSESSMENT EXECUTION
# ============================================================

def run_assessment():
    """Generate and save the contextual assessment."""

    print("=" * 65)
    print("TALA - CONTEXTUAL WELLNESS AND MOVEMENT ASSESSMENT")
    print("=" * 65)

    print("\n[1/3] Loading reconstruction and gravity outputs...")

    mapped_coordinates, ik_report, gravity_report = (
        load_assessment_inputs()
    )

    print(f"      Coordinate samples: {len(mapped_coordinates)}")
    print(f"      Coordinate columns: {len(mapped_coordinates.columns)}")
    print("      IK report loaded.")
    print("      Gravity report loaded.")

    print("\n[2/3] Preparing daily wellness check-in...")

    daily_checkin = get_example_daily_checkin()

    print("      Example check-in loaded.")

    print("\n[3/3] Generating assessment...")

    assessment = generate_assessment(
        daily_checkin=daily_checkin,
        mapped_coordinates=mapped_coordinates,
        ik_report=ik_report,
        gravity_report=gravity_report,
    )

    ASSESSMENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ASSESSMENT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            assessment,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("ASSESSMENT COMPLETED")
    print("=" * 65)

    print(f"\nOutput: {ASSESSMENT_OUTPUT_PATH}")

    print("\nAssessment summary:")
    print(
        f"  Movement observations: "
        f"{len(assessment['movement_observations'])}"
    )
    print(
        f"  Wellness flags: "
        f"{len(assessment['wellness_flags'])}"
    )
    print(
        f"  Context flags: "
        f"{len(assessment['context_flags'])}"
    )
    print(
        f"  Contextual insights: "
        f"{len(assessment['contextual_insights'])}"
    )

    print("\n" + "=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_assessment()
