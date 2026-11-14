
"""Run gravity-aware energy analysis on Phase 3.10 outputs."""

import argparse
from pathlib import Path

from tala.kinematic_reconstruction.gravity import (
    run_gravity_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results"
    / "kinematic_reconstruction"
    / "scaled_model.osim"
)

DEFAULT_COORDINATES = (
    PROJECT_ROOT
    / "results"
    / "kinematic_reconstruction"
    / "mapped_coordinates.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "gravity_analysis"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate gravity-aware potential energy "
            "from Phase 3.10 reconstructed movement."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--coordinates",
        type=Path,
        default=DEFAULT_COORDINATES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    run_gravity_analysis(
        model_path=args.model,
        coordinates_path=args.coordinates,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
