"""Run the Phase 3.10 full-body inverse-kinematics pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tala.kinematic_reconstruction.pipeline import main


if __name__ == "__main__":
    main()