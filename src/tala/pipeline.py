
"""Unified TALA assessment and evidence-explanation pipeline."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from tala.assessment import generate_assessment
from tala.assessment.integration import integrate_movement_assessment
from tala.baselines import compare_with_baseline, record_session
from tala.baselines.engine import DEFAULT_HISTORY_PATH
from tala.rag.service import explain_assessment


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RECONSTRUCTION_DIR = (
    PROJECT_ROOT / "results" / "kinematic_reconstruction"
)

DEFAULT_GRAVITY_DIR = (
    PROJECT_ROOT / "results" / "gravity_analysis"
)

DEFAULT_MOVEMENT_DIR = (
    PROJECT_ROOT / "results" / "movement_analysis"
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "tala_output.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _file_sha256(path: Path) -> str:
    """Return a stable SHA-256 fingerprint for an analysis artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def run_tala_pipeline(
    *,
    question: str,
    daily_checkin: dict[str, Any] | None = None,
    session_id: str | None = None,
    baseline_scope: str | None = None,
    reconstruction_dir: Path = DEFAULT_RECONSTRUCTION_DIR,
    gravity_dir: Path = DEFAULT_GRAVITY_DIR,
    movement_dir: Path = DEFAULT_MOVEMENT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run assessment, baseline comparison, and RAG explanation."""

    if not question.strip():
        raise ValueError("A RAG question is required.")

    run_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # 1. Load existing analysis artifacts
    # --------------------------------------------------------

    coordinates_path = reconstruction_dir / "mapped_coordinates.csv"
    ik_path = reconstruction_dir / "ik_report.json"
    gravity_path = gravity_dir / "gravity_analysis.json"
    movement_path = movement_dir / "movement_analysis.json"

    for path in (
        coordinates_path,
        ik_path,
        gravity_path,
        movement_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required TALA analysis output not found: {path}"
            )

    # Identify the recording from the exact movement-analysis artifact.
    recording_fingerprint = _file_sha256(movement_path)

    # An explicitly supplied session ID takes precedence.
    # Otherwise, repeated processing of the same artifact reuses its ID.
    session_id = session_id or f"sha256:{recording_fingerprint}"

    mapped_coordinates = pd.read_csv(coordinates_path)

    ik_report = _load_json(ik_path)
    gravity_report = _load_json(gravity_path)
    movement_report = _load_json(movement_path)

    # --------------------------------------------------------
    # 2. Generate the existing contextual assessment
    # --------------------------------------------------------

    assessment = generate_assessment(
        daily_checkin=daily_checkin,
        mapped_coordinates=mapped_coordinates,
        ik_report=ik_report,
        gravity_report=gravity_report,
    )

    # --------------------------------------------------------
    # 3. Compare against previous compatible recordings
    # --------------------------------------------------------

    baseline = compare_with_baseline(
        movement_report,
        history_path=history_path,
        session_id=session_id,
        recording_fingerprint=recording_fingerprint,
        baseline_scope=baseline_scope,
    )

    # --------------------------------------------------------
    # 4. Integrate movement analysis and baseline
    # --------------------------------------------------------

    assessment = integrate_movement_assessment(
        assessment=assessment,
        movement_report=movement_report,
        baseline_comparison=baseline.to_dict(),
    )

    # --------------------------------------------------------
    # 5. Generate evidence-grounded explanation
    # --------------------------------------------------------

    rag_result = explain_assessment(
        question=question,
        assessment=assessment,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 6. Assemble unified output
    # --------------------------------------------------------

    result = {
        "run_id": run_id,
        "session_id": session_id,
        "recording_fingerprint": recording_fingerprint,
        "assessment": assessment,
        "rag": rag_result,
    }

    # --------------------------------------------------------
    # 7. Save output and record this recording for future baselines
    # --------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    record_session(
        movement_report,
        session_id=session_id,
        recording_fingerprint=recording_fingerprint,
        history_path=history_path,
        baseline_scope=baseline_scope,
    )

    return result
