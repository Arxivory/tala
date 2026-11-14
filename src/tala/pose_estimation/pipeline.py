"""End-to-end Phase 3.3 pose-estimation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import yaml

from tala.common.io import save_dataframe_csv, save_json
from tala.pose_estimation.landmarks import (
    create_mp_image,
    create_pose_landmarker,
    extract_landmark_rows,
)
from tala.pose_estimation.quality import build_extraction_report
from tala.pose_estimation.video import (
    create_video_writer,
    draw_pose_overlay,
    open_video,
    read_video_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "phase3_3_pose_extraction.yaml"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pose_landmarker"
    / "pose_landmarker_full.task"
)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """Load the Phase 3.3 YAML configuration."""
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = resolve_project_path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_video_metadata(
    config: dict,
    input_path: Path,
    video_properties: dict,
    processed_frames: int,
) -> dict:
    """Build metadata describing the input video and extraction run."""
    return {
        "input_video": str(input_path),
        "provenance": config["data"]["provenance"],
        "source": config["data"]["source"],
        "original_segment_start_seconds": (
            config["data"]["original_segment_start_seconds"]
        ),
        "original_segment_end_seconds": (
            config["data"]["original_segment_end_seconds"]
        ),
        "fps": video_properties["fps"],
        "width": video_properties["width"],
        "height": video_properties["height"],
        "reported_frame_count": (
            video_properties["reported_frame_count"]
        ),
        "processed_frame_count": processed_frames,
        "duration_seconds": video_properties["duration_seconds"],
        "processing_timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "pose_model": str(DEFAULT_MODEL_PATH),
        "coordinate_note": (
            "Normalized image landmarks and model-estimated "
            "world landmarks. World coordinates are not "
            "camera-calibrated measurements."
        ),
    }


def run_pose_estimation(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """Execute the complete Phase 3.3 pose-estimation pipeline."""
    config = load_config(config_path)

    input_path = resolve_project_path(
        config["data"]["input_video"]
    )

    output_dir = resolve_project_path(
        config["output"]["directory"]
    )

    model_path = DEFAULT_MODEL_PATH

    # ---------------------------------------------------------
    # 1. Validate inputs and prepare output directories
    # ---------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Open video and read metadata
    # ---------------------------------------------------------

    cap = open_video(input_path)
    video_properties = read_video_metadata(cap)

    fps = video_properties["fps"]
    width = video_properties["width"]
    height = video_properties["height"]

    output_video_path = (
        visualization_dir / "pose_overlay.mp4"
    )

    writer = None

    pose_rows = []
    frame_index = 0
    detected_frames = 0

    try:
        # -----------------------------------------------------
        # 3. Initialize annotated-video writer
        # -----------------------------------------------------

        writer = create_video_writer(
            output_video_path,
            fps,
            width,
            height,
        )

        # -----------------------------------------------------
        # 4. Initialize MediaPipe and process every frame
        # -----------------------------------------------------

        with create_pose_landmarker(
            model_path,
            config["pose"],
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                timestamp_ms = round(
                    frame_index * 1000.0 / fps
                )

                mp_image = create_mp_image(frame)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Extract landmark rows.
                frame_rows = extract_landmark_rows(
                    result=result,
                    frame_index=frame_index,
                    fps=fps,
                    timestamp_ms=timestamp_ms,
                )

                if frame_rows:
                    detected_frames += 1
                    pose_rows.extend(frame_rows)

                    annotated = draw_pose_overlay(
                        frame,
                        result.pose_landmarks,
                    )
                else:
                    annotated = frame.copy()

                writer.write(annotated)

                frame_index += 1

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    # ---------------------------------------------------------
    # 5. Construct and save landmark dataset
    # ---------------------------------------------------------

    landmarks_df = pd.DataFrame(pose_rows)

    save_dataframe_csv(
        landmarks_df,
        output_dir / "pose_landmarks.csv",
    )

    processed_frames = frame_index

    # ---------------------------------------------------------
    # 6. Generate extraction report
    # ---------------------------------------------------------

    extraction_report = build_extraction_report(
        landmarks_df=landmarks_df,
        processed_frames=processed_frames,
        detected_frames=detected_frames,
    )

    metadata = build_video_metadata(
        config=config,
        input_path=input_path,
        video_properties=video_properties,
        processed_frames=processed_frames,
    )

    # ---------------------------------------------------------
    # 7. Save metadata and report
    # ---------------------------------------------------------

    save_json(
        metadata,
        output_dir / "video_metadata.json",
    )

    save_json(
        extraction_report,
        output_dir / "extraction_report.json",
    )

    # ---------------------------------------------------------
    # 8. Print execution summary
    # ---------------------------------------------------------

    print("=" * 65)
    print("TALA - PHASE 3.3: POSE EXTRACTION")
    print("=" * 65)
    print(f"Processed frames: {processed_frames}")
    print(f"Detected frames: {detected_frames}")
    print(
        f"Detection rate: "
        f"{extraction_report['detection_rate']:.2%}"
    )
    print(f"Landmark rows: {len(landmarks_df)}")
    print(f"FPS: {fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output_dir}")

    return {
        "metadata": metadata,
        "report": extraction_report,
        "landmarks": landmarks_df,
        "output_directory": output_dir,
    }


def main() -> None:
    """Command-line entry point."""
    run_pose_estimation()
