"""End-to-end Phase 3.10 full-body inverse-kinematics pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import opensim as osim

from tala.kinematic_reconstruction.inputs import (
    DIRECT_MARKERS,
    load_inputs,
)
from tala.kinematic_reconstruction.scaling import (
    estimate_body_scale_factors,
    estimate_pose_scale,
    scale_model,
    write_scaled_model,
)
from tala.kinematic_reconstruction.transforms import (
    estimate_pelvis_translation,
    transform_pose_to_model,
)
from tala.kinematic_reconstruction.marker_mapping import (
    add_virtual_wrist_markers,
    create_marker_targets,
    neutralize_unobserved_limbs,
)
from tala.kinematic_reconstruction.inverse_kinematics import (
    read_motion_file,
    run_inverse_kinematics,
    write_coordinate_csv,
    write_trc,
)
from tala.kinematic_reconstruction.diagnostics import (
    compute_marker_fitting_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "opensim"
    / "RajagopalLaiUhlrich2023.osim"
)

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pose_estimation"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "kinematic_reconstruction"
)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_report(
    *,
    model_path,
    scaled_model_path,
    landmarks_path,
    metadata_path,
    frame_index,
    timestamps,
    fps,
    timestamp_source,
    model,
    in_degrees,
    weights,
    valid_counts,
    skipped_markers,
    neutralized_limbs,
    pose_scale,
    segment_scale_ratios,
    body_scale_factors,
    root_report,
    valid_targets_per_frame,
    coordinate_diagnostics,
    marker_fitting_diagnostics,
    motion_path,
    coordinates_path,
    marker_fitting_errors_path,
    trc_path,
    report_path,
):
    """Assemble the Phase 3.10 execution and diagnostic report."""
    coordinate_set = model.getCoordinateSet()

    coordinate_names = [
        coordinate_set.get(i).getName()
        for i in range(coordinate_set.getSize())
    ]

    report = {
        "stage": "phase3_10_full_body_inverse_kinematics",
        "status": "motion_file_generated",
        "method": (
            "OpenSim InverseKinematicsTool fitting model markers "
            "to MediaPipe pose landmarks."
        ),
        "source_model": str(model_path),
        "scaled_model": str(scaled_model_path),
        "input_landmarks": str(landmarks_path),
        "input_metadata": str(metadata_path),
        "frame_count": int(len(frame_index)),
        "duration_seconds": float(
            timestamps[-1] - timestamps[0]
        ),
        "sampling_rate_hz": fps,
        "timestamp_source": timestamp_source,
        "coordinate_count": len(coordinate_names),
        "coordinates": coordinate_names,
        "motion_angles_in_degrees": in_degrees,
        "marker_mapping": {
            marker: {
                "pose_landmark": DIRECT_MARKERS[marker][0],
                "weight": weight,
                "valid_source_frames": valid_counts[
                    DIRECT_MARKERS[marker][0]
                ],
                "valid_source_frame_percentage": (
                    100.0
                    * valid_counts[DIRECT_MARKERS[marker][0]]
                    / len(frame_index)
                ),
            }
            for marker, weight in weights.items()
            if marker in DIRECT_MARKERS
        },
        "landmark_coverage": {
            landmark: {
                "valid_source_frames": count,
                "valid_source_frame_percentage": (
                    100.0 * count / len(frame_index)
                ),
            }
            for landmark, count in valid_counts.items()
        },
        "skipped_markers": skipped_markers,
        "neutralized_unobserved_limbs": {
            limb: {
                "assumption": (
                    "Assumed neutral posture, not recovered movement."
                ),
                "coordinate_targets": {
                    name: {
                        "value_radians": value,
                        "value_degrees": float(
                            np.degrees(value)
                        ),
                    }
                    for name, value in targets.items()
                },
            }
            for limb, targets in neutralized_limbs.items()
        },
        "neutralization_scope": (
            "Coordinate tasks are constant over the full motion. "
            "They are not switched on only during missing frames."
        ),
        "pose_to_model_scale": pose_scale,
        "segment_scale_ratios": segment_scale_ratios,
        "applied_body_scale_factors": body_scale_factors,
        "root_translation_estimation": root_report,
        "valid_ik_markers_per_frame": {
            "minimum": int(valid_targets_per_frame.min()),
            "maximum": int(valid_targets_per_frame.max()),
        },
        "coordinate_diagnostics": coordinate_diagnostics,
        "marker_fitting_diagnostics": marker_fitting_diagnostics,
        "limitations": [
            "MediaPipe world landmarks are estimated and are not camera-calibrated motion capture.",
            "Pose scale and body segment scale factors are estimated from monocular landmarks; segment proportions may be biased by pose estimation errors.",
            "Pelvis camera-depth translation is not observable from hip-centered world landmarks and is held at zero.",
            "Virtual wrist markers are attached to the model hand origins; the model wrist flexion and deviation coordinates are locked.",
            "Markers with no samples above the confidence threshold are omitted; motion of an unobserved limb cannot be recovered reliably.",
            "Single-camera depth ambiguity can affect pelvis and limb rotations.",
            "Marker fit is kinematic only; no inverse dynamics or joint loads are computed.",
        ],
        "output_files": {
            "motion": str(motion_path),
            "scaled_model": str(scaled_model_path),
            "mapped_coordinates": str(coordinates_path),
            "marker_fitting_errors": str(marker_fitting_errors_path),
            "marker_targets": str(trc_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    return report


def run_kinematic_reconstruction(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
):
    """Run the complete Phase 3.10 full-body IK workflow."""
    input_dir = resolve_project_path(input_dir)
    output_dir = resolve_project_path(output_dir)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"OpenSim model not found: {MODEL_PATH}"
        )

    # ---------------------------------------------------------
    # 1. Load and preprocess Phase 3.3 inputs
    # ---------------------------------------------------------

    (
        points,
        image_points,
        sample_status,
        valid_counts,
        frame_index,
        timestamps,
        fps,
        image_width,
        image_height,
        timestamp_source,
        landmarks_path,
        metadata_path,
    ) = load_inputs(input_dir)

    # ---------------------------------------------------------
    # 2. Load OpenSim model and add virtual wrist markers
    # ---------------------------------------------------------

    model = osim.Model(str(MODEL_PATH))

    add_virtual_wrist_markers(model)

    state = model.initSystem()
    model.realizePosition(state)

    # ---------------------------------------------------------
    # 3. Estimate pose scale and body scale factors
    # ---------------------------------------------------------

    pose_scale, segment_scale_ratios = estimate_pose_scale(
        points,
        model,
        state,
    )

    body_scale_factors = estimate_body_scale_factors(
        points,
        model,
        state,
        pose_scale,
        segment_scale_ratios,
    )

    scale_model(
        model,
        state,
        body_scale_factors,
    )

    state = model.initSystem()
    model.realizePosition(state)

    # ---------------------------------------------------------
    # 4. Estimate root translation and transform landmarks
    # ---------------------------------------------------------

    root_translation, root_report = estimate_pelvis_translation(
        points,
        image_points,
        pose_scale,
        image_width,
        image_height,
    )

    model_points = transform_pose_to_model(
        points,
        pose_scale,
        root_translation,
    )

    # ---------------------------------------------------------
    # 5. Create marker targets and neutral coordinate tasks
    # ---------------------------------------------------------

    targets, weights, skipped_markers = create_marker_targets(
        model_points
    )

    if len(targets) < 6:
        raise ValueError(
            "Fewer than six IK markers have usable pose data. "
            f"Usable markers: {len(targets)}; skipped: "
            f"{sorted(skipped_markers)}."
        )

    neutral_coordinate_targets, neutralized_limbs = (
        neutralize_unobserved_limbs(
            model,
            valid_counts,
        )
    )

    valid_targets_per_frame = np.zeros(
        len(frame_index),
        dtype=int,
    )

    for target in targets.values():
        valid_targets_per_frame += (
            np.isfinite(target).all(axis=1)
        )

    if np.min(valid_targets_per_frame) < 6:
        raise ValueError(
            "At least one video frame has fewer than six "
            "usable IK markers."
        )

    # ---------------------------------------------------------
    # 6. Prepare output paths
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaled_model_path = output_dir / "scaled_model.osim"
    trc_path = output_dir / "pose_targets.trc"
    motion_path = output_dir / "opensim_motion.mot"
    coordinates_path = output_dir / "mapped_coordinates.csv"

    marker_fitting_errors_path = (
        output_dir / "marker_fitting_errors.csv"
    )

    report_path = output_dir / "ik_report.json"

    # ---------------------------------------------------------
    # 7. Write scaled model and TRC targets
    # ---------------------------------------------------------

    write_scaled_model(
        model,
        scaled_model_path,
        MODEL_PATH,
    )

    write_trc(
        targets,
        weights,
        timestamps,
        fps,
        trc_path,
    )

    # ---------------------------------------------------------
    # 8. Execute OpenSim inverse kinematics
    # ---------------------------------------------------------

    run_inverse_kinematics(
        model,
        weights,
        neutral_coordinate_targets,
        trc_path,
        motion_path,
        timestamps,
    )

    # ---------------------------------------------------------
    # 9. Read and validate OpenSim motion output
    # ---------------------------------------------------------

    motion_table, in_degrees = read_motion_file(
        motion_path
    )

    motion_times = motion_table[
        "time"
    ].to_numpy(dtype=float)

    if (
        len(motion_table) != len(timestamps)
        or not np.allclose(
            motion_times,
            timestamps,
            atol=1e-6,
        )
    ):
        raise ValueError(
            "OpenSim IK output times do not match "
            "the video frame times."
        )

    # ---------------------------------------------------------
    # 10. Export mapped coordinates
    # ---------------------------------------------------------

    coordinate_diagnostics = write_coordinate_csv(
        motion_table,
        in_degrees,
        model,
        frame_index,
        timestamps,
        coordinates_path,
    )

    # ---------------------------------------------------------
    # 11. Compute marker-fitting diagnostics
    # ---------------------------------------------------------

    diagnostic_model = osim.Model(
        str(scaled_model_path)
    )

    marker_fitting_diagnostics = (
        compute_marker_fitting_diagnostics(
            diagnostic_model,
            motion_table,
            in_degrees,
            targets,
            sample_status,
            frame_index,
            timestamps,
            marker_fitting_errors_path,
        )
    )

    # ---------------------------------------------------------
    # 12. Build and save report
    # ---------------------------------------------------------

    report = build_report(
        model_path=MODEL_PATH,
        scaled_model_path=scaled_model_path,
        landmarks_path=landmarks_path,
        metadata_path=metadata_path,
        frame_index=frame_index,
        timestamps=timestamps,
        fps=fps,
        timestamp_source=timestamp_source,
        model=model,
        in_degrees=in_degrees,
        weights=weights,
        valid_counts=valid_counts,
        skipped_markers=skipped_markers,
        neutralized_limbs=neutralized_limbs,
        pose_scale=pose_scale,
        segment_scale_ratios=segment_scale_ratios,
        body_scale_factors=body_scale_factors,
        root_report=root_report,
        valid_targets_per_frame=valid_targets_per_frame,
        coordinate_diagnostics=coordinate_diagnostics,
        marker_fitting_diagnostics=marker_fitting_diagnostics,
        motion_path=motion_path,
        coordinates_path=coordinates_path,
        marker_fitting_errors_path=marker_fitting_errors_path,
        trc_path=trc_path,
        report_path=report_path,
    )

    # ---------------------------------------------------------
    # 13. Print execution summary
    # ---------------------------------------------------------

    coordinate_names = report["coordinates"]

    print("=" * 72)
    print("PHASE 3.10 COMPLETE: FULL-BODY INVERSE KINEMATICS")
    print("=" * 72)
    print(f"Frames: {len(frame_index)}")
    print(f"Model/pose uniform scale: {pose_scale:.4f}")
    print(f"Scaled model: {scaled_model_path}")
    print(f"IK markers: {len(weights)}")

    if skipped_markers:
        print(
            "Skipped markers without confident samples: "
            f"{sorted(skipped_markers)}"
        )

    if neutralized_limbs:
        print(
            "Neutralized unobserved limbs: "
            f"{sorted(neutralized_limbs)}"
        )

    print(f"Coordinates: {len(coordinate_names)}")
    print(f"Motion file: {motion_path}")
    print(f"Mapped coordinates: {coordinates_path}")
    print(f"IK report: {report_path}")

    return report


def main(argv=None):
    """Command-line entry point for Phase 3.10."""
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenSim full-body IK on Phase 3.3 "
            "pose-estimation outputs."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing pose_landmarks.csv and "
            "video_metadata.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for the scaled model and IK outputs."
        ),
    )

    args = parser.parse_args(argv)

    run_kinematic_reconstruction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )