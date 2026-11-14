"""MediaPipe-to-OpenSim marker mapping and IK target preparation."""

from __future__ import annotations

import numpy as np
import opensim as osim


DIRECT_MARKERS = {
    "RSJC": ("right_shoulder", 2.0),
    "LSJC": ("left_shoulder", 2.0),
    "REJC": ("right_elbow", 2.0),
    "LEJC": ("left_elbow", 2.0),
    "IK_WRIST_R": ("right_wrist", 1.5),
    "IK_WRIST_L": ("left_wrist", 1.5),
    "RHJC": ("right_hip", 2.0),
    "LHJC": ("left_hip", 2.0),
    "RKJC": ("right_knee", 2.0),
    "LKJC": ("left_knee", 2.0),
    "RAJC": ("right_ankle", 1.5),
    "LAJC": ("left_ankle", 1.5),
    "RTOE": ("right_foot_index", 0.75),
    "LTOE": ("left_foot_index", 0.75),
}

UNOBSERVED_ELBOW_FLEXION_DEGREES = 15.0


def add_virtual_wrist_markers(model) -> None:
    """Attach virtual wrist markers to the model hand origins."""
    for side in ("r", "l"):
        marker = osim.Marker()

        marker.setName(f"IK_WRIST_{side.upper()}")
        marker.setParentFrame(
            model.getBodySet().get(f"hand_{side}")
        )
        marker.set_location(osim.Vec3(0, 0, 0))

        model.addMarker(marker)


def create_marker_targets(model_points):
    """Build marker target arrays and corresponding IK weights."""
    targets = {}
    weights = {}
    skipped = {}

    for marker_name, (landmark_name, weight) in DIRECT_MARKERS.items():
        target = model_points[landmark_name]

        if not np.isfinite(target).all(axis=1).any():
            skipped[marker_name] = {
                "pose_landmark": landmark_name,
                "reason": (
                    "No samples met the visibility/presence threshold."
                ),
            }
            continue

        targets[marker_name] = target
        weights[marker_name] = weight

    return targets, weights, skipped


def neutralize_unobserved_limbs(
    model,
    valid_counts: dict,
):
    """Create constant neutral-coordinate tasks for unobserved limbs."""
    coordinate_set = model.getCoordinateSet()

    targets = {}
    neutralized_limbs = {}

    for side in ("right", "left"):
        suffix = "r" if side == "right" else "l"

        groups = {}

        elbow_has_samples = (
            valid_counts[f"{side}_elbow"] > 0
        )

        wrist_has_samples = (
            valid_counts[f"{side}_wrist"] > 0
        )

        if not elbow_has_samples and not wrist_has_samples:
            groups[f"{side}_arm"] = [
                f"arm_flex_{suffix}",
                f"arm_add_{suffix}",
                f"arm_rot_{suffix}",
                f"elbow_flex_{suffix}",
                f"pro_sup_{suffix}",
            ]

        elif not elbow_has_samples or not wrist_has_samples:
            groups[f"{side}_arm_distal"] = [
                f"elbow_flex_{suffix}",
                f"pro_sup_{suffix}",
            ]

        hip_has_samples = (
            valid_counts[f"{side}_hip"] > 0
        )

        distal_leg_has_samples = any(
            valid_counts[f"{side}_{landmark}"] > 0
            for landmark in (
                "knee",
                "ankle",
                "foot_index",
            )
        )

        if not hip_has_samples and not distal_leg_has_samples:
            groups[f"{side}_leg"] = [
                f"hip_flexion_{suffix}",
                f"hip_adduction_{suffix}",
                f"hip_rotation_{suffix}",
                f"knee_angle_{suffix}",
                f"knee_angle_{suffix}_beta",
                f"ankle_angle_{suffix}",
            ]

        elif hip_has_samples and not distal_leg_has_samples:
            groups[f"{side}_leg_distal"] = [
                f"knee_angle_{suffix}",
                f"knee_angle_{suffix}_beta",
                f"ankle_angle_{suffix}",
            ]

        for limb_name, coordinate_names in groups.items():
            limb_targets = {}

            for coordinate_name in coordinate_names:
                if not coordinate_set.contains(coordinate_name):
                    raise ValueError(
                        "Model is missing the coordinate needed to "
                        f"neutralize the unobserved {limb_name}: "
                        f"{coordinate_name}"
                    )

                coordinate = coordinate_set.get(coordinate_name)

                value = float(
                    coordinate.getDefaultValue()
                )

                if coordinate_name == f"elbow_flex_{suffix}":
                    value = float(
                        np.deg2rad(
                            UNOBSERVED_ELBOW_FLEXION_DEGREES
                        )
                    )

                if not (
                    float(coordinate.getRangeMin())
                    <= value
                    <= float(coordinate.getRangeMax())
                ):
                    raise ValueError(
                        f"Neutral value {value} is outside the range "
                        f"for {coordinate_name}."
                    )

                targets[coordinate_name] = value
                limb_targets[coordinate_name] = value

            neutralized_limbs[limb_name] = limb_targets

    return targets, neutralized_limbs