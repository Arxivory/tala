"""Pose-to-model scale estimation and OpenSim model scaling."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import opensim as osim

SCALE_SEGMENTS = [
    ("right_shoulder", "right_elbow", "RSJC", "REJC"),
    ("left_shoulder", "left_elbow", "LSJC", "LEJC"),
    ("right_elbow", "right_wrist", "REJC", "IK_WRIST_R"),
    ("left_elbow", "left_wrist", "LEJC", "IK_WRIST_L"),
    ("right_hip", "right_knee", "RHJC", "RKJC"),
    ("left_hip", "left_knee", "LHJC", "LKJC"),
    ("right_knee", "right_ankle", "RKJC", "RAJC"),
    ("left_knee", "left_ankle", "LKJC", "LAJC"),
    ("right_ankle", "right_foot_index", "RAJC", "RTOE"),
    ("left_ankle", "left_foot_index", "LAJC", "LTOE"),
]


def marker_position(
    model,
    state,
    marker_name: str,
) -> np.ndarray:
    """Get a model marker's ground position as a NumPy vector."""
    marker = model.getMarkerSet().get(marker_name)
    position = marker.getLocationInGround(state)

    return np.array(
        [position.get(i) for i in range(3)],
        dtype=float,
    )


def estimate_pose_scale(
    points,
    model,
    state,
):
    """Estimate uniform pose-to-model scale from segment lengths."""
    ratios = {}

    for start, end, model_start, model_end in SCALE_SEGMENTS:
        pose_lengths = np.linalg.norm(
            points[start] - points[end],
            axis=1,
        )

        pose_lengths = pose_lengths[
            np.isfinite(pose_lengths)
            & (pose_lengths > 1e-4)
        ]

        if len(pose_lengths) < 10:
            continue

        model_length = np.linalg.norm(
            marker_position(model, state, model_start)
            - marker_position(model, state, model_end)
        )

        ratios[f"{start}_to_{end}"] = float(
            model_length / np.median(pose_lengths)
        )

    if len(ratios) < 4:
        raise ValueError(
            "Insufficient valid segment lengths to align pose and model."
        )

    scale = float(np.median(list(ratios.values())))

    if not np.isfinite(scale) or not 0.5 <= scale <= 3.0:
        raise ValueError(f"Implausible pose-to-model scale: {scale}")

    return scale, ratios


def estimate_body_scale_factors(
    points,
    model,
    state,
    pose_scale: float,
    segment_scale_ratios: dict,
):
    """Estimate individual OpenSim body segment scale factors.

    An implausible pelvis-width estimate falls back to the median scale of
    the other measured segments because hip landmarks are especially noisy.
    """
    body_factors = {}

    for start, end, model_start, model_end in SCALE_SEGMENTS:
        key = f"{start}_to_{end}"

        if key not in segment_scale_ratios:
            continue

        body_length_factor = (
            pose_scale / segment_scale_ratios[key]
        )

        if start.endswith("_shoulder"):
            side = start.split("_")[0]
            body_names = [
                f"humerus_{'r' if side == 'right' else 'l'}"
            ]

        elif start.endswith("_elbow"):
            side = start.split("_")[0]
            suffix = "r" if side == "right" else "l"
            body_names = [
                f"ulna_{suffix}",
                f"radius_{suffix}",
            ]

        elif start.endswith("_hip"):
            side = start.split("_")[0]
            body_names = [
                f"femur_{'r' if side == 'right' else 'l'}"
            ]

        elif start.endswith("_knee"):
            side = start.split("_")[0]
            body_names = [
                f"tibia_{'r' if side == 'right' else 'l'}"
            ]

        else:
            side = start.split("_")[0]
            body_names = [
                f"calcn_{'r' if side == 'right' else 'l'}"
            ]

        for body_name in body_names:
            body_factors[body_name] = body_length_factor

    shoulder_pose = (
        points["left_shoulder"] + points["right_shoulder"]
    ) / 2.0

    hip_pose = (
        points["left_hip"] + points["right_hip"]
    ) / 2.0

    torso_lengths = np.linalg.norm(
        shoulder_pose - hip_pose,
        axis=1,
    )

    torso_lengths = torso_lengths[
        np.isfinite(torso_lengths)
    ]

    shoulder_model = (
        marker_position(model, state, "LSJC")
        + marker_position(model, state, "RSJC")
    ) / 2.0

    hip_model = (
        marker_position(model, state, "LHJC")
        + marker_position(model, state, "RHJC")
    ) / 2.0

    torso_model_length = float(
        np.linalg.norm(shoulder_model - hip_model)
    )

    if len(torso_lengths) >= 10 and torso_model_length > 1e-4:
        body_factors["torso"] = (
            pose_scale
            * float(np.median(torso_lengths))
            / torso_model_length
        )

    hip_widths = np.linalg.norm(
        points["left_hip"] - points["right_hip"],
        axis=1,
    )

    hip_widths = hip_widths[
        np.isfinite(hip_widths)
    ]

    model_hip_width = float(
        np.linalg.norm(
            marker_position(model, state, "LHJC")
            - marker_position(model, state, "RHJC")
        )
    )

    if len(hip_widths) >= 10 and model_hip_width > 1e-4:
        body_factors["pelvis"] = (
            pose_scale
            * float(np.median(hip_widths))
            / model_hip_width
        )

    if "pelvis" in body_factors:
        segment_factors = [
            factor
            for name, factor in body_factors.items()
            if name != "pelvis"
        ]

        if segment_factors:
            reference_scale = float(np.median(segment_factors))
            pelvis_scale = body_factors["pelvis"]

            if not (
                reference_scale / 1.25
                <= pelvis_scale
                <= reference_scale * 1.25
            ):
                warnings.warn(
                    "Pelvis width scale estimate "
                    f"({pelvis_scale:.3f}) is inconsistent with other "
                    f"segments ({reference_scale:.3f}); using the median "
                    "segment scale instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                body_factors["pelvis"] = reference_scale

    mirrored_bodies = [
        ("humerus_r", "humerus_l"),
        ("ulna_r", "ulna_l"),
        ("radius_r", "radius_l"),
        ("femur_r", "femur_l"),
        ("tibia_r", "tibia_l"),
        ("calcn_r", "calcn_l"),
    ]

    for right_body, left_body in mirrored_bodies:
        if (
            right_body not in body_factors
            and left_body in body_factors
        ):
            body_factors[right_body] = body_factors[left_body]

        elif (
            left_body not in body_factors
            and right_body in body_factors
        ):
            body_factors[left_body] = body_factors[right_body]

    invalid = {
        name: factor
        for name, factor in body_factors.items()
        if (
            not np.isfinite(factor)
            or not 0.4 <= factor <= 2.0
        )
    }

    if invalid:
        raise ValueError(
            f"Implausible body segment scale factors: {invalid}"
        )

    return body_factors


def scale_model(
    model,
    state,
    body_factors: dict,
) -> None:
    """Apply segment scale factors to an OpenSim model."""
    scale_set = osim.ScaleSet()

    for body_name, factor in body_factors.items():
        scale = osim.Scale()

        scale.setName(body_name)
        scale.setSegmentName(body_name)

        if body_name.startswith("calcn_"):
            scale.setScaleFactors(osim.Vec3(factor, 1, 1))

        elif body_name == "pelvis":
            scale.setScaleFactors(osim.Vec3(1, 1, factor))

        else:
            scale.setScaleFactors(osim.Vec3(1, factor, 1))

        scale.setApply(True)
        scale_set.cloneAndAppend(scale)

    if not model.scale(state, scale_set, False):
        raise RuntimeError("OpenSim failed to scale the model.")


def write_scaled_model(
    model,
    output_path: str | Path,
    source_model_path: str | Path,
) -> None:
    """Write the scaled model and resolve relative geometry paths."""
    output_path = Path(output_path)
    source_model_path = Path(source_model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.printToXML(str(output_path))

    tree = ET.parse(output_path)

    geometry_directories = [
        source_model_path.parent,
        source_model_path.parent / "Geometry",
    ]

    for element in tree.getroot().iter():
        value = (element.text or "").strip()

        if Path(value).suffix.lower() not in {
            ".vtp", ".stl", ".obj", ".ply"
        }:
            continue

        mesh_path = Path(value)

        if mesh_path.is_absolute():
            continue

        resolved = next(
            (
                (directory / mesh_path).resolve()
                for directory in geometry_directories
                if (directory / mesh_path).exists()
            ),
            None,
        )

        if resolved is not None:
            element.text = str(resolved)

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )