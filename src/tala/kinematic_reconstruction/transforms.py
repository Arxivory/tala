"""Coordinate transformation and pelvis translation estimation."""

from __future__ import annotations

import numpy as np

from tala.kinematic_reconstruction.preprocessing import (
    prepare_landmark_samples,
)


PELVIS_TY_DEFAULT = 0.94


def finite_pair_average(first, second):
    """Average two arrays while ignoring unavailable values."""
    pair = np.stack(
        [first, second],
        axis=0,
    ).astype(float)

    valid = np.isfinite(pair)
    counts = valid.sum(axis=0)

    return np.divide(
        np.nansum(pair, axis=0),
        counts,
        out=np.full(
            pair.shape[1:],
            np.nan,
            dtype=float,
        ),
        where=counts > 0,
    )


def estimate_pelvis_translation(
    points,
    image_points,
    pose_scale: float,
    image_width: int,
    image_height: int,
):
    """Estimate pelvis translation from image-space hip movement."""
    shoulder_world_y = finite_pair_average(
        points["left_shoulder"][:, 1],
        points["right_shoulder"][:, 1],
    )

    ankle_world_y = finite_pair_average(
        points["left_ankle"][:, 1],
        points["right_ankle"][:, 1],
    )

    shoulder_image_y = finite_pair_average(
        image_points["left_shoulder"][:, 1],
        image_points["right_shoulder"][:, 1],
    )

    ankle_image_y = finite_pair_average(
        image_points["left_ankle"][:, 1],
        image_points["right_ankle"][:, 1],
    )

    valid_scale = (
        np.isfinite(shoulder_world_y)
        & np.isfinite(ankle_world_y)
        & np.isfinite(shoulder_image_y)
        & np.isfinite(ankle_image_y)
    )

    image_height_normalized = np.abs(
        shoulder_image_y[valid_scale]
        - ankle_image_y[valid_scale]
    )

    world_height = (
        np.abs(
            shoulder_world_y[valid_scale]
            - ankle_world_y[valid_scale]
        )
        * pose_scale
    )

    usable = image_height_normalized > 1e-6

    if usable.sum() < 10:
        raise ValueError(
            "Insufficient valid body-scale observations "
            "for pelvis tracking."
        )

    meters_per_normalized_y = float(
        np.median(
            world_height[usable]
            / image_height_normalized[usable]
        )
    )

    meters_per_normalized_x = (
        meters_per_normalized_y
        * image_width
        / image_height
    )

    hip_image_x = finite_pair_average(
        image_points["left_hip"][:, 0],
        image_points["right_hip"][:, 0],
    )

    hip_image_y = finite_pair_average(
        image_points["left_hip"][:, 1],
        image_points["right_hip"][:, 1],
    )

    hip_image_x = prepare_landmark_samples(
        hip_image_x[:, None],
        "image-space hip midpoint x",
    )[:, 0]

    hip_image_y = prepare_landmark_samples(
        hip_image_y[:, None],
        "image-space hip midpoint y",
    )[:, 0]

    pelvis_ty = (
        PELVIS_TY_DEFAULT
        - (hip_image_y - hip_image_y[0])
        * meters_per_normalized_y
    )

    pelvis_tz = (
        -(hip_image_x - hip_image_x[0])
        * meters_per_normalized_x
    )

    root_translation = np.column_stack([
        np.zeros(len(pelvis_ty)),
        pelvis_ty,
        pelvis_tz,
    ])

    report = {
        "reference_pelvis_ty_m": PELVIS_TY_DEFAULT,
        "meters_per_normalized_y": meters_per_normalized_y,
        "meters_per_normalized_x": meters_per_normalized_x,
        "pelvis_tx": (
            "Held at zero; hip-centered MediaPipe world landmarks "
            "do not provide camera-depth translation."
        ),
    }

    return root_translation, report


def transform_pose_to_model(
    points,
    pose_scale: float,
    root_translation: np.ndarray,
):
    """Transform MediaPipe world coordinates into model coordinates."""
    transformed = {}

    for name, coordinates in points.items():
        coordinates = np.asarray(
            coordinates,
            dtype=float,
        )

        model_axes = np.column_stack([
            -coordinates[:, 2],
            -coordinates[:, 1],
            -coordinates[:, 0],
        ])

        transformed[name] = (
            pose_scale * model_axes + root_translation
        )

    return transformed