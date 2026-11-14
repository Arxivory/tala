"""Coordinate and marker-fitting diagnostics for OpenSim IK."""

from __future__ import annotations

import numpy as np
import opensim as osim
import pandas as pd

from tala.kinematic_reconstruction.inputs import DIRECT_MARKERS
from tala.kinematic_reconstruction.scaling import marker_position


def compute_marker_fitting_diagnostics(
    model,
    motion_table,
    in_degrees,
    targets,
    sample_status,
    frame_index,
    timestamps,
    output_path,
):
    """Calculate marker-fitting residuals over the reconstructed motion."""
    coordinate_set = model.getCoordinateSet()
    marker_set = model.getMarkerSet()

    state = model.initSystem()
    record_rows = []

    marker_targets = {}

    for marker_name, (landmark_name, _) in DIRECT_MARKERS.items():
        if not marker_set.contains(marker_name):
            raise ValueError(
                "Fitting diagnostic marker is not in the model: "
                f"{marker_name}"
            )

        marker_targets[marker_name] = (
            targets.get(
                marker_name,
                np.full((len(frame_index), 3), np.nan),
            ),
            landmark_name,
        )

    for row_index, (frame, timestamp) in enumerate(
        zip(frame_index, timestamps)
    ):
        for index in range(coordinate_set.getSize()):
            coordinate = coordinate_set.get(index)
            name = coordinate.getName()

            if name not in motion_table.columns:
                raise ValueError(
                    "Coordinate missing from IK motion while computing "
                    f"marker fit: {name}"
                )

            value = float(
                motion_table[name].iloc[row_index]
            )

            if (
                in_degrees
                and coordinate.getMotionType()
                == osim.Coordinate.Rotational
            ):
                value = float(np.radians(value))

            coordinate.setValue(state, value, False)

        model.assemble(state)
        model.realizePosition(state)

        for marker_name, (
            marker_target,
            landmark_name,
        ) in marker_targets.items():

            target_position = marker_target[row_index]

            fitted_position = marker_position(
                model,
                state,
                marker_name,
            )

            has_target = np.isfinite(
                target_position
            ).all()

            error = (
                float(
                    np.linalg.norm(
                        fitted_position - target_position
                    )
                )
                if has_target
                else np.nan
            )

            record_rows.append({
                "frame_index": int(frame),
                "time": float(timestamp),
                "marker": marker_name,
                "pose_landmark": landmark_name,
                "sample_status": sample_status[
                    landmark_name
                ][row_index],
                "target_x_m": (
                    float(target_position[0])
                    if has_target else np.nan
                ),
                "target_y_m": (
                    float(target_position[1])
                    if has_target else np.nan
                ),
                "target_z_m": (
                    float(target_position[2])
                    if has_target else np.nan
                ),
                "fitted_x_m": float(fitted_position[0]),
                "fitted_y_m": float(fitted_position[1]),
                "fitted_z_m": float(fitted_position[2]),
                "error_m": error,
            })

    errors = pd.DataFrame(record_rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        output_path,
        index=False,
        float_format="%.10f",
    )

    def error_metrics(values):
        finite = np.asarray(
            values,
            dtype=float,
        )

        finite = finite[np.isfinite(finite)]

        if finite.size == 0:
            return {
                "sample_count": 0,
                "rms_error_m": None,
                "mean_error_m": None,
                "maximum_error_m": None,
            }

        return {
            "sample_count": int(finite.size),
            "rms_error_m": float(
                np.sqrt(np.mean(np.square(finite)))
            ),
            "mean_error_m": float(np.mean(finite)),
            "maximum_error_m": float(np.max(finite)),
        }

    finite_errors = errors["error_m"].to_numpy(dtype=float)

    if not np.isfinite(finite_errors).any():
        raise ValueError(
            "No observed or interpolated marker targets are available "
            "for fitting diagnostics."
        )

    per_marker = {}

    for marker_name, (
        _,
        landmark_name,
    ) in marker_targets.items():

        marker_rows = errors[
            errors["marker"] == marker_name
        ]

        marker_errors = marker_rows[
            "error_m"
        ].to_numpy(dtype=float)

        observed_errors = marker_rows.loc[
            marker_rows["sample_status"] == "observed",
            "error_m",
        ].to_numpy(dtype=float)

        per_marker[marker_name] = {
            "pose_landmark": landmark_name,
            "sample_counts": {
                status: int(
                    marker_rows["sample_status"].eq(status).sum()
                )
                for status in (
                    "observed",
                    "interpolated",
                    "unavailable",
                )
            },
            "fitting": error_metrics(marker_errors),
            "observed_only_fitting": error_metrics(
                observed_errors
            ),
        }

    error_over_time = []

    for frame, frame_rows in errors.groupby(
        "frame_index",
        sort=True,
    ):
        frame_errors = frame_rows[
            "error_m"
        ].to_numpy(dtype=float)

        finite = frame_errors[
            np.isfinite(frame_errors)
        ]

        error_over_time.append({
            "frame_index": int(frame),
            "time_seconds": float(
                frame_rows["time"].iloc[0]
            ),
            "marker_count": int(finite.size),
            "mean_error_m": (
                float(np.mean(finite))
                if finite.size else None
            ),
            "rms_error_m": (
                float(np.sqrt(np.mean(np.square(finite))))
                if finite.size else None
            ),
            "maximum_error_m": (
                float(np.max(finite))
                if finite.size else None
            ),
        })

    status_counts = {
        status: int(
            errors["sample_status"].eq(status).sum()
        )
        for status in (
            "observed",
            "interpolated",
            "unavailable",
        )
    }

    return {
        "interpretation": (
            "Marker-fitting agreement with transformed pose targets; "
            "not a measure of anatomical accuracy."
        ),
        "overall_fitting": error_metrics(finite_errors),
        "observed_only_fitting": error_metrics(
            errors.loc[
                errors["sample_status"] == "observed",
                "error_m",
            ].to_numpy(dtype=float)
        ),
        "sample_counts": status_counts,
        "per_marker": per_marker,
        "error_over_time": error_over_time,
        "csv_path": str(output_path),
    }