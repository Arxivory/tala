"""OpenSim inverse kinematics execution and motion-file handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import opensim as osim
import pandas as pd


NEUTRAL_COORDINATE_TASK_WEIGHT = 10.0


def write_trc(
    targets,
    weights,
    timestamps,
    fps,
    output_path: Path,
):
    """Write transformed marker trajectories in OpenSim TRC format."""
    marker_names = list(targets)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(
            f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}\n"
        )

        file.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )

        file.write(
            f"{fps:.8f}\t{fps:.8f}\t{len(timestamps)}\t"
            f"{len(marker_names)}\tm\t{fps:.8f}\t1\t"
            f"{len(timestamps)}\n"
        )

        file.write(
            "Frame#\tTime\t"
            + "\t\t\t".join(marker_names)
            + "\n"
        )

        axis_names = "\t".join(
            axis
            for index in range(len(marker_names))
            for axis in (
                f"X{index + 1}",
                f"Y{index + 1}",
                f"Z{index + 1}",
            )
        )

        file.write("\t\t" + axis_names + "\n")

        for frame_index, timestamp in enumerate(timestamps):
            row = [
                str(frame_index + 1),
                f"{timestamp:.10f}",
            ]

            for marker_name in marker_names:
                coordinates = targets[marker_name][frame_index]

                row.extend(
                    (
                        f"{value:.8f}"
                        if np.isfinite(value)
                        else "nan"
                    )
                    for value in coordinates
                )

            file.write("\t".join(row) + "\n")


def run_inverse_kinematics(
    model,
    marker_weights,
    neutral_coordinate_targets,
    trc_path: Path,
    mot_path: Path,
    timestamps,
):
    """Run OpenSim InverseKinematicsTool on the prepared TRC targets."""
    tasks = osim.IKTaskSet()
    model_markers = model.getMarkerSet()

    for marker_name, weight in marker_weights.items():
        if not model_markers.contains(marker_name):
            raise ValueError(
                f"IK marker is not present in the model: {marker_name}"
            )

        task = osim.IKMarkerTask()

        task.setName(marker_name)
        task.setApply(True)
        task.setWeight(float(weight))

        tasks.cloneAndAppend(task)

    coordinate_set = model.getCoordinateSet()

    for coordinate_name, value in neutral_coordinate_targets.items():
        if not coordinate_set.contains(coordinate_name):
            raise ValueError(
                "IK neutral-coordinate task is not in the model: "
                f"{coordinate_name}"
            )

        task = osim.IKCoordinateTask()

        task.setName(coordinate_name)
        task.setApply(True)
        task.setValueType(
            osim.IKCoordinateTask.ManualValue
        )
        task.setValue(float(value))
        task.setWeight(NEUTRAL_COORDINATE_TASK_WEIGHT)

        tasks.cloneAndAppend(task)

    tool = osim.InverseKinematicsTool()

    tool.setModel(model)
    tool.setMarkerDataFileName(str(trc_path))
    tool.setOutputMotionFileName(str(mot_path))
    tool.setStartTime(float(timestamps[0]))
    tool.setEndTime(float(timestamps[-1]))
    tool.set_IKTaskSet(tasks)

    if not tool.run():
        raise RuntimeError(
            "OpenSim InverseKinematicsTool failed."
        )

    if not mot_path.exists():
        raise FileNotFoundError(
            "OpenSim IK did not produce the motion file: "
            f"{mot_path}"
        )


def read_motion_file(path: str | Path):
    """Read an OpenSim MOT file and detect its angular unit convention."""
    path = Path(path)

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    end_header = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "endheader"
        ),
        None,
    )

    if end_header is None:
        raise ValueError(
            f"Missing endheader in OpenSim motion: {path}"
        )

    in_degrees = any(
        line.strip().lower() == "indegrees=yes"
        for line in lines[:end_header]
    )

    table = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=end_header + 1,
        engine="python",
    )

    if "time" not in table.columns:
        raise ValueError(
            "OpenSim IK motion has no time column."
        )

    return table, in_degrees


def write_coordinate_csv(
    table,
    in_degrees: bool,
    model,
    frame_index,
    timestamps,
    output_path: Path,
):
    """Write model coordinates in radians/meters with range diagnostics."""
    coordinate_set = model.getCoordinateSet()

    output = pd.DataFrame({
        "frame_index": frame_index,
        "time": timestamps,
    })

    diagnostics = {}

    for index in range(coordinate_set.getSize()):
        coordinate = coordinate_set.get(index)
        name = coordinate.getName()

        if name not in table.columns:
            raise ValueError(
                f"Coordinate missing from IK motion output: {name}"
            )

        values = table[name].to_numpy(dtype=float)

        rotational = (
            coordinate.getMotionType()
            == osim.Coordinate.Rotational
        )

        if in_degrees and rotational:
            values = np.radians(values)

        output[name] = values

        lower = float(coordinate.getRangeMin())
        upper = float(coordinate.getRangeMax())

        minimum = float(np.min(values))
        maximum = float(np.max(values))

        if (
            minimum < lower - 1e-5
            or maximum > upper + 1e-5
        ):
            raise ValueError(
                f"{name} exceeds the model range. "
                f"Observed [{minimum:.6f}, {maximum:.6f}], "
                f"allowed [{lower:.6f}, {upper:.6f}]."
            )

        diagnostics[name] = {
            "minimum": minimum,
            "maximum": maximum,
            "model_range": [lower, upper],
            "units": (
                "radians" if rotational else "meters"
            ),
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
        float_format="%.10f",
    )

    return diagnostics