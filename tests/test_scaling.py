import numpy as np
import pytest

from tala.kinematic_reconstruction.scaling import (
    estimate_body_scale_factors,
)


class _Vector:
    def __init__(self, values):
        self.values = values

    def get(self, index):
        return self.values[index]


class _Marker:
    def __init__(self, position):
        self.position = position

    def getLocationInGround(self, _state):
        return _Vector(self.position)


class _MarkerSet:
    def __init__(self, markers):
        self.markers = markers

    def get(self, name):
        return _Marker(self.markers[name])


class _Model:
    def __init__(self):
        self.markers = _MarkerSet(
            {
                "LSJC": (0.0, 1.0, -0.1),
                "RSJC": (0.0, 1.0, 0.1),
                "LHJC": (0.0, 0.0, -0.1),
                "RHJC": (0.0, 0.0, 0.1),
            }
        )

    def getMarkerSet(self):
        return self.markers


def _points(hip_half_width):
    return {
        "left_shoulder": np.tile((0.0, 1.0, 0.0), (10, 1)),
        "right_shoulder": np.tile((0.0, 1.0, 0.0), (10, 1)),
        "left_hip": np.tile((0.0, 0.0, -hip_half_width), (10, 1)),
        "right_hip": np.tile((0.0, 0.0, hip_half_width), (10, 1)),
    }


def _segment_ratios():
    return {
        "left_shoulder_to_left_elbow": 1.0,
        "right_hip_to_right_knee": 1.0,
        "left_hip_to_left_knee": 1.0,
        "right_knee_to_right_ankle": 1.0,
        "left_knee_to_left_ankle": 1.0,
    }


def test_extreme_pelvis_scale_uses_median_segment_scale():
    with pytest.warns(RuntimeWarning, match="Pelvis width scale estimate"):
        factors = estimate_body_scale_factors(
            _points(hip_half_width=0.156),
            _Model(),
            None,
            pose_scale=1.0,
            segment_scale_ratios=_segment_ratios(),
        )

    assert factors["pelvis"] == pytest.approx(1.0)


def test_consistent_pelvis_scale_is_preserved():
    factors = estimate_body_scale_factors(
        _points(hip_half_width=0.11),
        _Model(),
        None,
        pose_scale=1.0,
        segment_scale_ratios=_segment_ratios(),
    )

    assert factors["pelvis"] == pytest.approx(1.1)
