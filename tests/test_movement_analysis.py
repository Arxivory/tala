
"""Focused tests for TALA movement analysis."""

import numpy as np

from tala.movement_analysis import (
    get_profile,
    segment_movement,
    count_repetitions,
    evaluate_completeness,
)


def analyze_squat(signal):
    """Run the movement-analysis pipeline on a synthetic squat signal."""

    profile = get_profile("squat")

    signal = np.asarray(signal, dtype=float)

    segmentation = segment_movement(
        signal=signal,
        profile=profile,
        frame_indices=np.arange(len(signal)),
        timestamps=np.arange(len(signal), dtype=float) / 30.0,
        smoothing_window=1,
        min_phase_frames=1,
        min_prominence=0.01,
    )

    repetitions = count_repetitions(
        segmentation,
        profile,
    )

    completeness = evaluate_completeness(
        signal,
        segmentation,
        profile,
    )

    return segmentation, repetitions, completeness


def test_complete_squat_counts_one_repetition():
    signal = [
        1.50,
        1.35,
        1.10,
        0.90,
        0.80,
        0.90,
        1.10,
        1.35,
        1.50,
    ]

    segmentation, repetitions, completeness = analyze_squat(signal)

    assert len(segmentation.segments) >= 2
    assert repetitions["repetition_count"] == 1
    assert completeness["status"] == "complete"
    assert completeness["is_complete"] is True


def test_incomplete_squat_does_not_count_repetition():
    signal = [
        1.50,
        1.35,
        1.10,
        0.90,
        0.80,
        0.90,
        1.00,
    ]

    segmentation, repetitions, completeness = analyze_squat(signal)

    assert len(segmentation.segments) >= 2
    assert repetitions["repetition_count"] == 0
    assert completeness["status"] == "incomplete"
    assert completeness["is_complete"] is False


def test_flat_signal_has_unknown_completeness():
    signal = np.ones(20)

    segmentation, _, completeness = analyze_squat(signal)

    assert segmentation.signal_range == 0
    assert completeness["status"] == "unknown"
    assert completeness["is_complete"] is None


def test_non_cyclic_profile_does_not_count_repetitions():
    profile = get_profile("plank")

    signal = np.linspace(0.0, 1.0, 20)

    segmentation = segment_movement(
        signal=signal,
        profile=profile,
        smoothing_window=1,
    )

    repetitions = count_repetitions(
        segmentation,
        profile,
    )

    completeness = evaluate_completeness(
        signal,
        segmentation,
        profile,
    )

    assert repetitions["status"] == "not_applicable"
    assert repetitions["repetition_count"] is None
    assert completeness["status"] == "not_applicable"
