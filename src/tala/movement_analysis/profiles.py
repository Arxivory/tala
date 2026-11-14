
"""Exercise metadata and movement-pattern definitions.

Profiles describe the expected interpretation of a movement signal.
They do not, by themselves, validate pose tracking or exercise recognition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MovementProfile:
    """Configuration describing an exercise's movement pattern."""

    name: str
    display_name: str

    # Name of the signal expected by default.
    # The caller may supply another signal when appropriate.
    primary_signal: str

    # Human-readable biomechanical interpretation of the signal.
    signal_description: str

    # Labels assigned to intervals where the signal increases/decreases.
    increasing_phase: str
    decreasing_phase: str

    # Expected extreme at the beginning/end of a complete repetition.
    # "high" means near the maximum signal value.
    # "low" means near the minimum signal value.
    cycle_start_extreme: str

    # Joints or body regions relevant to this exercise.
    primary_joints: tuple[str, ...]

    # Whether this profile represents a cyclic movement.
    is_cyclic: bool = True

    # Relative tolerance used when evaluating whether an endpoint
    # has returned sufficiently close to the expected extreme.
    completion_tolerance: float = 0.15

    # Additional notes about interpretation and validation.
    notes: str = ""


MOVEMENT_PROFILES: dict[str, MovementProfile] = {
    "squat": MovementProfile(
        name="squat",
        display_name="Squat",
        primary_signal="com_y_m",
        signal_description=(
            "Whole-body center-of-mass vertical position. "
            "A higher value generally represents a more extended posture."
        ),
        increasing_phase="ascent",
        decreasing_phase="descent",
        cycle_start_extreme="high",
        primary_joints=("hip", "knee", "ankle"),
        notes=(
            "Current reference exercise. COM-based phase interpretation "
            "is supported by the existing squat reconstruction workflow."
        ),
    ),

    "lunge": MovementProfile(
        name="lunge",
        display_name="Lunge",
        primary_signal="pelvis_ty",
        signal_description=(
            "Pelvis vertical position. A decrease commonly accompanies "
            "the lowering portion of a lunge."
        ),
        increasing_phase="return",
        decreasing_phase="lowering",
        cycle_start_extreme="high",
        primary_joints=("hip", "knee", "ankle"),
        notes=(
            "Profile definition only. Lunge-specific pose mapping, "
            "left/right leg identification, and validation are still required."
        ),
    ),

    "push_up": MovementProfile(
        name="push_up",
        display_name="Push-up",
        primary_signal="elbow_angle_mean",
        signal_description=(
            "Mean elbow flexion/extension signal. The precise sign and "
            "angle convention must be verified for the selected pose pipeline."
        ),
        increasing_phase="pushing",
        decreasing_phase="lowering",
        cycle_start_extreme="high",
        primary_joints=("shoulder", "elbow", "wrist"),
        notes=(
            "The supplied signal must be defined consistently. "
            "A validated upper-body pose and joint-angle pipeline is required."
        ),
    ),

    "jumping_jack": MovementProfile(
        name="jumping_jack",
        display_name="Jumping Jack",
        primary_signal="arm_separation",
        signal_description=(
            "A measure of arm separation or elevation. Higher values "
            "should represent the open-arm position."
        ),
        increasing_phase="opening",
        decreasing_phase="closing",
        cycle_start_extreme="low",
        primary_joints=("shoulder", "hip", "ankle"),
        notes=(
            "Profile definition only. A suitable arm-position signal and "
            "reliable full-body tracking are required."
        ),
    ),

    "plank": MovementProfile(
        name="plank",
        display_name="Plank",
        primary_signal="body_alignment",
        signal_description=(
            "A measure of body alignment or posture stability."
        ),
        increasing_phase="entering_hold",
        decreasing_phase="leaving_hold",
        cycle_start_extreme="high",
        primary_joints=("shoulder", "hip", "ankle"),
        is_cyclic=False,
        notes=(
            "A static hold is not naturally represented by repetition "
            "counting. A dedicated hold-duration and posture-stability "
            "analyzer is needed."
        ),
    ),
}


def get_profile(exercise: str) -> MovementProfile:
    """Return a registered exercise profile."""

    key = exercise.strip().lower()

    if key not in MOVEMENT_PROFILES:
        supported = ", ".join(sorted(MOVEMENT_PROFILES))
        raise ValueError(
            f"Unsupported exercise '{exercise}'. "
            f"Supported profiles: {supported}"
        )

    return MOVEMENT_PROFILES[key]


def list_profiles() -> list[str]:
    """Return the names of registered exercise profiles."""

    return sorted(MOVEMENT_PROFILES)
