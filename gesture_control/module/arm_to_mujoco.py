"""
Mapping utilities that convert normalized left-arm joint commands into MuJoCo
control targets for the AeroPiper left arm.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Joint limits read from `assets/aero_piper_left.xml`
# Order: [left_joint1..left_joint6]
ARM_MIN = np.array(
    [-2.618, 0.0, -2.697, -1.832, -1.22, -3.14],
    dtype=float,
)
ARM_MAX = np.array(
    [2.618, 3.14, 0.0, 1.832, 1.22, 3.14],
    dtype=float,
)


def normalized_to_mujoco(values: Iterable[float]) -> np.ndarray:
    """
    Map normalized joint commands in [-1, 1] to MuJoCo control targets.

    Parameters
    ----------
    values:
        Iterable of four normalized DOFs in [-1, 1] representing:
        [base_yaw, shoulder_pitch, elbow_flex, wrist_roll]

    Returns
    -------
    np.ndarray
        Array of six floats clipped to the actual joint limits.
    """

    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (4,):
        raise ValueError(
            f"Expected 4 normalized arm DOFs, received shape {arr.shape!r}"
        )
    arr = np.clip(arr, -1.0, 1.0)
    joint_targets = np.zeros(6, dtype=float)
    # Neutral posture for uncontrollable joints (J3 shoulder roll, J5 wrist pitch)
    joint_targets[:] = (ARM_MIN + ARM_MAX) * 0.5

    # Drive the DOFs we currently trust from vision: base yaw (J1), shoulder
    # pitch (J2), elbow flex (J4), and wrist roll (J6).
    mapping = [0, 1, 3, 5]
    for value, joint_idx in zip(arr, mapping):
        ratio = (value + 1.0) * 0.5  # [-1,1] -> [0,1]
        joint_targets[joint_idx] = ARM_MIN[joint_idx] + (ARM_MAX[joint_idx] - ARM_MIN[joint_idx]) * ratio

    return joint_targets


def clamp_joint_targets(targets: Iterable[float]) -> np.ndarray:
    """
    Clamp arbitrary joint targets to the valid MuJoCo ctrl range for safety.
    """

    arr = np.asarray(list(targets), dtype=float)
    if arr.shape != (6,):
        raise ValueError(
            f"Expected 6 arm joint targets, received shape {arr.shape!r}"
        )
    return np.clip(arr, ARM_MIN, ARM_MAX)

