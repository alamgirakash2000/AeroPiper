"""
Control the AeroPiper left arm and hand simultaneously using one webcam feed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))

from arm_to_mujoco import normalized_to_mujoco  # type: ignore[import]
from combo_tracker import ComboGestureController  # type: ignore[import]
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]

ARM_LABELS = [
    "BaseYaw",
    "ShoulderPitch",
    "ShoulderRoll",
    "ElbowFlex",
    "WristPitch",
    "WristRoll",
]
HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]
CONTROL_IDXS = [0, 1, 3, 5]


def _load_mujoco_scene():
    model = mujoco.MjModel.from_xml_path("assets/scene_left.xml")
    data = mujoco.MjData(model)
    return model, data


def set_positions(data_obj, positions):
    """Set target positions for the 13 actuators (6 arm + 7 hand)."""
    assert len(positions) == 13
    data_obj.ctrl[:] = positions


def _apply_deadband(values: np.ndarray, previous: Optional[np.ndarray], threshold: float):
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


def _write_two_line_block(line1: str, line2: str):
    sys.stdout.write(f"{line1}\033[K\n{line2}\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.flush()


def _print_combo_status(
    arm_normalized: np.ndarray,
    arm_joints: np.ndarray,
    prefix: str,
):
    arm_norm_stats = " ".join(
        f"{name}:{value:+4.2f}" for name, value in zip(ARM_LABELS, arm_normalized)
    )
    joint_stats = " ".join(
        f"J{idx + 1}:{value:+5.2f}" for idx, value in enumerate(arm_joints)
    )
    line1 = f"\r{prefix} Arm DOFs {arm_norm_stats}"
    line2 = f"[Joints] {joint_stats}"
    _write_two_line_block(line1, line2)


def run_tracking_only(controller: ComboGestureController, args):
    last_print = 0.0
    arm_last: Optional[np.ndarray] = None
    hand_last: Optional[np.ndarray] = None
    try:
        while not controller.should_stop():
            arm_values, hand_values = controller.update()
            arm_filtered = _apply_deadband(arm_values, arm_last, args.arm_update_threshold)
            hand_filtered = _apply_deadband(hand_values, hand_last, args.hand_update_threshold)
            arm_last = arm_filtered
            hand_last = hand_filtered

            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                arm_joints = normalized_to_mujoco(arm_filtered[CONTROL_IDXS])
                _print_combo_status(
                    arm_filtered,
                    arm_joints,
                    prefix="[Combo|Print]",
                )
                last_print = now
    finally:
        print()


def run_simulation(controller: ComboGestureController, args):
    model, data = _load_mujoco_scene()
    last_print = 0.0
    arm_last: Optional[np.ndarray] = None
    hand_last: Optional[np.ndarray] = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not controller.should_stop():
            arm_values, hand_values = controller.update()

            arm_filtered = _apply_deadband(arm_values, arm_last, args.arm_update_threshold)
            hand_filtered = _apply_deadband(hand_values, hand_last, args.hand_update_threshold)
            arm_last = arm_filtered
            hand_last = hand_filtered

            arm_targets = normalized_to_mujoco(arm_filtered[CONTROL_IDXS])
            hand_targets = physical_to_mujoco(hand_filtered)

            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                _print_combo_status(
                    arm_filtered,
                    arm_targets,
                    prefix="[Combo|Sim ]",
                )
                last_print = now

            targets = np.concatenate([arm_targets, hand_targets])
            set_positions(data, targets)
            mujoco.mj_step(model, data)
            viewer.sync()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the AeroPiper left arm + hand with one webcam feed.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the shared preview window.",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable the mirror effect in the preview window.",
    )
    parser.add_argument(
        "--arm-smoothing",
        type=float,
        default=0.4,
        help="EMA weight for arm pose samples (default: 0.4).",
    )
    parser.add_argument(
        "--hand-smoothing",
        type=float,
        default=0.25,
        help="EMA weight for hand gesture samples (default: 0.25).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.05,
        help="Seconds between CLI print updates.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Skip launching MuJoCo; just print filtered values.",
    )
    parser.add_argument(
        "--arm-update-threshold",
        type=float,
        default=0.03,
        help="Minimum normalized change required to emit new arm values (default: 0.03).",
    )
    parser.add_argument(
        "--hand-update-threshold",
        type=float,
        default=3.0,
        help="Minimum change (0-100 scale) required to emit new hand values (default: 3).",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.35,
        help="Maximum normalized arm jump allowed per frame; spikes are clamped (default: 0.35).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = None

    try:
        controller = ComboGestureController(
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            arm_smoothing=args.arm_smoothing,
            hand_smoothing=args.hand_smoothing,
            max_step=args.max_step,
        )

        if args.print_only:
            run_tracking_only(controller, args)
        else:
            run_simulation(controller, args)

    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()

