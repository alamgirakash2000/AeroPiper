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
from hand_tracker import DEFAULT_YOLO_WEIGHTS  # type: ignore[import]
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


_print_combo_status_initialized = False

def _print_combo_status(
    arm_normalized: np.ndarray,
    hand_physical: np.ndarray,
    arm_joints: np.ndarray,
    hand_joints: np.ndarray,
):
    global _print_combo_status_initialized
    
    # Line 1: Physical pose values with labels (arm + hand)
    arm_vals = " ".join(f"{name}:{v:+.2f}" for name, v in zip(ARM_LABELS, arm_normalized))
    hand_vals = " ".join(f"{name}:{v:.1f}" for name, v in zip(HAND_LABELS, hand_physical))
    # Line 2: Joint values being sent with labels (arm + hand)
    arm_j = " ".join(f"J{i+1}:{v:+.2f}" for i, v in enumerate(arm_joints))
    hand_j = " ".join(f"{name}:{v:.2f}" for name, v in zip(HAND_LABELS, hand_joints))
    
    if not _print_combo_status_initialized:
        # First time: just print the two lines
        print(f"Pose:  Arm[{arm_vals}] Hand[{hand_vals}]")
        print(f"Joint: Arm[{arm_j}] Hand[{hand_j}]", end="", flush=True)
        _print_combo_status_initialized = True
    else:
        # Subsequent times: move up, clear, and rewrite both lines
        print(f"\033[A\r\033[KPose:  Arm[{arm_vals}] Hand[{hand_vals}]")
        print(f"\r\033[KJoint: Arm[{arm_j}] Hand[{hand_j}]", end="", flush=True)


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
                hand_joints = physical_to_mujoco(hand_filtered)
                _print_combo_status(
                    arm_filtered,
                    hand_filtered,
                    arm_joints,
                    hand_joints,
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
                    hand_filtered,
                    arm_targets,
                    hand_targets,
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
    parser.add_argument(
        "--yolo-weights",
        default=DEFAULT_YOLO_WEIGHTS,
        help=(
            "Path or Ultralytics alias for the YOLO pose checkpoint "
            f"(default: {DEFAULT_YOLO_WEIGHTS})."
        ),
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO inference resolution (square) in pixels.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.45,
        help="YOLO detection confidence threshold (0-1).",
    )
    parser.add_argument(
        "--handedness",
        choices=("left", "right"),
        default="left",
        help="Hand to track in the frame.",
    )
    parser.add_argument(
        "--pseudo-depth-scale",
        type=float,
        default=0.08,
        help="Scale factor for pseudo-depth estimation (tune per camera).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of YOLO detections to keep per frame.",
    )
    parser.add_argument(
        "--hand-backend",
        choices=("yolo", "mediapipe"),
        default="yolo",
        help="Primary hand landmark backend (default: yolo).",
    )
    parser.add_argument(
        "--no-mediapipe-fallback",
        action="store_true",
        help="Disable automatic fallback to MediaPipe when YOLO lacks 21 keypoints.",
    )
    parser.add_argument(
        "--thumb-flexion-gain",
        type=float,
        default=1.8,
        help="Multiplier applied to both thumb flexion DOFs (default: 1.8).",
    )
    parser.add_argument(
        "--thumb-flexion-bias",
        type=float,
        default=0.0,
        help="Additive bias applied after thumb flexion gain is applied.",
    )
    parser.add_argument(
        "--thumb-abd-gain",
        type=float,
        default=1.5,
        help="Multiplier applied to thumb abduction values (default: 1.5).",
    )
    parser.add_argument(
        "--thumb-abd-bias",
        type=float,
        default=0.0,
        help="Additive bias applied to thumb abduction after gain.",
    )
    parser.add_argument(
        "--thumb-abd-invert",
        action="store_true",
        help="Invert thumb abduction output (useful if your camera flips the axes).",
    )
    parser.add_argument(
        "--thumb-abd-smoothing",
        type=float,
        default=0.35,
        help="EMA weight for thumb abduction stabilizer (0 disables smoothing).",
    )
    parser.add_argument(
        "--thumb-abd-deadband",
        type=float,
        default=0.08,
        help="Value below which thumb abduction output is forced to zero.",
    )
    parser.add_argument(
        "--thumb1-curve",
        type=float,
        default=0.75,
        help="Exponent applied to the Thumb1 channel (default: 0.75 to boost mid-range).",
    )
    parser.add_argument(
        "--thumb2-curve",
        type=float,
        default=1.05,
        help="Exponent applied to the Thumb2 channel (default: 1.05 for subtle curve).",
    )
    parser.add_argument(
        "--thumb1-freeze-seconds",
        type=float,
        default=0.4,
        help="Hold Thumb1 steady for this many seconds after start/reset to isolate abduction.",
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
            hand_backend=args.hand_backend,
            hand_yolo_weights=args.yolo_weights,
            hand_yolo_imgsz=args.yolo_imgsz,
            hand_yolo_confidence=args.yolo_confidence,
            hand_yolo_device=args.yolo_device,
            hand_handedness=args.handedness,
            hand_pseudo_depth_scale=args.pseudo_depth_scale,
            hand_max_hands=args.max_hands,
            hand_allow_mediapipe_fallback=not args.no_mediapipe_fallback,
            hand_thumb_flexion_gain=args.thumb_flexion_gain,
            hand_thumb_flexion_bias=args.thumb_flexion_bias,
            hand_thumb_abd_gain=args.thumb_abd_gain,
            hand_thumb_abd_bias=args.thumb_abd_bias,
            hand_thumb_abd_invert=args.thumb_abd_invert,
            hand_thumb_abd_smoothing=args.thumb_abd_smoothing,
            hand_thumb_abd_deadband=args.thumb_abd_deadband,
            hand_thumb1_curve=args.thumb1_curve,
            hand_thumb2_curve=args.thumb2_curve,
            hand_thumb1_freeze_seconds=args.thumb1_freeze_seconds,
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

