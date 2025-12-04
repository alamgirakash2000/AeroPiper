"""
Control both AeroPiper arms + hands with one webcam feed (combo tracker), using
the dual-with-frame scene. Same gesture drives both sides for responsiveness.
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
import cv2  # type: ignore[import]
import mediapipe as mp  # type: ignore[import]
mp_drawing = mp.solutions.drawing_utils

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))

from arm_to_mujoco import normalized_to_mujoco  # type: ignore[import]
from combo_tracker import ComboGestureController  # type: ignore[import]
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]
from hands_control import process_hands  # type: ignore[import]


ARM_LABELS = [
    "BaseYaw",
    "ShoulderPitch",
    "ShoulderRoll",
    "ElbowFlex",
    "WristPitch",
    "WristRoll",
]
HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]
CONTROL_IDXS = [0, 1, 3, 5]  # map normalized arm DOFs -> joints (J1, J2, J3, J6)

MODEL_PATH = "assets/scene.xml"

# Actuator layout in dual_with_frame: right arm 0-5, right hand 6-12, left arm 13-18, left hand 19-25
RIGHT_ARM_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 13)
LEFT_ARM_SLICE = slice(13, 19)
LEFT_HAND_SLICE = slice(19, 26)

# Mid pose for arms (matches MJCF home)
ARM_MID = np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0], dtype=float)


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


def _norm_to_range(model: mujoco.MjModel, jname: str, norm_val: float) -> float:
    """Map normalized [-1, 1] to the joint's range midpoint-based."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        return 0.0
    lo, hi = model.jnt_range[jid]
    return lo + (norm_val + 1.0) * 0.5 * (hi - lo)


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
                    prefix="[Combo2|Print]",
                )
                last_print = now
    finally:
        print()


def run_simulation(controller: ComboGestureController, args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    last_print = 0.0
    arm_last: Optional[np.ndarray] = None
    hand_last: Optional[np.ndarray] = None

    # Start at midpose for both arms, fingers open
    data.ctrl[:] = 0
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.ctrl[RIGHT_ARM_SLICE] = ARM_MID
    data.ctrl[LEFT_ARM_SLICE] = ARM_MID
    mujoco.mj_forward(model, data)

    # Fast dual-hand tracker (independent left/right) using MediaPipe
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    show_overlay = not args.no_preview

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not controller.should_stop():
            arm_values, hand_values = controller.update()

            arm_filtered = _apply_deadband(arm_values, arm_last, args.arm_update_threshold)
            hand_filtered = _apply_deadband(hand_values, hand_last, args.hand_update_threshold)
            arm_last = arm_filtered
            hand_last = hand_filtered

            # Try to get independent left/right hand measurements
            left_hand_phys = hand_filtered
            right_hand_phys = hand_filtered

            frame = getattr(controller, "_frame", None)
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb)
                detected = process_hands(results)
                if "left" in detected:
                    left_hand_phys = detected["left"]
                if "right" in detected:
                    right_hand_phys = detected["right"]

                if show_overlay:
                    overlay = frame.copy()
                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                overlay,
                                hand_lms,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                            )
                    cv2.putText(
                        overlay,
                        "Both hands tracking (Q/ESC to quit)",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Both Hands Preview", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

            # Custom mapping:
            # shoulder (yaw/pitch) -> joints 1,2 ; elbow -> joint3 ; wrist pitch -> joint5 ; wrist roll -> joint6
            j1 = _norm_to_range(model, "joint1", arm_filtered[0])
            j2 = _norm_to_range(model, "joint2", arm_filtered[1])
            j3 = _norm_to_range(model, "joint3", arm_filtered[3])
            j5 = _norm_to_range(model, "joint5", arm_filtered[4])
            j6 = _norm_to_range(model, "joint6", arm_filtered[5])
            arm_targets = np.array([j1, j2, j3, 0.0, j5, j6], dtype=float)
            right_hand_targets = physical_to_mujoco(right_hand_phys)
            left_hand_targets = physical_to_mujoco(left_hand_phys)

            # Apply same commands to both sides (mirrored control)
            data.ctrl[RIGHT_ARM_SLICE] = arm_targets
            data.ctrl[LEFT_ARM_SLICE] = arm_targets
            data.ctrl[RIGHT_HAND_SLICE] = right_hand_targets
            data.ctrl[LEFT_HAND_SLICE] = left_hand_targets

            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                _print_combo_status(
                    arm_filtered,
                    arm_targets,
                    prefix="[Combo2|Sim ]",
                )
                last_print = now

            mujoco.mj_step(model, data)
            viewer.sync()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control both AeroPiper arms + hands with one webcam feed.",
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
        default=0.35,
        help="EMA weight for arm pose samples (default: 0.35 for responsiveness).",
    )
    parser.add_argument(
        "--hand-smoothing",
        type=float,
        default=0.2,
        help="EMA weight for hand gesture samples (default: 0.2 for responsiveness).",
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
        default=0.02,
        help="Minimum normalized change required to emit new arm values (default: 0.02).",
    )
    parser.add_argument(
        "--hand-update-threshold",
        type=float,
        default=2.0,
        help="Minimum change (0-100 scale) required to emit new hand values (default: 2).",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.3,
        help="Maximum normalized arm jump allowed per frame; spikes are clamped (default: 0.3).",
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


