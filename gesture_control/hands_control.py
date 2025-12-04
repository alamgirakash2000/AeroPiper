#!/usr/bin/env python3
"""
Track both hands with MediaPipe and drive the dual scene (frame version).
Arms stay at midpose; both hands follow real-time finger gestures.
"""
import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2  # type: ignore[import]
import mediapipe as mp  # type: ignore[import]
import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]


MODEL_PATH = "assets/scene.xml"

# Actuator layout (right first, then left) in the dual-with-frame MJCF
RIGHT_ARM_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 13)
LEFT_ARM_SLICE = slice(13, 19)
LEFT_HAND_SLICE = slice(19, 26)

# Mid pose for arms (matches MJCF home)
ARM_MID = np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0], dtype=float)

# Helpers for landmark math (adapted from hand_tracker.py)
def _vector_angle(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-5 or v2_norm < 1e-5:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _angle_to_ratio(v1, v2, open_deg: float, closed_deg: float):
    angle = _vector_angle(v1, v2)
    open_rad = np.deg2rad(open_deg)
    closed_rad = np.deg2rad(closed_deg)
    if not np.isfinite(open_rad):
        open_rad = 0.0
    if not np.isfinite(closed_rad) or closed_rad <= open_rad + 1e-4:
        closed_rad = open_rad + 1.0
    span = closed_rad - open_rad
    normalized = (angle - open_rad) / span
    return float(np.clip(normalized, 0.0, 1.0))


def _finger_curl(points, mcp_idx, pip_idx, tip_idx, open_deg: float, closed_deg: float):
    base = points[mcp_idx]
    mid = points[pip_idx]
    fingertip = points[tip_idx]
    v1 = mid - base
    v2 = fingertip - mid
    return _angle_to_ratio(v1, v2, open_deg, closed_deg)


def _joint_flexion(points, root_idx, mid_idx, tip_idx, open_deg: float, closed_deg: float):
    root = points[root_idx]
    mid = points[mid_idx]
    tip = points[tip_idx]
    v1 = mid - root
    v2 = tip - mid
    return _angle_to_ratio(v1, v2, open_deg, closed_deg)


def _thumb_abduction(points, hl, lm):
    wrist = points[lm(hl, "WRIST")]
    thumb_mcp = points[lm(hl, "THUMB_MCP")]
    index_mcp = points[lm(hl, "INDEX_MCP", "INDEX_FINGER_MCP")]

    thumb_vec = thumb_mcp - wrist
    index_vec = index_mcp - wrist
    angle_ratio = _angle_to_ratio(thumb_vec, index_vec, 0.0, 90.0)

    thumb_tip = points[lm(hl, "THUMB_TIP")]
    pinky_mcp = points[lm(hl, "PINKY_MCP", "PINKY_FINGER_MCP")]
    gap = np.linalg.norm(thumb_tip - index_mcp)
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    palm_width = max(palm_width, 1e-4)
    gap_ratio = np.clip((gap / palm_width - 0.0) / 1.5, 0.0, 1.0)

    combined = 0.8 * angle_ratio + 0.2 * gap_ratio
    raw_val = 1.0 - combined  # invert so open -> 0, closed -> 1
    return float(np.clip(raw_val, 0.0, 1.0))


def _lm(hl, *names):
    for name in names:
        if hasattr(hl, name):
            return getattr(hl, name).value
    raise AttributeError(f"HandLandmark missing attrs: {names}")


def landmarks_to_physical(hand_landmarks) -> np.ndarray:
    """Return 7 physical DOFs (0..100) for one hand."""
    hl = mp.solutions.hands.HandLandmark
    points = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)
    values = np.zeros(7, dtype=np.float32)

    values[0] = _thumb_abduction(points, hl, _lm)
    values[1] = _joint_flexion(
        points,
        _lm(hl, "THUMB_CMC"),
        _lm(hl, "THUMB_MCP"),
        _lm(hl, "THUMB_IP"),
        5.0,
        65.0,
    )
    values[2] = _joint_flexion(
        points,
        _lm(hl, "THUMB_MCP"),
        _lm(hl, "THUMB_IP"),
        _lm(hl, "THUMB_TIP"),
        5.0,
        115.0,
    )
    values[3] = _finger_curl(
        points,
        _lm(hl, "INDEX_MCP", "INDEX_FINGER_MCP"),
        _lm(hl, "INDEX_PIP", "INDEX_FINGER_PIP"),
        _lm(hl, "INDEX_TIP", "INDEX_FINGER_TIP"),
        5.0,
        110.0,
    )
    values[4] = _finger_curl(
        points,
        _lm(hl, "MIDDLE_MCP", "MIDDLE_FINGER_MCP"),
        _lm(hl, "MIDDLE_PIP", "MIDDLE_FINGER_PIP"),
        _lm(hl, "MIDDLE_TIP", "MIDDLE_FINGER_TIP"),
        5.0,
        110.0,
    )
    values[5] = _finger_curl(
        points,
        _lm(hl, "RING_MCP", "RING_FINGER_MCP"),
        _lm(hl, "RING_PIP", "RING_FINGER_PIP"),
        _lm(hl, "RING_TIP", "RING_FINGER_TIP"),
        5.0,
        110.0,
    )
    values[6] = _finger_curl(
        points,
        _lm(hl, "PINKY_MCP", "PINKY_FINGER_MCP"),
        _lm(hl, "PINKY_PIP", "PINKY_FINGER_PIP"),
        _lm(hl, "PINKY_TIP", "PINKY_FINGER_TIP"),
        5.0,
        110.0,
    )

    return np.clip(values * 100.0, 0.0, 100.0)


def process_hands(results) -> Dict[str, np.ndarray]:
    """
    Return dict with 'left'/'right' -> 7 physical DOFs (0..100).
    Handles cases where MediaPipe mislabels both as the same hand by assigning
    the first to right and the second to left.
    """
    if not results.multi_hand_landmarks:
        return {}
    handedness_list = getattr(results, "multi_handedness", None)
    raw: list[Tuple[str, np.ndarray]] = []
    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        label = None
        if handedness_list and idx < len(handedness_list):
            label = handedness_list[idx].classification[0].label.lower()
        if label not in ("left", "right"):
            label = f"hand{idx}"
        raw.append((label, landmarks_to_physical(hand_lms)))

    out: Dict[str, np.ndarray] = {}
    # Prefer explicit labels if distinct
    for label, vals in raw:
        if label in ("left", "right") and label not in out:
            out[label] = vals
    # If we have two hands but labels collided, assign by order
    if len(raw) >= 2 and ("left" not in out or "right" not in out):
        first, second = raw[0][1], raw[1][1]
        if "right" not in out:
            out["right"] = first
        if "left" not in out:
            out["left"] = second
    return out


def hard_clamp_joint_actuators(model: mujoco.MjModel, data: mujoco.MjData):
    """Force joint-actuated DoFs to match their ctrl values exactly."""
    for act_id in range(model.nu):
        if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
            j_id = model.actuator_trnid[act_id][0]
            qadr = model.jnt_qposadr[j_id]
            data.qpos[qadr] = data.ctrl[act_id]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control both AeroPiper hands via webcam gestures (MediaPipe)."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window.")
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=2.0,
        help="Minimum change (0-100 scale) to emit a new value (default: 2).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.2,
        help="EMA weight in [0,1] for new gesture samples (default: 0.2 for lower latency).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Camera setup
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,  # faster
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Initialize arms at mid pose
    data.ctrl[:] = 0
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.ctrl[RIGHT_ARM_SLICE] = ARM_MID
    data.ctrl[LEFT_ARM_SLICE] = ARM_MID
    hard_clamp_joint_actuators(model, data)

    # State for smoothing and deadband
    last_left: Optional[np.ndarray] = None
    last_right: Optional[np.ndarray] = None
    alpha = float(np.clip(args.smoothing, 0.0, 1.0))
    threshold = max(args.update_threshold, 0.0)

    def _apply_deadband(current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        if previous is None:
            return current
        out = previous.copy()
        mask = np.abs(current - previous) >= threshold
        out[mask] = current[mask]
        return out

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            last_print = 0.0
            while viewer.is_running():
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                detected = process_hands(results)

                # Defaults (open hand)
                left_phys = np.zeros(7, dtype=float)
                right_phys = np.zeros(7, dtype=float)

                if "left" in detected:
                    left_phys = detected["left"]
                if "right" in detected:
                    right_phys = detected["right"]

                # Smooth + deadband
                if last_left is None:
                    left_sm = left_phys
                else:
                    left_sm = alpha * left_phys + (1.0 - alpha) * last_left
                if last_right is None:
                    right_sm = right_phys
                else:
                    right_sm = alpha * right_phys + (1.0 - alpha) * last_right

                left_sm = _apply_deadband(left_sm, last_left)
                right_sm = _apply_deadband(right_sm, last_right)
                last_left, last_right = left_sm, right_sm

                # Map to MuJoCo ctrl
                left_ctrl = physical_to_mujoco(left_sm)
                right_ctrl = physical_to_mujoco(right_sm)

                data.ctrl[LEFT_HAND_SLICE] = left_ctrl
                data.ctrl[RIGHT_HAND_SLICE] = right_ctrl

                # Physics step then clamp arms
                mujoco.mj_step(model, data)
                hard_clamp_joint_actuators(model, data)

                # Preview
                if not args.no_preview:
                    overlay = frame.copy()
                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                overlay, hand_lms, mp.solutions.hands.HAND_CONNECTIONS
                            )
                    cv2.imshow("Both Hands (Q/ESC to quit)", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                # Light periodic print for debugging
                now = time.time()
                if now - last_print > 0.5:
                    print(
                        f"\rLeft: {left_sm.round(1)} | Right: {right_sm.round(1)}",
                        end="",
                        flush=True,
                    )
                    last_print = now

                viewer.sync()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(hands, "close"):
            hands.close()
        print()


if __name__ == "__main__":
    main()


