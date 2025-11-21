################################################################
###########left_arm_gesture###############
"""
Control the AeroPiper left arm in MuJoCo with real-time webcam arm tracking.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))

from arm_to_mujoco import normalized_to_mujoco  # type: ignore[import]
from arm_tracker import ArmGestureController  # type: ignore[import]
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]

ARM_LABELS = [
    "BaseYaw",
    "ShoulderPitch",
    "ShoulderRoll",
    "ElbowFlex",
    "WristPitch",
    "WristRoll",
]

CONTROL_IDXS = [0, 1, 3, 5]  # DOFs mapped to MuJoCo joints

HAND_OPEN_TARGETS = physical_to_mujoco(np.zeros(7))


def _load_mujoco_scene():
    model = mujoco.MjModel.from_xml_path("assets/scene_left.xml")
    data = mujoco.MjData(model)
    return model, data


def set_positions(data_obj, positions):
    """Set target positions for the 13 actuators (6 arm + 7 hand)."""
    assert len(positions) == 13
    data_obj.ctrl[:] = positions


def _apply_deadband(values: np.ndarray, previous: np.ndarray | None, threshold: float):
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


def _write_two_line_block(line1: str, line2: str):
    """Write two persistent CLI lines without scrolling."""
    sys.stdout.write(f"{line1}\033[K\n{line2}\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.flush()


def _print_status(
    normalized: np.ndarray,
    joints: np.ndarray,
    prefix: str = "[Arm|Print]",
):
    norm_stats = " ".join(
        f"{name}:{value:+4.2f}" for name, value in zip(ARM_LABELS, normalized)
    )
    joint_stats = " ".join(
        f"J{idx + 1}:{value:+5.2f}" for idx, value in enumerate(joints)
    )
    line1 = f"\r{prefix} Arm DOFs {norm_stats}"
    line2 = f"[Joints] {joint_stats}"
    _write_two_line_block(line1, line2)


def run_tracking_only(
    controller: ArmGestureController,
    update_threshold: float,
    print_interval: float = 0.1,
):
    """Continuously print the six normalized arm values and joint targets."""
    last_print = 0.0
    last_sent: np.ndarray | None = None
    try:
        while not controller.should_stop():
            normalized = controller.update()
            filtered = _apply_deadband(normalized, last_sent, update_threshold)
            last_sent = filtered
            joint_targets = normalized_to_mujoco(filtered[CONTROL_IDXS])
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, joint_targets)
                last_print = now
    finally:
        print()


def run_simulation(controller, update_threshold: float, print_interval: float):
    model, data = _load_mujoco_scene()
    last_sent: np.ndarray | None = None
    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not controller.should_stop():
            normalized = controller.update()
            filtered = _apply_deadband(normalized, last_sent, update_threshold)
            last_sent = filtered
            arm_targets = normalized_to_mujoco(filtered[CONTROL_IDXS])
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, arm_targets, prefix="[Arm|Sim ]")
                last_print = now

            targets = np.concatenate([arm_targets, HAND_OPEN_TARGETS])
            set_positions(data, targets)
            mujoco.mj_step(model, data)
            viewer.sync()
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the AeroPiper left arm with real-time webcam-based tracking.",
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
        help="Disable the OpenCV preview window if running headless.",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable mirroring in the preview window.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.4,
        help="EMA weight in [0,1] for new pose samples (default: 0.4).",
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
        help="Skip launching MuJoCo; just print the filtered normalized values.",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=0.03,
        help="Minimum change in normalized value required to emit a new value (default: 0.03).",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.35,
        help="Maximum normalized jump allowed per frame; larger spikes are clamped (default: 0.35).",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n-pose.pt",
        help="Ultralytics YOLO pose weights file to load for arm tracking.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.45,
        help="Confidence threshold for YOLO pose detections (default: 0.45).",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.5,
        help="IoU threshold for YOLO pose NMS (default: 0.5).",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="Image size (pixels) used by YOLO during inference (default: 640).",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Torch device string for YOLO (e.g., 'cuda:0' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=1,
        help="Maximum number of pose detections per frame (default: 1).",
    )
    parser.add_argument(
        "--yolo-frame-skip",
        type=int,
        default=1,
        help="Number of frames to skip between YOLO inferences (default: 1).",
    )
    parser.add_argument(
        "--yolo-half",
        action="store_true",
        help="Use half-precision (FP16) inference when running on CUDA devices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = None
    try:
        controller = ArmGestureController(
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            smoothing=args.smoothing,
            max_step=args.max_step,
            yolo_weights=args.yolo_model,
            yolo_confidence=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_max_det=args.yolo_max_det,
            yolo_frame_skip=args.yolo_frame_skip,
            yolo_half_precision=args.yolo_half,
        )
        threshold = max(args.update_threshold, 0.0)
        if args.print_only:
            run_tracking_only(
                controller,
                threshold,
                args.print_interval,
            )
        else:
            run_simulation(controller, threshold, args.print_interval)
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
