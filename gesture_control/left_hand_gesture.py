import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]
from hand_tracker import DEFAULT_YOLO_WEIGHTS, HandGestureController  # type: ignore[import]

HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]


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


_print_status_initialized = False

def _print_status(
    physical: np.ndarray,
    mujoco_vals: np.ndarray,
    prefix: str = "",
):
    global _print_status_initialized
    
    # Line 1: Physical pose values with labels
    pose_vals = " ".join(f"{name}:{v:.1f}" for name, v in zip(HAND_LABELS, physical))
    # Line 2: Joint values being sent with labels
    joint_vals = " ".join(f"{name}:{v:.2f}" for name, v in zip(HAND_LABELS, mujoco_vals))
    
    if not _print_status_initialized:
        # First time: just print the two lines
        print(f"Pose:  {pose_vals}")
        print(f"Joint: {joint_vals}", end="", flush=True)
        _print_status_initialized = True
    else:
        # Subsequent times: move up, clear, and rewrite both lines
        print(f"\033[A\r\033[KPose:  {pose_vals}")
        print(f"\r\033[KJoint: {joint_vals}", end="", flush=True)


def run_tracking_only(
    controller,
    update_threshold: float,
    print_interval: float = 0.1,
):
    """Continuously print the 7 physical control values without launching MuJoCo."""
    last_print = 0.0
    last_sent: np.ndarray | None = None
    try:
        while not controller.should_stop():
            physical_hand = controller.update()
            filtered = _apply_deadband(physical_hand, last_sent, update_threshold)
            last_sent = filtered
            hand_targets = physical_to_mujoco(filtered)
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, hand_targets)
                last_print = now
    finally:
        print()


def run_simulation(controller, update_threshold: float, print_interval: float):
    model, data = _load_mujoco_scene()
    arm_targets = np.zeros(6)
    last_sent: np.ndarray | None = None
    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not controller.should_stop():
            physical_hand = controller.update()
            filtered = _apply_deadband(physical_hand, last_sent, update_threshold)
            last_sent = filtered
            hand_targets = physical_to_mujoco(filtered)
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, hand_targets)
                last_print = now
            targets = np.concatenate([arm_targets, hand_targets])
            set_positions(data, targets)
            mujoco.mj_step(model, data)
            viewer.sync()
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Control the AeroPiper left hand with real-time webcam-based gestures "
            "while keeping the arm fixed."
        )
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
        default=0.35,
        help="EMA weight in [0,1] for new gesture samples (default: 0.35).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="Seconds between CLI print updates.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Skip launching MuJoCo; just print the filtered physical values.",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=5.0,
        help="Minimum change (0-100 scale) required to emit a new value (default: 5).",
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
        controller = HandGestureController(
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            smoothing=args.smoothing,
            yolo_weights=args.yolo_weights,
            yolo_device=args.yolo_device,
            yolo_imgsz=args.yolo_imgsz,
            yolo_confidence=args.yolo_confidence,
            handedness=args.handedness,
            pseudo_depth_scale=args.pseudo_depth_scale,
            max_hands=args.max_hands,
            backend=args.hand_backend,
            mediapipe_fallback=not args.no_mediapipe_fallback,
            thumb_flexion_gain=args.thumb_flexion_gain,
            thumb_flexion_bias=args.thumb_flexion_bias,
            thumb_abd_gain=args.thumb_abd_gain,
            thumb_abd_bias=args.thumb_abd_bias,
            thumb_abd_invert=args.thumb_abd_invert,
            thumb_abd_smoothing=args.thumb_abd_smoothing,
            thumb_abd_deadband=args.thumb_abd_deadband,
            thumb1_curve=args.thumb1_curve,
            thumb2_curve=args.thumb2_curve,
            thumb1_freeze_seconds=args.thumb1_freeze_seconds,
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
