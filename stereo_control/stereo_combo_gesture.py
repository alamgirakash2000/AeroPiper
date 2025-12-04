"""
Stereo 3D Arm + Hand Gesture Control for AeroPiper

Uses two calibrated webcams for true 3D tracking of BOTH arm and hand.
Controls all 13 DOFs (6 arm + 7 hand) with view-angle invariant tracking.

Prerequisites:
    1. Two webcams
    2. Stereo calibration file
    3. YOLO pose model (yolov8n-pose.pt in project root)

Usage:
    python stereo_control/stereo_combo_gesture.py --calibration stereo_calibration_default.pkl --left-camera 1 --right-camera 2
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

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gesture_control", "module"))

from stereo_calibration import StereoCalibrator  # type: ignore[import]
from stereo_combo_tracker import StereoComboTracker  # type: ignore[import]
from arm_to_mujoco import normalized_to_mujoco  # type: ignore[import]
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]

ARM_LABELS = ["BaseYaw", "ShoulderPitch", "ShoulderRoll", "ElbowFlex", "WristPitch", "WristRoll"]
HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]
CONTROL_IDXS = [0, 1, 3, 5]  # Controlled arm joints


def _load_mujoco_scene():
    """Load MuJoCo scene."""
    model = mujoco.MjModel.from_xml_path("assets/scene_left.xml")
    data = mujoco.MjData(model)
    return model, data


def set_positions(data_obj, positions):
    """Set positions for 13 actuators (6 arm + 7 hand)."""
    assert len(positions) == 13
    data_obj.ctrl[:] = positions


def _apply_deadband(values: np.ndarray, previous: Optional[np.ndarray], threshold: float):
    """Apply deadband filter."""
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


def _write_two_line_block(line1: str, line2: str):
    """Write two lines and move cursor up."""
    sys.stdout.write(f"{line1}\033[K\n{line2}\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.flush()


def _print_combo_status(
    arm_normalized: np.ndarray,
    hand_physical: np.ndarray,
    arm_joints: np.ndarray,
    prefix: str,
):
    """Print arm and hand status."""
    arm_stats = " ".join(f"{name}:{val:+4.2f}" for name, val in zip(ARM_LABELS, arm_normalized))
    hand_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(HAND_LABELS, hand_physical))
    joint_stats = " ".join(f"J{idx+1}:{val:+5.2f}" for idx, val in enumerate(arm_joints))
    
    line1 = f"\r{prefix} Arm: {arm_stats} | Hand: {hand_stats}"
    line2 = f"[Joints] {joint_stats}"
    _write_two_line_block(line1, line2)


def run_tracking_only(tracker: StereoComboTracker, args):
    """Run tracking without MuJoCo (print values only)."""
    last_print = 0.0
    arm_last: Optional[np.ndarray] = None
    hand_last: Optional[np.ndarray] = None
    
    try:
        while not tracker.should_stop():
            arm_values, hand_values = tracker.update()
            
            arm_filtered = _apply_deadband(arm_values, arm_last, args.arm_update_threshold)
            hand_filtered = _apply_deadband(hand_values, hand_last, args.hand_update_threshold)
            arm_last = arm_filtered
            hand_last = hand_filtered
            
            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                arm_joints = normalized_to_mujoco(arm_filtered[CONTROL_IDXS])
                _print_combo_status(
                    arm_filtered,
                    hand_filtered,
                    arm_joints,
                    prefix="[3D Combo|Print]",
                )
                last_print = now
    finally:
        print()


def run_simulation(tracker: StereoComboTracker, args):
    """Run MuJoCo simulation with stereo arm+hand tracking."""
    model, data = _load_mujoco_scene()
    last_print = 0.0
    arm_last: Optional[np.ndarray] = None
    hand_last: Optional[np.ndarray] = None
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not tracker.should_stop():
            arm_values, hand_values = tracker.update()
            
            arm_filtered = _apply_deadband(arm_values, arm_last, args.arm_update_threshold)
            hand_filtered = _apply_deadband(hand_values, hand_last, args.hand_update_threshold)
            arm_last = arm_filtered
            hand_last = hand_filtered
            
            # Convert to MuJoCo control
            arm_targets = normalized_to_mujoco(arm_filtered[CONTROL_IDXS])
            hand_targets = physical_to_mujoco(hand_filtered)
            
            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                _print_combo_status(
                    arm_filtered,
                    hand_filtered,
                    arm_targets,
                    prefix="[3D Combo|Sim ]",
                )
                last_print = now
            
            targets = np.concatenate([arm_targets, hand_targets])
            set_positions(data, targets)
            mujoco.mj_step(model, data)
            viewer.sync()
    
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control AeroPiper arm + hand with stereo 3D tracking"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Stereo calibration file",
    )
    parser.add_argument(
        "--left-camera",
        type=int,
        required=True,
        help="Left camera index",
    )
    parser.add_argument(
        "--right-camera",
        type=int,
        required=True,
        help="Right camera index",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable mirror in preview",
    )
    parser.add_argument(
        "--arm-smoothing",
        type=float,
        default=0.4,
        help="EMA weight for arm (default: 0.4)",
    )
    parser.add_argument(
        "--hand-smoothing",
        type=float,
        default=0.35,
        help="EMA weight for hand (default: 0.35)",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.05,
        help="Print update interval (seconds)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Skip MuJoCo; just print values",
    )
    parser.add_argument(
        "--arm-update-threshold",
        type=float,
        default=0.03,
        help="Min arm change to emit (default: 0.03)",
    )
    parser.add_argument(
        "--hand-update-threshold",
        type=float,
        default=3.0,
        help="Min hand change to emit (default: 3.0)",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.35,
        help="Max arm jump per frame (default: 0.35)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLO pose model path (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--yolo-frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame with YOLO (default: 1)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load calibration
    print(f"Loading calibration from: {args.calibration}")
    calibration = StereoCalibrator.load_calibration(args.calibration)
    
    if calibration is None:
        print(f"ERROR: Failed to load {args.calibration}")
        print("\nCreate default calibration with:")
        print("  python stereo_control/create_default_calibration.py --left 0 --right 1")
        sys.exit(1)
    
    print("[OK] Calibration loaded")
    print(f"  Image size: {calibration['image_size']}")
    
    tracker = None
    try:
        # Create tracker
        print(f"\nInitializing stereo tracker (cameras {args.left_camera}, {args.right_camera})...")
        tracker = StereoComboTracker(
            calibration=calibration,
            left_camera_index=args.left_camera,
            right_camera_index=args.right_camera,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            arm_smoothing=args.arm_smoothing,
            hand_smoothing=args.hand_smoothing,
            max_step=args.max_step,
            yolo_model=args.yolo_model,
            yolo_frame_skip=args.yolo_frame_skip,
        )
        print("[OK] Tracker initialized")
        
        print("\n" + "="*70)
        print("STEREO 3D ARM + HAND TRACKING")
        print("="*70)
        print("Tracking: 13 DOFs (6 arm + 7 hand) in true 3D")
        print("Mode: View-angle invariant (values constant when you move)")
        print("\nPress Q in preview window or Ctrl+C to exit\n")
        
        if args.print_only:
            run_tracking_only(tracker, args)
        else:
            run_simulation(tracker, args)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tracker is not None:
            tracker.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()

