"""
Stereo Dual Hand Tracking - Track BOTH hands, control with LEFT

Tracks both left and right hands in 3D using stereo vision.
Prints values for BOTH hands in terminal.
Only LEFT hand controls the MuJoCo robot.

Usage:
    python stereo_control/stereo_dual_hand_gesture.py --calibration stereo_calibration_default.pkl --left-camera 1 --right-camera 2
"""

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
from stereo_dual_hand_tracker import StereoDualHandTracker  # type: ignore[import]
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]

HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]


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


def _print_dual_hand_status(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    mujoco_vals: Optional[np.ndarray],
    prefix: str,
):
    """Print both hands' status."""
    # Left hand
    if left_hand is not None:
        left_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(HAND_LABELS, left_hand))
        left_line = f"LEFT:  {left_stats}"
    else:
        left_line = "LEFT:  Not detected"
    
    # Right hand
    if right_hand is not None:
        right_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(HAND_LABELS, right_hand))
        right_line = f"RIGHT: {right_stats}"
    else:
        right_line = "RIGHT: Not detected"
    
    # MuJoCo control (only from left hand)
    if mujoco_vals is not None:
        tendon_stats = " ".join(f"T{idx+1}:{val:6.3f}" for idx, val in enumerate(mujoco_vals))
        mujoco_line = f"[MuJoCo Control] {tendon_stats}"
    else:
        mujoco_line = "[MuJoCo Control] No left hand"
    
    # Print three lines
    print(f"\r{prefix} {left_line}\033[K", end="")
    print(f"\n{right_line}\033[K", end="")
    print(f"\n{mujoco_line}\033[K", end="")
    print("\033[F\033[F", end="", flush=True)  # Move cursor up 2 lines


def run_tracking_only(tracker: StereoDualHandTracker, args):
    """Run tracking without MuJoCo (print values only)."""
    last_print = 0.0
    left_last: Optional[np.ndarray] = None
    
    print("\n" * 3)  # Space for 3 lines
    
    try:
        while not tracker.should_stop():
            left_hand, right_hand = tracker.update()
            
            # Apply deadband to left hand (for control)
            if left_hand is not None:
                left_filtered = _apply_deadband(left_hand, left_last, args.update_threshold)
                left_last = left_filtered
                hand_targets = physical_to_mujoco(left_filtered)
            else:
                left_filtered = None
                hand_targets = None
            
            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                _print_dual_hand_status(
                    left_filtered,
                    right_hand,  # Right hand printed but not controlled
                    hand_targets,
                    prefix="[Dual 3D|Print]",
                )
                last_print = now
    finally:
        print("\n\n\n")  # Clear lines


def run_simulation(tracker: StereoDualHandTracker, args):
    """Run MuJoCo simulation with dual hand tracking."""
    model, data = _load_mujoco_scene()
    arm_targets = np.zeros(6)  # Arm fixed
    last_print = 0.0
    left_last: Optional[np.ndarray] = None
    
    print("\n" * 3)  # Space for 3 lines
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not tracker.should_stop():
            left_hand, right_hand = tracker.update()
            
            # Apply deadband and control with LEFT hand only
            if left_hand is not None:
                left_filtered = _apply_deadband(left_hand, left_last, args.update_threshold)
                left_last = left_filtered
                hand_targets = physical_to_mujoco(left_filtered)
                
                # Send to MuJoCo (only left hand controls!)
                targets = np.concatenate([arm_targets, hand_targets])
                set_positions(data, targets)
            else:
                left_filtered = None
                hand_targets = None
            
            now = time.time()
            if now - last_print >= max(args.print_interval, 1e-3):
                _print_dual_hand_status(
                    left_filtered,
                    right_hand,  # Right hand visible but doesn't control
                    hand_targets,
                    prefix="[Dual 3D|Sim ]",
                )
                last_print = now
            
            mujoco.mj_step(model, data)
            viewer.sync()
    
    print("\n\n\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track BOTH hands in 3D, control MuJoCo with LEFT hand only"
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
        "--smoothing",
        type=float,
        default=0.35,
        help="EMA smoothing (default: 0.35)",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="Print interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Skip MuJoCo; just print values",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=5.0,
        help="Min change to emit (default: 5.0)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load calibration
    print(f"Loading calibration from: {args.calibration}")
    calibration = StereoCalibrator.load_calibration(args.calibration)
    
    if calibration is None:
        print(f"ERROR: Failed to load {args.calibration}")
        print("\nCreate calibration with:")
        print("  python stereo_control/create_default_calibration.py --left 1 --right 2")
        sys.exit(1)
    
    print("[OK] Calibration loaded")
    
    tracker = None
    try:
        # Create dual hand tracker
        print(f"Initializing dual hand tracker (cameras {args.left_camera}, {args.right_camera})...")
        tracker = StereoDualHandTracker(
            calibration=calibration,
            left_camera_index=args.left_camera,
            right_camera_index=args.right_camera,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            smoothing=args.smoothing,
        )
        print("[OK] Tracker initialized")
        
        print("\n" + "="*70)
        print("STEREO DUAL HAND TRACKING")
        print("="*70)
        print("Tracking: BOTH hands (left AND right)")
        print("Control:  Only LEFT hand controls MuJoCo")
        print("Display:  Both hands shown in terminal and preview")
        print("\nShow both hands to see values for each!")
        print("Press Q or Ctrl+C to exit\n")
        
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


