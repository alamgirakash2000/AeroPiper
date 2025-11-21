#!/usr/bin/env python3
"""
Control only the AeroPiper physical hand using webcam gesture tracking.
The arm remains stationary in home position.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from typing import Optional
from collections import deque

import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))

from hand_tracker import DEFAULT_YOLO_WEIGHTS, HandGestureController  # type: ignore[import]

# Import robot connection utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aero_open_sdk.aero_hand import AeroHand

# ============================================
# CONFIGURATION
# ============================================

HAND_PORT = "/dev/ttyACM0"
HAND_BAUDRATE = 921600

HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]

# ============================================
# SETUP FUNCTIONS
# ============================================

def setup_serial_permissions():
    """Setup serial port permissions automatically"""
    print("\n" + "=" * 70)
    print("CHECKING SERIAL PORT ACCESS")
    print("=" * 70)
    
    try:
        # Check if port exists
        if not os.path.exists(HAND_PORT):
            print(f"✗ Serial port {HAND_PORT} not found!")
            print(f"→ Please check if the hand is connected")
            print(f"→ Try: ls /dev/ttyACM* /dev/ttyUSB*")
            return False
        
        # Try to open the port to check permissions
        try:
            # Test if we can access it
            with open(HAND_PORT, 'r') as f:
                pass
            print(f"✓ Serial port {HAND_PORT} is accessible!")
            return True
        except PermissionError:
            # Need to fix permissions
            print(f"\n→ Serial port needs permission fix...")
            print("→ Requesting sudo access (you may need to enter password)...\n")
            
            # Fix permissions with sudo
            subprocess.run(["sudo", "chmod", "666", HAND_PORT], check=True)
            
            # Verify
            try:
                with open(HAND_PORT, 'r') as f:
                    pass
                print("✓ Serial port permissions fixed!")
                return True
            except:
                print("✗ Permission fix failed")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"✗ Permission setup failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def init_hand():
    """Initialize Aero Hand"""
    print("\n" + "=" * 70)
    print("INITIALIZING AERO HAND")
    print("=" * 70)
    
    # Setup serial port permissions automatically
    if not setup_serial_permissions():
        return None
    
    print("\n→ Connecting to Aero Hand...")
    hand = AeroHand(port=HAND_PORT, baudrate=HAND_BAUDRATE)
    print(f"    ✓ Connected")
    print("    → Initializing...")
    time.sleep(1.0)  # Give hand time to be ready
    print("    ✓ Ready\n")
    return hand


# ============================================
# CONTROL FUNCTIONS
# ============================================

def _apply_deadband(values: np.ndarray, previous: Optional[np.ndarray], threshold: float):
    """Apply deadband filter to reduce jitter"""
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


class StabilityFilter:
    """
    Advanced filter that holds values steady when hand is not moving significantly.
    Uses a moving window to detect stability and applies median filtering.
    """
    def __init__(self, window_size: int = 5, stability_threshold: float = 3.0):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.history = deque(maxlen=window_size)
        self.stable_value: Optional[np.ndarray] = None
        self.is_stable = False
    
    def update(self, values: np.ndarray) -> np.ndarray:
        """
        Process new values and return filtered output.
        If hand is stable (low variance), hold the stable position.
        """
        values = np.asarray(values, dtype=float)
        
        # Add to history
        self.history.append(values.copy())
        
        # Need at least 3 samples to detect stability
        if len(self.history) < 3:
            self.stable_value = values.copy()
            return values.copy()
        
        # Calculate median (removes spikes)
        history_array = np.array(list(self.history))
        median_values = np.median(history_array, axis=0)
        
        # Calculate variance across the window for each joint
        variances = np.var(history_array, axis=0)
        max_variance = np.max(variances)
        
        # If variance is low, hand is stable - hold position
        if max_variance < self.stability_threshold:
            if not self.is_stable:
                # Just became stable, save this position
                self.stable_value = median_values.copy()
                self.is_stable = True
            # Return the frozen stable value
            return self.stable_value.copy()
        else:
            # Hand is moving, return median filtered values
            self.is_stable = False
            self.stable_value = median_values.copy()
            return median_values.copy()
    
    def reset(self):
        """Reset the filter state"""
        self.history.clear()
        self.stable_value = None
        self.is_stable = False


_print_hand_status_initialized = False

def _print_hand_status(physical: np.ndarray, sent_values: np.ndarray):
    """Print current hand pose and joint values"""
    global _print_hand_status_initialized
    
    # Line 1: Physical pose values from estimation
    pose_vals = " ".join(f"{name}:{v:.1f}" for name, v in zip(HAND_LABELS, physical))
    # Line 2: ACTUAL values being sent to robot (with swap applied)
    # Labels show what each value controls on the physical hand
    sent_labels = HAND_LABELS
    joint_vals = " ".join(f"{name}:{v:.1f}" for name, v in zip(sent_labels, sent_values))
    
    if not _print_hand_status_initialized:
        # First time: just print the two lines
        print(f"Pose:  {pose_vals}")
        print(f"Sent:  {joint_vals}", end="", flush=True)
        _print_hand_status_initialized = True
    else:
        # Subsequent times: move up, clear, and rewrite both lines
        print(f"\033[A\r\033[KPose:  {pose_vals}")
        print(f"\r\033[KSent:  {joint_vals}", end="", flush=True)




def run_hand_control(
    controller: HandGestureController,
    hand: AeroHand,
    args,
):
    """Main control loop for physical hand"""
    print("\n" + "=" * 70)
    print("HAND GESTURE CONTROL ACTIVE")
    print("=" * 70)
    print("→ Move your hand in front of the camera")
    print("→ The physical hand will mirror your gestures")
    print("→ Press Ctrl+C to stop\n")
    
    last_print = 0.0
    hand_last: Optional[np.ndarray] = None
    last_update_time = 0.0
    hand_sent: Optional[np.ndarray] = None  # Track what we actually sent
    
    # Initialize stability filter for anti-jitter
    stability_filter = None
    if args.use_stability:
        stability_filter = StabilityFilter(
            window_size=args.stability_window,
            stability_threshold=args.stability_threshold
        )
        print(f"→ Stability filter enabled (window={args.stability_window}, threshold={args.stability_threshold})\n")
    
    try:
        while not controller.should_stop():
            # Get hand values only (HandGestureController returns only hand values)
            hand_values = controller.update()
            
            # Apply stability filter first (removes jitter and spikes)
            if stability_filter:
                hand_values = stability_filter.update(hand_values)
            
            # Apply deadband filtering
            hand_filtered = _apply_deadband(
                hand_values, 
                hand_last, 
                args.update_threshold
            )
            
            # Check if values changed significantly
            values_changed = hand_last is None or not np.allclose(
                hand_filtered, hand_last, atol=args.update_threshold
            )
            
            # Rate limit updates to physical hand
            now = time.time()
            if values_changed and (now - last_update_time >= args.update_rate):
                # Send PHYSICAL values directly to hand (NOT MuJoCo converted!)
                # physical values are already in 0-100 scale which the hand expects
                hand_seven = [float(v) for v in hand_filtered]  # 7 DOF physical values
                
                # Save what we're actually sending
                hand_sent = np.array(hand_seven)
                
                # Convert to 16-joint format and send to hand
                hand_sixteen = hand.convert_seven_joints_to_sixteen(hand_seven)
                hand.set_joint_positions(hand_sixteen)
                
                if args.verbose:
                    stable_indicator = " [STABLE]" if (stability_filter and stability_filter.is_stable) else ""
                    print(f"\n[SEND] Physical 7-joint: {hand_seven}{stable_indicator}")
                
                # Small delay for hand to process command (hand needs time)
                time.sleep(0.05)
                
                last_update_time = now
                hand_last = hand_filtered
            
            # Print status at regular intervals (show actual sent values, not MuJoCo converted)
            if now - last_print >= max(args.print_interval, 1e-3):
                # If we haven't sent anything yet, show the values we would send
                if hand_sent is None:
                    hand_sent_display = np.array([float(v) for v in hand_filtered])
                else:
                    hand_sent_display = hand_sent
                
                _print_hand_status(hand_filtered, hand_sent_display)
                last_print = now
                
    except KeyboardInterrupt:
        print("\n\n→ Interrupted, opening hand...")
        # Open hand on exit
        open_gesture = [0, 0, 0, 0, 0, 0, 0]
        hand_sixteen = hand.convert_seven_joints_to_sixteen(open_gesture)
        hand.set_joint_positions(hand_sixteen)
        time.sleep(1.0)
    finally:
        print("\n")


# ============================================
# COMMAND LINE ARGUMENTS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Control the AeroPiper physical hand with real-time webcam-based gestures "
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
        "--update-threshold",
        type=float,
        default=2.0,
        help="Minimum change (0-100 scale) required to emit a new value (default: 2.0).",
    )
    parser.add_argument(
        "--update-rate",
        type=float,
        default=0.1,
        help="Minimum time between physical hand updates in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=HAND_PORT,
        help=f"Serial port for Aero Hand (default: {HAND_PORT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug messages when sending commands to hand.",
    )
    parser.add_argument(
        "--use-stability",
        action="store_true",
        help="Enable stability filter to reduce jitter (recommended).",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=5,
        help="Number of frames for stability detection (default: 5).",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=3.0,
        help="Variance threshold for detecting stable hand (default: 3.0, lower = more aggressive).",
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
    parser.add_argument(
        "--disable-thumb1",
        action="store_true",
        help="Force Thumb1 to 0 for abduction isolation testing.",
    )
    return parser.parse_args()


# ============================================
# MAIN
# ============================================

def main():
    args = parse_args()
    
    # Update port if specified
    global HAND_PORT
    HAND_PORT = args.port
    
    controller = None
    hand = None
    
    try:
        # Initialize hand
        hand = init_hand()
        if hand is None:
            print("✗ Failed to initialize hand")
            return False
        
        # Initialize gesture controller (hand-only tracking)
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
            thumb1_disabled=args.disable_thumb1,
        )
        
        # Run control loop
        run_hand_control(controller, hand, args)
        
        return True
        
    except KeyboardInterrupt:
        print("\n→ Shutting down...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if controller is not None:
            controller.close()
        if hand is not None:
            print("→ Closing hand connection...")
            hand.close()
            print("✓ Done\n")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

