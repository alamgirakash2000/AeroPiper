#!/usr/bin/env python3
"""
Demo script for AeroPiper with integrated Intel RealSense D435 cameras.
This script runs the robot simulation and displays the camera feeds from the
simulated RealSense D435 cameras mounted on each wrist.
"""

import time
import argparse
import os
import sys

import numpy as np
import cv2

# Allow running as `python demos/demo_with_camera.py` (so repo root is on sys.path)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aeropiper as suite
from aeropiper.robots import MobileRobot
from aeropiper.utils.input_utils import *

MAX_FR = 25  # max frame rate for running simulation


def create_camera_display(left_img, right_img):
    """
    Create a combined display of left and right wrist camera images.
    
    Args:
        left_img (np.ndarray): Left wrist camera image
        right_img (np.ndarray): Right wrist camera image
    
    Returns:
        np.ndarray: Combined display image
    """
    # Ensure both images have the same height
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]
    
    if h1 != h2:
        # Resize right to match left height
        right_img = cv2.resize(right_img, (int(w2 * h1 / h2), h1))
        h2, w2 = right_img.shape[:2]
    
    # Stack horizontally
    display = np.hstack((left_img, right_img))
    
    # Add labels
    cv2.putText(display, "Left Wrist Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, "Right Wrist Camera", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return display


def main():
    parser = argparse.ArgumentParser(description="AeroPiper with Intel RealSense D435 camera demo")
    parser.add_argument(
        "--gr",
        action="store_true",
        help="Use GR1ArmsOnly instead of AeroPiper.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera image height (default: 480)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera image width (default: 640)",
    )
    args = parser.parse_args()

    # Print welcome info
    print("=" * 70)
    print("Welcome to AeroPiper with Intel RealSense D435 Camera Integration!")
    print(f"Version: {suite.__version__}")
    print("=" * 70)
    print(suite.__logo__)
    print()

    # Create dict to hold options for environment
    options = {}
    robot_name = "GR1ArmsOnly" if args.gr else "AeroPiper"
    print(f"Using robot: {robot_name}")
    print(f"Camera resolution: {args.camera_width}x{args.camera_height}\n")

    # Choose environment
    options["env_name"] = choose_environment()

    # Configure robot
    if "TwoArm" in options["env_name"]:
        options["env_configuration"] = "single-robot"
        options["robots"] = robot_name
    elif "Humanoid" in options["env_name"]:
        options["robots"] = robot_name
    else:
        options["robots"] = robot_name

    # Initialize the environment with camera observations enabled
    print("Initializing simulation environment with camera observations...")
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,  # Enable offscreen rendering for camera obs
        renderer="mjviewer",
        ignore_done=True,
        use_camera_obs=True,  # Enable camera observations
        camera_names=["robot0_left_wrist_cam", "robot0_right_wrist_cam"],  # Both wrist cameras
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        control_freq=20,
    )
    obs = env.reset()
    
    # Keep free camera so you can rotate / pan / zoom with the mouse in main viewer
    env.viewer.set_camera(camera_id=-1)
    
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    print("Environment initialized successfully!")
    print("\nCamera feeds from Intel RealSense D435 (simulated):")
    print("  - Left wrist camera: Mounted on left arm")
    print("  - Right wrist camera: Mounted on right arm")
    print("\nControls:")
    print("  - Press 'q' in camera window to quit")
    print("  - Press 'r' in camera window to reset environment")
    print("  - Press SPACE in camera window to pause/unpause")
    print("  - Simulation window: Mouse to rotate/pan/zoom")
    print("\nStarting demo loop...\n")

    # Main loop
    step_count = 0
    paused = False
    
    try:
        for i in range(10000):
            start = time.time()
            
            if not paused:
                # Generate random action for robot
                action = np.random.randn(*env.action_spec[0].shape)
                
                # Step the simulation
                obs, reward, done, _ = env.step(action)
                step_count += 1
            
            # Render the main simulation window
            env.render()
            
            # Extract camera images from observations
            # The camera observations are stored as '<camera_name>_image' in the obs dict
            left_cam_key = "robot0_left_wrist_cam_image"
            right_cam_key = "robot0_right_wrist_cam_image"
            
            if left_cam_key in obs and right_cam_key in obs:
                left_img = obs[left_cam_key]
                right_img = obs[right_cam_key]
                
                # MuJoCo returns images in RGB format, convert to BGR for OpenCV
                left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
                
                # Create combined display
                display = create_camera_display(left_img_bgr, right_img_bgr)
                
                # Add status information
                status_text = f"Step: {step_count} | FPS: {1.0/(time.time()-start+0.001):.1f}"
                if paused:
                    status_text += " | PAUSED"
                cv2.putText(display, status_text, (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Show the camera feeds
                cv2.imshow("AeroPiper - Intel RealSense D435 Cameras", display)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit signal received. Exiting...")
                    break
                elif key == ord('r'):
                    print("\nResetting environment...")
                    obs = env.reset()
                    step_count = 0
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'}")
            else:
                print(f"Warning: Camera observations not found in obs dict. Available keys: {obs.keys()}")
                time.sleep(1)
            
            # Limit frame rate if necessary
            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)
            
            # Print progress every 100 steps
            if step_count % 100 == 0 and step_count > 0:
                print(f"Steps completed: {step_count}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
