"""
Create Default Calibration for Logitech C920x

This creates an APPROXIMATE calibration using typical C920x camera parameters.
This is good enough for testing and basic functionality.

For BEST results, do proper calibration later with checkerboard pattern.

Usage:
    python stereo_control/create_default_calibration.py --left 0 --right 1
"""

import argparse
import pickle
import numpy as np
import cv2


def create_c920x_default_calibration(
    left_index: int,
    right_index: int,
    resolution: tuple = (1280, 720),
    baseline_mm: float = 120.0,  # Typical separation for desktop setup
):
    """
    Create default calibration for Logitech C920x cameras.
    
    C920x typical specs:
    - Resolution: 1920×1080 max, we use 1280×720
    - Focal length: ~1000 pixels at 1080p, ~600-700 at 720p
    - Distortion: Minimal (good quality lens)
    - FOV: 78° diagonal
    
    Args:
        left_index: Left camera index
        right_index: Right camera index
        resolution: (width, height) - should match what cameras will use
        baseline_mm: Distance between camera centers in mm
    """
    
    width, height = resolution
    
    # Logitech C920x typical intrinsic parameters (720p)
    # These are approximations - proper calibration will be more accurate
    focal_length_px = 700.0  # Typical for C920 at 720p
    cx = width / 2.0  # Principal point X (image center)
    cy = height / 2.0  # Principal point Y (image center)
    
    # Camera matrix (intrinsics)
    camera_matrix = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients (C920x has good lens, minimal distortion)
    # [k1, k2, p1, p2, k3]
    dist_coeffs = np.array([0.05, -0.02, 0, 0, 0], dtype=np.float64)
    
    # Both cameras assumed identical (same model)
    camera_matrix_left = camera_matrix.copy()
    camera_matrix_right = camera_matrix.copy()
    dist_coeffs_left = dist_coeffs.copy()
    dist_coeffs_right = dist_coeffs.copy()
    
    # Stereo extrinsics (relative position/orientation)
    # Assume cameras are parallel, separated horizontally by baseline
    R = np.eye(3, dtype=np.float64)  # No rotation (parallel cameras)
    T = np.array([[baseline_mm], [0], [0]], dtype=np.float64)  # Horizontal separation
    
    # Compute rectification (even though cameras are "ideal", we still need these)
    # For parallel cameras, rectification is simple
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    
    # Projection matrices
    P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera_matrix @ np.hstack([np.eye(3), np.array([[-baseline_mm], [0], [0]])])
    
    # Disparity-to-depth mapping matrix
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, focal_length_px],
        [0, 0, -1/baseline_mm, 0]
    ], dtype=np.float64)
    
    # Create identity rectification maps (no distortion correction for now)
    map_left = cv2.initUndistortRectifyMap(
        camera_matrix_left,
        dist_coeffs_left,
        R1,
        P1,
        resolution,
        cv2.CV_32FC1
    )
    
    map_right = cv2.initUndistortRectifyMap(
        camera_matrix_right,
        dist_coeffs_right,
        R2,
        P2,
        resolution,
        cv2.CV_32FC1
    )
    
    calibration = {
        "image_size": resolution,
        "camera_matrix_left": camera_matrix_left,
        "dist_coeffs_left": dist_coeffs_left,
        "camera_matrix_right": camera_matrix_right,
        "dist_coeffs_right": dist_coeffs_right,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map_left": map_left,
        "map_right": map_right,
        "rms_error": 0.0,  # Not computed for default calibration
        "is_default": True,  # Flag to indicate this is approximate
        "camera_model": "Logitech C920x (default)",
        "baseline_mm": baseline_mm,
    }
    
    return calibration


def main():
    parser = argparse.ArgumentParser(
        description="Create default calibration for Logitech C920x cameras"
    )
    parser.add_argument(
        "--left",
        type=int,
        required=True,
        help="Left camera index",
    )
    parser.add_argument(
        "--right",
        type=int,
        required=True,
        help="Right camera index",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=120.0,
        help="Distance between cameras in mm (default: 120mm = 12cm)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Camera resolution as WIDTHxHEIGHT (default: 1920x1080 for C920x). Use 1920x1080 if tracking fails at distance.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stereo_calibration_default.pkl",
        help="Output calibration file (default: stereo_calibration_default.pkl)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test cameras after creating calibration",
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    print("\n" + "="*60)
    print("CREATING DEFAULT CALIBRATION")
    print("="*60)
    print(f"Camera model: Logitech C920x (assumed)")
    print(f"Left camera:  Index {args.left}")
    print(f"Right camera: Index {args.right}")
    print(f"Resolution:   {width}×{height}")
    print(f"Baseline:     {args.baseline}mm")
    print("\nWARNING: This is an APPROXIMATE calibration using typical C920x specs.")
    print("For best accuracy, do proper calibration later with:")
    print("  python stereo_control/module/stereo_calibration.py\n")
    
    # Verify cameras are accessible
    print("Checking cameras...")
    cap_left = cv2.VideoCapture(args.left, cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(args.right, cv2.CAP_DSHOW)
    
    if not cap_left.isOpened():
        print(f"ERROR: Cannot open camera {args.left}")
        return
    if not cap_right.isOpened():
        print(f"ERROR: Cannot open camera {args.right}")
        cap_left.release()
        return
    
    # Set resolution
    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify resolution
    actual_w_l = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h_l = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_w_r = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h_r = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[OK] Left camera:  {actual_w_l}x{actual_h_l}")
    print(f"[OK] Right camera: {actual_w_r}x{actual_h_r}")
    
    if (actual_w_l, actual_h_l) != (width, height):
        print(f"\nNOTE: Left camera resolution adjusted to {actual_w_l}x{actual_h_l}")
        resolution = (actual_w_l, actual_h_l)
    
    # Create calibration
    print("\nGenerating default calibration parameters...")
    calibration = create_c920x_default_calibration(
        args.left,
        args.right,
        resolution=resolution,
        baseline_mm=args.baseline,
    )
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(calibration, f)
    
    print(f"\n[OK] Saved default calibration to: {args.output}")
    
    # Test if requested
    if args.test:
        print("\n" + "="*60)
        print("TESTING STEREO FEED")
        print("="*60)
        print("Showing both camera views...")
        print("Press Q or ESC to exit")
        
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            if not ret_l or not ret_r:
                break
            
            # Rectify
            frame_l_rect = cv2.remap(
                frame_l,
                calibration["map_left"][0],
                calibration["map_left"][1],
                cv2.INTER_LINEAR
            )
            frame_r_rect = cv2.remap(
                frame_r,
                calibration["map_right"][0],
                calibration["map_right"][1],
                cv2.INTER_LINEAR
            )
            
            # Show side by side
            combined = np.hstack((frame_l_rect, frame_r_rect))
            
            cv2.putText(
                combined,
                "Left Camera",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.putText(
                combined,
                "Right Camera",
                (width + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.putText(
                combined,
                "Default Calibration (Approximate)",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            cv2.imshow("Stereo Feed Test", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("You can now test hand tracking with:")
    print(f"  python stereo_control/stereo_hand_gesture.py --calibration {args.output}\n")
    print("Or test print-only mode:")
    print(f"  python stereo_control/stereo_hand_gesture.py --calibration {args.output} --print-only\n")
    print("TIP: For better accuracy, do proper calibration later:")
    print("  python stereo_control/module/stereo_calibration.py --left {} --right {}".format(
        args.left, args.right
    ))
    print()


if __name__ == "__main__":
    main()

