"""
Stereo Camera Calibration Utility

This module helps you calibrate two cameras for stereo vision:
1. Calibrate each camera individually (intrinsics: focal length, distortion)
2. Calibrate stereo pair (extrinsics: relative rotation and translation)

Usage:
    python stereo_control/module/stereo_calibration.py --left 0 --right 1
"""

import argparse
import os
import pickle
from typing import Optional, Tuple

import cv2
import numpy as np


class StereoCalibrator:
    """Stereo camera calibration using checkerboard pattern."""

    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0,
    ):
        """
        Args:
            checkerboard_size: Inner corners (width, height) - e.g., (9, 6) for 10x7 squares
            square_size_mm: Size of each checkerboard square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        # 3D points in real world space (in mm)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)
        objp *= square_size_mm
        self.objp = objp

        # Storage for calibration data
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane

    def capture_calibration_images(
        self,
        left_camera: cv2.VideoCapture,
        right_camera: cv2.VideoCapture,
        num_samples: int = 20,
    ) -> bool:
        """
        Interactive capture of calibration images.
        
        Instructions:
        - Position the checkerboard so both cameras can see it
        - Press SPACE to capture when both cameras detect the pattern
        - Move the checkerboard to different positions/angles
        - Press Q to finish early
        
        Returns:
            True if enough samples were captured
        """
        print(f"\n{'='*60}")
        print("STEREO CALIBRATION - IMAGE CAPTURE")
        print(f"{'='*60}")
        print(f"Target: {num_samples} image pairs")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners")
        print(f"Square size: {self.square_size_mm}mm")
        print("\nPattern Display:")
        print("  - Printed paper: mount flat on cardboard")
        print("  - iPad/tablet: full-screen, max brightness, avoid glare")
        print("\nInstructions:")
        print("  1. Show checkerboard to BOTH cameras")
        print("  2. Press SPACE when pattern is detected (green overlay)")
        print("  3. Move board to different positions/angles/distances")
        print("  4. Capture from multiple orientations for best results")
        print("  5. Press Q to finish\n")

        count = 0
        while count < num_samples:
            ret_l, frame_l = left_camera.read()
            ret_r, frame_r = right_camera.read()

            if not ret_l or not ret_r:
                print("Failed to read from cameras")
                return False

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            found_l, corners_l = cv2.findChessboardCorners(
                gray_l, self.checkerboard_size, None
            )
            found_r, corners_r = cv2.findChessboardCorners(
                gray_r, self.checkerboard_size, None
            )

            # Draw the corners
            display_l = frame_l.copy()
            display_r = frame_r.copy()

            if found_l and found_r:
                # Refine corner positions
                corners_l_refined = cv2.cornerSubPix(
                    gray_l, corners_l, (11, 11), (-1, -1), self.criteria
                )
                corners_r_refined = cv2.cornerSubPix(
                    gray_r, corners_r, (11, 11), (-1, -1), self.criteria
                )

                cv2.drawChessboardCorners(
                    display_l, self.checkerboard_size, corners_l_refined, found_l
                )
                cv2.drawChessboardCorners(
                    display_r, self.checkerboard_size, corners_r_refined, found_r
                )

                # Add status text
                status = f"READY - Press SPACE ({count}/{num_samples})"
                color = (0, 255, 0)
            else:
                status = f"Move checkerboard into view ({count}/{num_samples})"
                color = (0, 0, 255)
                corners_l_refined = None
                corners_r_refined = None

            cv2.putText(
                display_l,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                display_r,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Show both views
            combined = np.hstack((display_l, display_r))
            cv2.imshow("Stereo Calibration - Left | Right", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found_l and found_r:
                # Capture this pair
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_l_refined)
                self.imgpoints_right.append(corners_r_refined)
                count += 1
                print(f"[OK] Captured pair {count}/{num_samples}")
            elif key == ord("q"):
                print(f"\nFinished early with {count} samples")
                break

        cv2.destroyAllWindows()
        
        if count < 10:
            print(f"\nNOT enough samples ({count} < 10). Calibration failed.")
            return False
            
        print(f"\n[OK] Captured {count} calibration image pairs")
        return True

    def calibrate(
        self, image_size: Tuple[int, int]
    ) -> Optional[dict]:
        """
        Perform stereo calibration.
        
        Args:
            image_size: (width, height) of the camera images
            
        Returns:
            Dictionary with calibration parameters or None if failed
        """
        if len(self.objpoints) < 10:
            print("Not enough calibration samples")
            return None

        print("\nCalibrating cameras (this may take a minute)...")

        # Calibrate left camera
        print("  - Calibrating left camera...")
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_left,
            image_size,
            None,
            None,
        )

        # Calibrate right camera
        print("  - Calibrating right camera...")
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_right,
            image_size,
            None,
            None,
        )

        # Stereo calibration (get relative position/rotation between cameras)
        print("  - Computing stereo parameters...")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-5,
        )

        (
            ret_stereo,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            image_size,
            criteria=criteria_stereo,
            flags=flags,
        )

        # Stereo rectification
        print("  - Computing rectification maps...")
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
        )

        # Create rectification maps
        map_l = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
        )
        map_r = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
        )

        print(f"\n[OK] Calibration complete! (RMS error: {ret_stereo:.4f})")
        print(f"  Baseline distance: {np.linalg.norm(T):.1f}mm")
        print(f"  (This is the distance between camera centers)")

        return {
            "image_size": image_size,
            "camera_matrix_left": mtx_l,
            "dist_coeffs_left": dist_l,
            "camera_matrix_right": mtx_r,
            "dist_coeffs_right": dist_r,
            "R": R,  # Rotation matrix from left to right
            "T": T,  # Translation vector from left to right
            "R1": R1,  # Rectification rotation for left
            "R2": R2,  # Rectification rotation for right
            "P1": P1,  # Projection matrix for left
            "P2": P2,  # Projection matrix for right
            "Q": Q,  # Disparity-to-depth mapping matrix
            "map_left": map_l,
            "map_right": map_r,
            "rms_error": ret_stereo,
        }

    def save_calibration(self, calibration: dict, filepath: str):
        """Save calibration data to file."""
        with open(filepath, "wb") as f:
            pickle.dump(calibration, f)
        print(f"[OK] Saved calibration to: {filepath}")

    @staticmethod
    def load_calibration(filepath: str) -> Optional[dict]:
        """Load calibration data from file."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate stereo camera pair for 3D hand tracking"
    )
    parser.add_argument(
        "--left",
        type=int,
        default=0,
        help="Left camera index (default: 0)",
    )
    parser.add_argument(
        "--right",
        type=int,
        default=1,
        help="Right camera index (default: 1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of calibration image pairs to capture (default: 20)",
    )
    parser.add_argument(
        "--checkerboard-width",
        type=int,
        default=9,
        help="Number of inner corners horizontally (default: 9 for 10x7 board)",
    )
    parser.add_argument(
        "--checkerboard-height",
        type=int,
        default=6,
        help="Number of inner corners vertically (default: 6 for 10x7 board)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Size of checkerboard squares in millimeters (default: 25.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stereo_calibration.pkl",
        help="Output calibration file (default: stereo_calibration.pkl)",
    )

    args = parser.parse_args()

    # Open cameras
    print(f"Opening cameras (left={args.left}, right={args.right})...")
    cap_left = cv2.VideoCapture(args.left, cv2.CAP_DSHOW if os.name == "nt" else None)
    cap_right = cv2.VideoCapture(args.right, cv2.CAP_DSHOW if os.name == "nt" else None)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open cameras")
        return

    # Set camera properties
    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual image size
    ret, frame = cap_left.read()
    if not ret:
        print("Failed to read from left camera")
        return
    image_size = (frame.shape[1], frame.shape[0])
    print(f"Image size: {image_size[0]}x{image_size[1]}")

    # Create calibrator
    calibrator = StereoCalibrator(
        checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
        square_size_mm=args.square_size,
    )

    # Capture calibration images
    success = calibrator.capture_calibration_images(
        cap_left, cap_right, num_samples=args.samples
    )

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    if not success:
        print("\nCalibration failed - not enough samples captured")
        return

    # Perform calibration
    calibration = calibrator.calibrate(image_size)

    if calibration is None:
        print("\nCalibration failed")
        return

    # Save calibration
    calibrator.save_calibration(calibration, args.output)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Use this file with: --calibration {args.output}")
    print("\nNext steps:")
    print("  1. Run the stereo hand tracking:")
    print(f"     python stereo_control/stereo_hand_gesture.py --calibration {args.output}")
    print()


if __name__ == "__main__":
    main()

