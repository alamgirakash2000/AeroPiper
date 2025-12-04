"""
Stereo 3D Hand Tracking Module

Provides view-angle invariant hand tracking using two calibrated webcams.

Key components:
- stereo_calibration.py: Calibrate stereo camera pair
- stereo_hand_tracker.py: 3D hand tracking from stereo vision
- stereo_hand_gesture.py: Main control script for MuJoCo

See README.md for setup instructions and SETUP_GUIDE.md for quick start.
"""

__version__ = "1.0.0"
__all__ = ["StereoCalibrator", "StereoHandTracker"]

