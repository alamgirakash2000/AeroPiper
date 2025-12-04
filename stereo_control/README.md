# Stereo 3D Hand Tracking - Quick Reference

Two-camera 3D tracking for AeroPiper robot. View-angle invariant control.

---

## Setup (One Time)

### 1. Detect Cameras
```bash
python stereo_control/detect_cameras.py
```
Note which camera indices are your external cameras (ignore laptop webcam).

### 2. Create Calibration (1080p recommended for C920x)
```bash
python stereo_control/create_default_calibration.py --left 1 --right 2 --resolution 1920x1080 --test
```
Replace `1` and `2` with your camera indices. This creates `stereo_calibration_default.pkl`.

---

## Usage

### Option 1: Dual Hand Tracking (Track both, control with left)
```bash
# Print only (test)
python stereo_control/stereo_dual_hand_gesture.py \
    --calibration stereo_calibration_default.pkl \
    --left-camera 1 --right-camera 2 \
    --print-only

# With MuJoCo
python stereo_control/stereo_dual_hand_gesture.py \
    --calibration stereo_calibration_default.pkl \
    --left-camera 1 --right-camera 2
```
**Tracks:** Both hands  
**Controls:** Left hand only  
**DOFs:** 7 (fingers)

### Option 2: Arm + Single Hand
```bash
python stereo_control/stereo_combo_gesture.py \
    --calibration stereo_calibration_default.pkl \
    --left-camera 1 --right-camera 2
```
**Tracks:** Arm + left hand  
**Controls:** Full robot  
**DOFs:** 13 (6 arm + 7 hand)

### Option 3: Arm + Dual Hands (NEW!)
```bash
python stereo_control/stereo_dual_arm_hand.py \
    --calibration stereo_calibration_default.pkl \
    --left-camera 1 --right-camera 2
```
**Tracks:** Arm + both hands  
**Controls:** Arm + left hand  
**DOFs:** 13 (6 arm + 7 hand from left)

---

## Common Options

```bash
--print-only              # Test without MuJoCo
--smoothing 0.5           # Smoothing (0-1, higher = more responsive)
--no-preview              # Disable camera preview
--resolution 1920x1080    # Use 1080p (for calibration, default for C920x)
```

---

## Troubleshooting

**"Hand not detected"**
- Ensure hand visible in BOTH cameras
- Show left hand (anatomical left)
- Improve lighting
- Move closer (30-50cm)

**"Tracking intermittent"**
- Use 1080p: `--resolution 1920x1080` in calibration
- Better lighting
- Cameras stable (not moving)

**"Wrong hand tracked"**
- System automatically picks your anatomical left hand
- If confused, show only one hand

**"Can't track at distance"**
- Recreate calibration at 1080p (not 720p)
- Use arm tracking (YOLO works farther)

---

## Files

- `detect_cameras.py` - Find camera indices
- `create_default_calibration.py` - Quick calibration
- `stereo_dual_hand_gesture.py` - Both hands, left controls
- `stereo_combo_gesture.py` - Arm + single hand
- `stereo_dual_arm_hand.py` - Arm + both hands
- `module/stereo_calibration.py` - Full calibration tool
- `module/stereo_dual_hand_tracker.py` - Dual hand tracker
- `module/stereo_combo_tracker.py` - Arm+hand tracker

---

## Requirements

```bash
pip install opencv-python mediapipe ultralytics mujoco numpy
```

**Hardware:**
- 2 USB webcams (C920x recommended)
- Laptop/PC (CPU OK, GPU better for YOLO)

---

## Quick Start

```bash
# 1. Find cameras
python stereo_control/detect_cameras.py

# 2. Calibrate (use YOUR camera indices!)
python stereo_control/create_default_calibration.py --left 1 --right 2 --resolution 1920x1080

# 3. Track both hands
python stereo_control/stereo_dual_hand_gesture.py --calibration stereo_calibration_default.pkl --left-camera 1 --right-camera 2
```

Done!

