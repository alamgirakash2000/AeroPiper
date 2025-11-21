# AeroPiper: TetherIA Aero Hand + AgileX PiPER Arm

<p align="center">
  <img src="images/sequence.gif" alt="Demo" width="90%"/>
</p>

## Images

<p align="center">
  <img src="images/piper.png" alt="PiPER Arm" width="45%"/>
  <img src="images/aero_hand_open.png" alt="TetherIA Aero Hand" width="45%"/>
  <br/>
</p>

### Description
This repository provides end-to-end code and guidance to integrate the TetherIA Aero Hand with the AgileX PiPER 6‑DOF robotic arm. AeroPiper lets you:
- Initialize and exercise the physical arm and hand with one command.
- Run MuJoCo simulations of left, right, or dual arm+hand configurations.
- Prototype gesture sequences and arm trajectories with clear, well-structured examples.

Use this as a practical, open resource for research, education, and rapid prototyping with PiPER + Aero in both real and simulated environments.

### Official resources
- **TetherIA Aero Hand Open Docs**: `https://docs.tetheria.ai`
- **AgileX PiPER product page**: `https://global.agilex.ai/products/piper`
- **MuJoCo documentation**: `https://mujoco.readthedocs.io`

## Folder Structure
Below is the high-level layout (images directory omitted):

```text
.
├── aero_piper/
│   ├── aero_piper_left.xml
│   ├── aero_piper_right.xml
│   ├── scene_left.xml
│   ├── scene_right.xml
│   ├── scene_dual.xml
│   └── combined_arm_hand/
│       └── assets/            # STL/OBJ meshes for the arm and hands
├── scripts/
│   ├── robot_connection.py
│   ├── reach_task.py
│   ├── left_aero_piper_control.py
│   ├── right_aero_piper_control.py
│   └── dual_aero_control.py
├── arm_hand_control_dual.py
└── README.md
```

Brief descriptions:
- `aero_piper/`: MuJoCo scenes and resources for single and dual setups.
  - `scene_left.xml`, `scene_right.xml`, `scene_dual.xml`: ready-to-run scenes.
  - `aero_piper_left.xml`, `aero_piper_right.xml`: component models used by scenes.
  - `combined_arm_hand/assets/`: 3D meshes for the PiPER arm and Aero hands.
- `scripts/`:
  - `robot_connection.py`: one-click bring-up; configures CAN and serial, initializes both devices; exposes helpers.
  - `reach_task.py`: simple physical demo (approach, grasp, lift, home) using helpers.
  - `left_aero_piper_control.py`, `right_aero_piper_control.py`: single-arm+hand simulation runners.
  - `dual_aero_control.py`: dual-arm+hand simulation runner.
- `arm_hand_control_dual.py`: standalone dual simulation utility that builds left/right variants and launches a dual viewer.

## Installation
- **Python**: 3.13.5
- **NumPy**: 1.26.4
- **MuJoCo**: 3.3.7

Install required packages:

```bash
pip install piper_sdk
pip install aero-open-sdk
pip install mujoco
```

#### For the gesture control
```bash
pip install opencv-python mediapipe ultralytics
# Install the Torch build that matches your CUDA driver (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Ultralytics will automatically download public checkpoints (default `yolo11n-pose.pt`) the first time you run the hand tracker. To use a custom model, place the `.pt` file in `gesture_control/module/weights/` and pass `--yolo-weights /abs/path/to/file.pt`.

> **Need per-finger YOLO weights?** The stock `yolo11n-pose.pt` model is trained on COCO body pose (17 keypoints) and cannot emit finger joints. Train or download a 21-keypoint hand pose checkpoint (e.g., `yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640`) and run with `--yolo-weights /path/to/best.pt`. Until a hand-specific checkpoint is provided, the script automatically warns and falls back to MediaPipe landmarks.


## Work with the Physical Robot
After connecting the Aero Hand to the PiPER arm using the 3D printed mount, plug in both devices and run:

```bash
python scripts/robot_connection.py
python scripts/reach_task.py
```

The scripts will automatically:
- Setup CAN interface for PiPER arm (requires sudo, will prompt for password)
- Fix serial port permissions for Aero Hand (requires sudo, will prompt for password)
- Initialize both devices
- Run test movements

This takes ~3 minutes and calibrates the hand to its zero positions.

## Work in the MuJoCo Simulation

<p align="center">
  <img src="images/left.png" alt="Left setup" width="45%"/>
  <img src="images/right.png" alt="Right setup" width="45%"/>
</p>

### Left hand
```bash
python scripts/left_aero_piper_control.py
```
Uses `aero_piper/scene_left.xml` and runs a reproducible random arm trajectory with a smooth gesture sequence.

### Right hand
```bash
python scripts/right_aero_piper_control.py
```
Uses `aero_piper/scene_right.xml` with the same control pattern.

## Use both hands with two arms

<p align="center">
  <img src="images/dual_1.png" alt="Dual 1" width="45%"/>
  <img src="images/dual_2.png" alt="Dual 2" width="45%"/>
  <br/>
</p>

Run the dual-arm+dual-hand simulation:

```bash
python scripts/dual_aero_control.py
```

This uses `aero_piper/scene_dual.xml`, driving two independent 6‑DOF arms and two 7‑DOF tendon-driven hands with staggered gesture cycles.

## Webcam Gesture Control

The `gesture_control/` folder contains lightweight MuJoCo demos that mirror your hand or arm motion from a single webcam.

- `gesture_control/left_hand_gesture.py`: tracks a real left hand with Ultralytics YOLO pose (requires a 21-keypoint hand checkpoint; falls back to MediaPipe if none is supplied) and drives the AeroPiper fingers (7 DOFs). Run it with:

  ```bash
  python gesture_control/left_hand_gesture.py
  ```

- `gesture_control/left_arm_gesture.py`: tracks your upper body with MediaPipe Pose and drives the four joints that matter most for quick teleop (J1 base yaw, J2 shoulder pitch, J3 elbow flex, J6 wrist roll). Fingers stay open by default so you can focus on the arm:

  ```bash
  python gesture_control/left_arm_gesture.py
  ```

- `gesture_control/left_combo_gesture.py`: shares a single webcam feed and mirrored preview window between both trackers so you can teleoperate all 13 DOFs (6 arm joints + 7 tendons) at once:

  ```bash
  python gesture_control/left_combo_gesture.py
  ```

All scripts mirror the webcam preview by default (toggle with `--no-mirror-preview`) and accept `--camera-index`, `--print-only`, smoothing/deadband flags, plus `--max-step` (for arm motion) to clamp sudden pose spikes. Make sure `opencv-python` and `mediapipe` are installed for the arm/combo trackers, and install `ultralytics` + `torch` (with the appropriate CUDA build) for the YOLO-based hand tracker. Use `--yolo-weights`, `--hand-backend`, `--thumb-flexion-gain`, `--thumb-flexion-bias`, `--thumb-abd-gain`, `--thumb-abd-bias`, `--thumb-abd-invert`, `--thumb-abd-smoothing`, `--thumb-abd-deadband`, `--thumb1-curve`, `--thumb2-curve`, `--thumb1-freeze-seconds`, `--yolo-device`, `--yolo-imgsz`, and `--pseudo-depth-scale` to fine-tune accuracy for your GPU/webcam setup. Place a 21-keypoint YOLO checkpoint in `gesture_control/module/weights/` or provide an absolute path if you want something other than the auto-downloaded `yolo11n-pose.pt`; otherwise the app will warn and continue with the MediaPipe backend.
