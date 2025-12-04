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
High-level layout (images omitted):

```text
.
├── assets/
│   ├── scene.xml                # main dual scene with frame
│   ├── aero_piper_left_hanging.xml
│   ├── aero_piper_right_hanging.xml
│   ├── frame/
│   └── xml_utils/
├── scripts/
│   ├── live_viewer.py
│   ├── test_keyframes.py
│   ├── dual_arm_sequence.py
│   ├── run_aero_sequence.py
│   └── robot_connection.py
├── gesture_control/
│   ├── hands_control.py
│   ├── arms_and_hands_control.py
│   └── physical_hand_gesture.py
├── images/
└── README.md
```

Brief descriptions:
- `assets/`: MuJoCo scenes (dual with frame, hanging variants) and arena helpers.
- `scripts/`:
  - `live_viewer.py`: hot-reload viewer for MJCF edits.
  - `test_keyframes.py`: mid-pose clamp; lets you drive joints via viewer sliders.
  - `dual_arm_sequence.py`: scripted dual-arm + hands trajectory demo.
  - `run_aero_sequence.py`: wrapper to launch predefined trajectories.
  - `robot_connection.py`: physical bring-up helper for CAN/serial.
- `gesture_control/`:
  - `hands_control.py`: dual-hand webcam control driving both simulated hands.
  - `arms_and_hands_control.py`: webcam control of arms + hands together.
  - `physical_hand_gesture.py`: physical hand gesture mapping utility.

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
python scripts/run_aero_sequence.py
```

The scripts will automatically:
- Setup CAN interface for PiPER arm (requires sudo, will prompt for password)
- Fix serial port permissions for Aero Hand (requires sudo, will prompt for password)
- Initialize both devices
- Run test movements

This takes ~3 minutes and calibrates the hand to its zero positions.

#### Physical hand gesture mapper:
  ```bash
  python gesture_control/physical_hand_gesture.py
  ```

## Work in MuJoCo

- Inspect joints/keyframes and drive via sliders (arms clamped, fingers responsive):
  ```bash
  python scripts/test_keyframes.py
  ```
- Run scripted dual-arm + hands trajectory demo:
  ```bash
  python scripts/dual_arm_sequence.py
  ```

## Gesture Control

- Dual-hand tracking to drive both simulated hands:
  ```bash
  python gesture_control/hands_control.py
  ```
  Tracks both hands with MediaPipe; maps to finger actuators in `assets/scene.xml`. Use `--no-preview` for less lag; tweak `--smoothing` / `--update-threshold` for responsiveness.

- Arms + hands together from one webcam:
  ```bash
  python gesture_control/arms_and_hands_control.py
  ```
  Controls arms (shoulder/elbow/wrist) and fingers in one loop; uses the same scene.
  ```
  Utility for mapping captured gestures to physical hand commands.

