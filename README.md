# AeroPiper: A Dual-Hand Manipulation System

<p align="center">
  <img src="images/demo.gif" alt="Demo" width="90%"/>
</p>

## Images

<p align="center">
  <img src="images/piper.png" alt="PiPER Arm" width="45%"/>
  <img src="images/aero_hand_open.png" alt="TetherIA Aero Hand" width="45%"/>
  <br/>
</p>

### Description
AeroPiper is a dual-hand manipulation system that combines two AgileX PiPER 6‑DOF robotic arms with two TetherIA Aero Open hands, targeting dexterous, human-like manipulation tasks. The project pairs high-fidelity MuJoCo simulation assets with reinforcement-learning baselines so you can prototype, train, and evaluate complex bimanual skills rapidly—then transfer them to real hardware.

### Official resources
- **TetherIA Aero Hand Open Docs**: `https://docs.tetheria.ai`
- **AgileX PiPER product page**: `https://global.agilex.ai/products/piper`
- **MuJoCo documentation**: `https://mujoco.readthedocs.io`

---

## Installation

## Setup (conda, Python 3.10)

```bash
conda create -n aeropiper python=3.10 -y
conda activate aeropiper
pip install -e .
```

## Install (pip-only)

From the repo root:
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Verify Installation
```bash
python demos/run.py
```
This file send 14 random action values to the 14 DOFs.

Notes:
- **Tkinter**: `teleop/gui.py` uses Tk. If your environment doesn't have it, install via conda (`conda install -n aeropiper tk -y`)
  or on Ubuntu/Debian: `sudo apt-get install -y python3-tk`

---

## Quick Start

### To run with any random action
```bash
python demos/demo_random_action.py
```


### To run and integrate Teleoperation: Follow the current GUI control
```bash
python teleop/gui.py
```
This GUI file control 6+6 DOFs for both arms, and two control values applies the all the 6 DOFs of each Gripper.


### Camera Integration

```bash
python demos/demo_with_camera.py
```
This demo displays the simulated Intel RealSense D435 camera feeds from both wrists alongside the robot simulation. The cameras are already integrated into the AeroPiper model. Press 'q' to quit, 'r' to reset, or SPACE to pause/unpause.