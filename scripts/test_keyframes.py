#!/usr/bin/env python3
"""
Test script to verify keyframes and joint control in the dual arm scene.
"""

import time
import argparse

import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from camera_preview import CameraPreviewer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test keyframes and joint control")
    parser.add_argument("--camera", action="store_true", help="Show wrist camera feeds in OpenCV windows")
    args = parser.parse_args()
    # Load the model
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)

    def apply_midpose():
        """Set 6+6 arm joints to specified mid values; zero everything else."""
        midpose = {
            "joint1": 0.0,
            "joint2": 1.57,
            "joint3": -1.35,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_joint1": 0.0,
            "left_joint2": 1.57,
            "left_joint3": -1.35,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 0.0,
        }

        data.qpos[:] = 0
        data.qvel[:] = 0
        data.ctrl[:] = 0

        for jname, val in midpose.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = val

            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if act_id >= 0:
                data.ctrl[act_id] = val

    # Initialize pose before anything else
    apply_midpose()
    
    print(f"Model loaded successfully!")
    print(f"Number of joints (nq): {model.nq}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of keyframes: {model.nkey}")
    
    # Print joint names
    print("\nJoint names:")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_id = model.joint(i).id
        print(f"  {i}: {joint_name} (qposadr: {model.jnt_qposadr[i]})")
    
    # Print keyframe names and data
    print("\nKeyframes:")
    for i in range(model.nkey):
        key_name = model.key(i).name
        print(f"  {i}: {key_name}")
        # Print first 10 qpos values
        qpos = model.key_qpos[i, :10]
        print(f"     First 10 qpos: {qpos}")
    
    print("\nStarting viewer...")
    print("Controls:")
    print("  - Use sliders in the viewer to control joints")
    print("  - Press 'Ctrl+P' to print current joint positions")
    print("  - Press number keys (1, 2, etc.) to load keyframes")
    print("  - Press Space to pause/unpause simulation")
    print("  - Live wrist camera windows (cv2) if installed; else no camera preview.")
    
    # Precompute which actuators directly drive joints (skip tendon actuators).
    # We'll hard-clamp those joints to their control values each frame so they
    # stay exactly at the commanded setpoint (no drift).
    joint_actuators = []
    for act_id in range(model.nu):
        if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
            j_id = model.actuator_trnid[act_id][0]
            qadr = model.jnt_qposadr[j_id]
            joint_actuators.append((qadr, act_id, model.joint(j_id).name))

    print("\nHard-clamping these joint actuators to ctrl values:")
    for _, act_id, jname in joint_actuators:
        print(f"  actuator {act_id}: joint '{jname}'")
    
    # Enable OpenCV camera preview if --camera flag is provided
    ENABLE_CAMERA_PREVIEW = args.camera
    if ENABLE_CAMERA_PREVIEW:
        previewer = CameraPreviewer(
            model,
            camera_names=["right_wrist_cam", "left_wrist_cam"],
            interval=0.05,  # ~20 FPS update rate
            log_prefix="test_keyframes",
            width=320,
            height=240,
        )

    print("\n=== Controls ===")
    print("  Use sliders in MuJoCo viewer to control joints")
    if ENABLE_CAMERA_PREVIEW:
        print("  Wrist camera feeds will open in OpenCV windows")
    else:
        print("  Use --camera flag to see wrist camera feeds")
    print("  Press ESC or close window to exit\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Step physics so tendon/hand actuators respond to ctrl
            mujoco.mj_step(model, data)

            # Hard-clamp joint-actuated arm DoFs to ctrl to prevent drift
            for qadr, act_id, _ in joint_actuators:
                data.qpos[qadr] = data.ctrl[act_id]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            # Sync viewer
            viewer.sync()
            
            # Optional camera preview (if enabled)
            if ENABLE_CAMERA_PREVIEW:
                previewer.update(data)

            # Simple pacing to avoid a busy loop
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if ENABLE_CAMERA_PREVIEW:
        previewer.close()

if __name__ == "__main__":
    main()

