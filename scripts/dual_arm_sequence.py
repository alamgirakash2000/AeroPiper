#!/usr/bin/env python3
import os
import sys
import signal
import atexit
import time

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from scripts/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from physical_to_mujoco import physical_to_mujoco
from camera_preview import CameraPreviewer

MODEL_PATH = "assets/scene.xml"

# Load the dual scene (right + left) with frame
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Actuator index layout (include order: right first, then left)
# right: [0..5]=arm(6), [6..12]=hand(7)
# left:  [13..18]=arm(6), [19..25]=hand(7)
RIGHT_ARM_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 13)
LEFT_ARM_SLICE = slice(13, 19)
LEFT_HAND_SLICE = slice(19, 26)

# Mid/home position for arms (matches MJCF keyframe)
ARM_MID = np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0])

# Physical trajectory format: ([7 DOF values], duration_seconds)
# Physical/MuJoCo order (matched): [thumb_abd, thumb_flex, thumb_tendon, index, middle, ring, pinky]
# Values in [0..100] where 0=open/straight, 100=closed/bent
PHYSICAL_TRAJECTORY_NAMED = [
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    ("Touch Pinkie",                           [100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 2),
    ("Touch Pinkie - Hold",                    [100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 2),
    ("Touch Ring",                             [100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 2),
    ("Touch Ring - Hold",                      [100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 2),
    ("Touch Middle",                           [83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 2),
    ("Touch Middle - Hold",                    [83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 2),
    ("Touch Index",                            [75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.5),
    ("Touch Index - Hold",                     [75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Open Palm - Hold",                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Peace Sign",                             [90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.5),
    ("Peace Sign - Thumb Flex",                [90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
    ("Peace Sign - Hold",                      [90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 1.0),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Open Palm - Hold",                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Rockstar Sign - Close Middle & Ring",    [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5),
    ("Rockstar Sign - Hold",                   [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 1.0),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
]

ACTIVE_TRAJECTORY_NAMED = PHYSICAL_TRAJECTORY_NAMED


def get_trajectory_control(t: float, trajectory_named, transition_time=0.3):
    """
    Get control values for a hand following a trajectory at time t.
    Returns control values for the 7 hand DOF.
    """
    total_time = sum(duration for _, _, duration in trajectory_named)
    t_loop = t % total_time

    accumulated_time = 0.0
    for idx, (_, pose, duration) in enumerate(trajectory_named):
        if accumulated_time + duration > t_loop:
            time_in_segment = t_loop - accumulated_time
            current_pose = np.array(pose, dtype=float)
            next_idx = (idx + 1) % len(trajectory_named)
            next_pose = np.array(trajectory_named[next_idx][1], dtype=float)

            current_ctrl = physical_to_mujoco(current_pose)
            next_ctrl = physical_to_mujoco(next_pose)

            if duration - time_in_segment < transition_time:
                trans = (transition_time - (duration - time_in_segment)) / transition_time
                trans = min(1.0, max(0.0, trans))
                smooth_trans = 3 * trans**2 - 2 * trans**3
                return current_ctrl + (next_ctrl - current_ctrl) * smooth_trans
            else:
                return current_ctrl

        accumulated_time += duration

    return physical_to_mujoco(trajectory_named[0][1])


def hard_clamp_joint_actuators(model: mujoco.MjModel, data: mujoco.MjData):
    """Force joint-actuated DoFs to match their ctrl values exactly."""
    for act_id in range(model.nu):
        if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
            j_id = model.actuator_trnid[act_id][0]
            qadr = model.jnt_qposadr[j_id]
            data.qpos[qadr] = data.ctrl[act_id]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


# Signal handling
def noop_cleanup():
    pass


atexit.register(noop_cleanup)


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}.")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Calculate total trajectory time
total_traj_time = sum(duration for _, _, duration in ACTIVE_TRAJECTORY_NAMED)


# Initialize to mid pose (arms) and open fingers
data.ctrl[:] = 0
data.qpos[:] = 0
data.qvel[:] = 0
data.ctrl[RIGHT_ARM_SLICE] = ARM_MID
data.ctrl[LEFT_ARM_SLICE] = ARM_MID
hard_clamp_joint_actuators(model, data)

# Arm oscillation around mid to make movement more visible
# Right arm and left arm use different amplitudes/phases so they don't mirror.
ARM_AMPL_RIGHT = np.array([0.5, 0.35, 0.25, 0.2, 0.2, 0.2])
ARM_FREQ_RIGHT = np.array([0.3, 0.2, 0.25, 0.15, 0.18, 0.12])
ARM_PHASE_RIGHT = np.zeros(6)

ARM_AMPL_LEFT = np.array([0.35, 0.25, 0.3, 0.22, 0.18, 0.15])
ARM_FREQ_LEFT = np.array([0.26, 0.22, 0.28, 0.17, 0.2, 0.14])
ARM_PHASE_LEFT = np.array([0.6, 0.4, 0.2, 0.8, 0.3, 0.5])  # radians

previewer = CameraPreviewer(camera_names=["right_wrist_cam", "left_wrist_cam"], interval=0.0, log_prefix="dual_arm_sequence")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        last_waypoint_idx = -1

        while viewer.is_running():
            # More noticeable arm motion around the mid pose
            arm_offset_right = ARM_AMPL_RIGHT * np.sin(2 * np.pi * ARM_FREQ_RIGHT * t + ARM_PHASE_RIGHT)
            arm_offset_left = ARM_AMPL_LEFT * np.sin(2 * np.pi * ARM_FREQ_LEFT * t + ARM_PHASE_LEFT)
            right_arm_ctrl = ARM_MID + arm_offset_right
            left_arm_ctrl = ARM_MID + arm_offset_left
            data.ctrl[RIGHT_ARM_SLICE] = right_arm_ctrl
            data.ctrl[LEFT_ARM_SLICE] = left_arm_ctrl

            # Get hand controls from trajectory
            right_hand_ctrl = get_trajectory_control(t, ACTIVE_TRAJECTORY_NAMED)
            left_hand_ctrl = get_trajectory_control(t, ACTIVE_TRAJECTORY_NAMED)

            # Apply to both hands (mirrored behavior)
            data.ctrl[RIGHT_HAND_SLICE] = right_hand_ctrl
            data.ctrl[LEFT_HAND_SLICE] = left_hand_ctrl

            # Advance physics for tendon-driven fingers and any dynamics
            mujoco.mj_step(model, data)

            # Clamp joint-actuated DoFs (arms) to ctrl (no drift)
            hard_clamp_joint_actuators(model, data)

            # Track waypoint internally (no printing)
            t_in_cycle = t % total_traj_time
            waypoint_idx = 0
            acc_time = 0.0
            for idx, (_, _, duration) in enumerate(ACTIVE_TRAJECTORY_NAMED):
                if acc_time + duration > t_in_cycle:
                    waypoint_idx = idx
                    break
                acc_time += duration
            if waypoint_idx != last_waypoint_idx:
                last_waypoint_idx = waypoint_idx

            # Optional live camera previews (OpenCV if available)
            viewer.sync()
            previewer.update(viewer)
            t += model.opt.timestep

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    previewer.close()

