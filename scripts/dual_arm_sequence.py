#!/usr/bin/env python3
"""
Dual-arm hand trajectory demo running on the AeroPiper base environment.

- Drives arm joints through gentle oscillations around a mid pose.
- Drives both hands through the canned finger trajectories from the original demo.
- Uses the base environment (no task objects) so it can be run anywhere.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# -----------------------------------------------------------------------------
# Path setup: make script runnable from repo root or elsewhere
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # .../aeropiper_playground
MODULE_DIR = SCRIPT_DIR / "module"
for p in (REPO_ROOT, MODULE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Optional noise suppression: avoid warp/menagerie downloads by default.
os.environ.setdefault("MUJOCO_MENAGERIE_PATH", "/tmp/mj_empty")
os.makedirs(os.environ["MUJOCO_MENAGERIE_PATH"], exist_ok=True)
os.environ.setdefault("MUJOCO_WARP_DISABLE", "1")

from physical_to_mujoco import physical_to_mujoco  # type: ignore
from envs import AeroPiperBase

# -----------------------------------------------------------------------------
# Trajectory definition (same as original demo)
# -----------------------------------------------------------------------------
ARM_MID = np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0])

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
    """Get control values for a hand following a trajectory at time t."""
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


def main():
    parser = argparse.ArgumentParser(description="Dual-arm AeroPiper sequence demo")
    parser.add_argument("--camera", action="store_true", help="Show wrist camera feeds in OpenCV windows (if available)")
    args = parser.parse_args()

    env = AeroPiperBase()
    env.reset_robot_to_home()

    model, data = env.model, env.data

    # Build actuator index slices from env actuator order
    right_arm_slice = slice(0, len(env.right_arm_actuators))
    right_hand_slice = slice(
        len(env.right_arm_actuators),
        len(env.right_arm_actuators) + len(env.right_hand_actuators),
    )
    left_arm_slice = slice(
        right_hand_slice.stop,
        right_hand_slice.stop + len(env.left_arm_actuators),
    )
    left_hand_slice = slice(
        left_arm_slice.stop,
        left_arm_slice.stop + len(env.left_hand_actuators),
    )

    # Arm oscillation parameters (matching original demo magnitudes)
    ARM_AMPL_RIGHT = np.array([0.5, 0.35, 0.25, 0.2, 0.2, 0.2])
    ARM_FREQ_RIGHT = np.array([0.3, 0.2, 0.25, 0.15, 0.18, 0.12])
    ARM_PHASE_RIGHT = np.zeros(6)

    ARM_AMPL_LEFT = np.array([0.35, 0.25, 0.3, 0.22, 0.18, 0.15])
    ARM_FREQ_LEFT = np.array([0.26, 0.22, 0.28, 0.17, 0.2, 0.14])
    ARM_PHASE_LEFT = np.array([0.6, 0.4, 0.2, 0.8, 0.3, 0.5])

    # Initialize arms at mid pose, open hands
    data.ctrl[:] = 0
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.ctrl[right_arm_slice] = ARM_MID
    data.ctrl[left_arm_slice] = ARM_MID
    mujoco.mj_forward(model, data)

    def hard_clamp_joint_actuators():
        """Force joint-actuated DoFs to match their ctrl values exactly (mirrors original demo)."""
        for act_id in range(model.nu):
            if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                j_id = model.actuator_trnid[act_id][0]
                qadr = model.jnt_qposadr[j_id]
                data.qpos[qadr] = data.ctrl[act_id]
                data.qvel[qadr] = 0.0
        mujoco.mj_forward(model, data)

    # Clamp once at start
    hard_clamp_joint_actuators()

    # Camera preview stub (only viewer here)
    if args.camera:
        print("Wrist camera feeds are defined in the model; viewer renders them. (No OpenCV preview wired here.)")

    dt = model.opt.timestep
    total_traj_time = sum(d for _, _, d in ACTIVE_TRAJECTORY_NAMED)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t = 0.0
            while viewer.is_running():
                # Arm motion
                arm_offset_right = ARM_AMPL_RIGHT * np.sin(2 * np.pi * ARM_FREQ_RIGHT * t + ARM_PHASE_RIGHT)
                arm_offset_left = ARM_AMPL_LEFT * np.sin(2 * np.pi * ARM_FREQ_LEFT * t + ARM_PHASE_LEFT)
                data.ctrl[right_arm_slice] = ARM_MID + arm_offset_right
                data.ctrl[left_arm_slice] = ARM_MID + arm_offset_left

                # Hand trajectories (same for both hands)
                hand_ctrl = get_trajectory_control(t, ACTIVE_TRAJECTORY_NAMED)
                data.ctrl[right_hand_slice] = hand_ctrl
                data.ctrl[left_hand_slice] = hand_ctrl

                # Step physics (uses env clamping/hold behavior)
                mujoco.mj_step(model, data)
                # Hard clamp arm joints like the original script
                hard_clamp_joint_actuators()

                viewer.sync()
                t += dt
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
