import mujoco
import mujoco.viewer
import numpy as np
import re

# Load the dual scene (right + left)
model = mujoco.MjModel.from_xml_path("assets/scene_dual.xml")
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
ctrlrange = np.array(model.actuator_ctrlrange)
print("Ctrl ranges (min, max):")
for i in range(model.nu):
    print(f"  {i:2d}: {model.actuator(i).name:>28s} -> {ctrlrange[i]}")


def degrees_to_control(degrees: np.ndarray, hand_ctrlrange: np.ndarray) -> np.ndarray:
    """Map 7 hand DOF degrees (0-90) to actuator ctrl values using provided ctrlrange.

    0 degrees -> fingers open (max tendon length)
    90 degrees -> fingers closed (min tendon length)
    """
    hand_min = hand_ctrlrange[:, 0]
    hand_max = hand_ctrlrange[:, 1]
    normalized = np.asarray(degrees, dtype=float) / 90.0
    return hand_max - (hand_max - hand_min) * normalized


# Hand gestures (7 DOF): [index, middle, ring, pinky, thumb_abd, thumb1, thumb2]
GESTURES = {
    'OPEN': [0, 0, 0, 0, 0, 0, 0],
    'PINCH': [45, 45, 45, 0, 0, 0, 20],
    'PEACE': [60, 30, 0, 0, 60, 60, 0],
    'POINT': [60, 30, 0, 60, 60, 60, 0],
    'THUMBS_UP': [0, 0, 60, 60, 60, 60, 45],
    'FIST': [70, 40, 70, 70, 70, 70, 0],
}

GESTURE_SEQUENCE = ['OPEN', 'PINCH', 'PEACE', 'POINT', 'THUMBS_UP', 'FIST', 'OPEN']


def gesture_sequence(t: float, hand_ctrlrange: np.ndarray, gesture_list, hold_time=2.0, transition_time=1.0):
    cycle_time = (hold_time + transition_time) * len(gesture_list)
    phase = (t % cycle_time)

    time_per_gesture = hold_time + transition_time
    gesture_idx = int(phase / time_per_gesture)
    gesture_idx = min(gesture_idx, len(gesture_list) - 1)
    next_gesture_idx = (gesture_idx + 1) % len(gesture_list)

    time_in_gesture = phase - (gesture_idx * time_per_gesture)
    if time_in_gesture < hold_time:
        transition = 0.0
    else:
        transition = (time_in_gesture - hold_time) / transition_time
        transition = 3 * transition**2 - 2 * transition**3

    current_gesture = GESTURES[gesture_list[gesture_idx]]
    next_gesture = GESTURES[gesture_list[next_gesture_idx]]
    current_controls = degrees_to_control(current_gesture, hand_ctrlrange)
    next_controls = degrees_to_control(next_gesture, hand_ctrlrange)
    return current_controls + (next_controls - current_controls) * transition


# Actuator index layout (include order: right first, then left)
# right: [0..5]=arm(6), [6..12]=hand(7)
# left:  [13..18]=arm(6), [19..25]=hand(7)
RIGHT_ARM_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 13)
LEFT_ARM_SLICE = slice(13, 19)
LEFT_HAND_SLICE = slice(19, 26)

import signal
import atexit

def noop_cleanup():
    pass

atexit.register(noop_cleanup)

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}.")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0

        # Separate arm target generators per robot
        rng_right = np.random.default_rng(7)
        rng_left = np.random.default_rng(21)

        arm_change_interval = 3.0
        next_arm_change = arm_change_interval

        current_arm_right = np.zeros(6)
        next_arm_right = rng_right.uniform(-0.5, 0.5, 6)

        current_arm_left = np.zeros(6)
        next_arm_left = rng_left.uniform(-0.5, 0.5, 6)

        # Pre-slice ctrl ranges for hands
        right_hand_ctrlrange = ctrlrange[RIGHT_HAND_SLICE]
        left_hand_ctrlrange = ctrlrange[LEFT_HAND_SLICE]

        while viewer.is_running():
            if t >= next_arm_change:
                current_arm_right = next_arm_right.copy()
                current_arm_left = next_arm_left.copy()

                next_arm_right = rng_right.uniform(-0.5, 0.5, 6)
                next_arm_left = rng_left.uniform(-0.5, 0.5, 6)
                next_arm_change = t + arm_change_interval

            # Smooth interpolation (same timing, independent targets)
            phase = (t - (next_arm_change - arm_change_interval)) / arm_change_interval
            phase = float(np.clip(phase, 0.0, 1.0))
            smooth = 3 * phase**2 - 2 * phase**3

            arm_targets_right = current_arm_right + (next_arm_right - current_arm_right) * smooth
            arm_targets_left = current_arm_left + (next_arm_left - current_arm_left) * smooth

            # Independent hand gesture cycles (left shifted by +1s)
            hand_targets_right = gesture_sequence(t, right_hand_ctrlrange, GESTURE_SEQUENCE, hold_time=2.0, transition_time=1.0)
            hand_targets_left = gesture_sequence(t + 1.0, left_hand_ctrlrange, GESTURE_SEQUENCE, hold_time=2.0, transition_time=1.0)

            # Build per-robot 13-length vectors
            targets_right = np.concatenate([arm_targets_right, hand_targets_right])
            targets_left = np.concatenate([arm_targets_left, hand_targets_left])

            # Concatenate as [right 13 | left 13]
            data.ctrl[:] = np.concatenate([targets_right, targets_left])

            mujoco.mj_step(model, data)
            viewer.sync()
            t += model.opt.timestep

except KeyboardInterrupt:
    print("\nInterrupted by user")



