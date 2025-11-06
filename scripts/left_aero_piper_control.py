import mujoco
import mujoco.viewer
import numpy as np

# Load the scene (which includes the robot model)
model = mujoco.MjModel.from_xml_path("aero_piper/scene_left.xml")
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
ctrlrange = np.array(model.actuator_ctrlrange)
print("Ctrl ranges (min, max):")
for i in range(model.nu):
    print(f"  {i:2d}: {model.actuator(i).name:>24s} -> {ctrlrange[i]}")

def set_positions(data, positions):
    """Set target positions for the 13 actuators (6 arm + 7 hand)"""
    assert len(positions) == 13
    data.ctrl[:] = positions

# Hand gestures (7 DOF): [index, middle, ring, pinky, thumb_abd, thumb1, thumb2]
# Values in degrees (0-90 range)
GESTURES = {
    'OPEN': [0, 0, 0, 0, 0, 0, 0],
    'PINCH': [45, 45, 45, 0, 0, 0, 20],
    'PEACE': [60, 30, 0, 0, 60, 60, 0],
    'POINT': [60, 30, 0, 60, 60, 60, 0],
    'THUMBS_UP': [0, 0, 60, 60, 60, 60, 45],
    'FIST': [70, 40, 70, 70, 70, 70, 0],
}

# Gesture sequence to cycle through
GESTURE_SEQUENCE = ['OPEN', 'PINCH', 'PEACE', 'POINT', 'THUMBS_UP', 'FIST', 'OPEN']

def degrees_to_control(degrees, ctrlrange):
    """
    Convert degree values (0-90) to actuator control values.
    
    For tendon actuators:
    0 degrees = fingers OPEN (extended) = MAXIMUM tendon length
    90 degrees = fingers CLOSED (bent) = MINIMUM tendon length
    """
    hand_min = ctrlrange[6:, 0]  # Minimum tendon length (fingers closed)
    hand_max = ctrlrange[6:, 1]  # Maximum tendon length (fingers open)
    
    # Normalize degrees to 0-1 range
    normalized = np.array(degrees) / 90.0
    
    # INVERTED mapping: 0 degrees -> max control (open), 90 degrees -> min control (closed)
    controls = hand_max - (hand_max - hand_min) * normalized
    return controls

def gesture_sequence(t, ctrlrange, gesture_list, hold_time=2.0, transition_time=1.0):
    """
    Cycle through gestures with smooth transitions.
    
    Args:
        t: Current time
        ctrlrange: Control ranges for actuators
        gesture_list: List of gesture names to cycle through
        hold_time: How long to hold each gesture (seconds)
        transition_time: How long to transition between gestures (seconds)
    """
    cycle_time = (hold_time + transition_time) * len(gesture_list)
    phase = (t % cycle_time)
    
    # Find current gesture index
    time_per_gesture = hold_time + transition_time
    gesture_idx = int(phase / time_per_gesture)
    gesture_idx = min(gesture_idx, len(gesture_list) - 1)
    
    # Find next gesture index
    next_gesture_idx = (gesture_idx + 1) % len(gesture_list)
    
    # Calculate transition progress within current gesture period
    time_in_gesture = phase - (gesture_idx * time_per_gesture)
    
    if time_in_gesture < hold_time:
        # Holding current gesture
        transition = 0.0
    else:
        # Transitioning to next gesture
        transition = (time_in_gesture - hold_time) / transition_time
        # Smooth transition (ease in-out)
        transition = 3 * transition**2 - 2 * transition**3
    
    # Get current and next gesture controls
    current_gesture = GESTURES[gesture_list[gesture_idx]]
    next_gesture = GESTURES[gesture_list[next_gesture_idx]]
    
    current_controls = degrees_to_control(current_gesture, ctrlrange)
    next_controls = degrees_to_control(next_gesture, ctrlrange)
    
    # Interpolate between gestures
    controls = current_controls + (next_controls - current_controls) * transition
    
    return controls

# Interactive viewer with finger sequence + random arm movement
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0
    np.random.seed(42)  # For reproducible random movements
    arm_change_interval = 3.0  # Change arm target every 3 seconds
    next_arm_change = arm_change_interval
    current_arm_targets = np.zeros(6)
    next_arm_targets = np.random.uniform(-0.5, 0.5, 6)
    
    while viewer.is_running():
        # Smooth random arm movement
        if t >= next_arm_change:
            current_arm_targets = next_arm_targets.copy()
            next_arm_targets = np.random.uniform(-0.5, 0.5, 6)
            next_arm_change = t + arm_change_interval
        
        # Interpolate between current and next arm targets
        transition = (t - (next_arm_change - arm_change_interval)) / arm_change_interval
        transition = min(1.0, max(0.0, transition))
        # Use smooth interpolation
        smooth_transition = 3 * transition**2 - 2 * transition**3  # Smoothstep
        arm_targets = current_arm_targets + (next_arm_targets - current_arm_targets) * smooth_transition
        
        # Get gesture sequence (holds each gesture for 2s, transitions for 1s)
        hand_targets = gesture_sequence(t, ctrlrange, GESTURE_SEQUENCE, hold_time=2.0, transition_time=1.0)
        
        # Combine arm and hand targets
        targets = np.concatenate([arm_targets, hand_targets])
        set_positions(data, targets)

        mujoco.mj_step(model, data)
        viewer.sync()
        t += 0.01
