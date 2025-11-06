import mujoco
import mujoco.viewer
import numpy as np
import re
from pathlib import Path
import xml.etree.ElementTree as ET

# Ensure left variant XML without defaults exists (avoids duplicate default classes on include)
left_src = Path("aero_piper/aero_piper_left.xml")
left_variant = Path("aero_piper/aero_piper_left_nodefault_shifted.xml")
right_src = Path("aero_piper/aero_piper_right.xml")
right_variant = Path("aero_piper/aero_piper_right_shifted.xml")

def build_left_variant(src: Path, dst: Path) -> None:
    xml_text = src.read_text()
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()

    # 1) Remove top-level <default> to avoid duplicate class names
    for child in list(root):
        if child.tag == 'default':
            root.remove(child)
            break

    # 2) Trim assets: keep only meshes needed for the left hand subtree (left_* and tetheria_mount)
    asset = root.find('asset')
    if asset is not None:
        for elem in list(asset):
            tag = elem.tag
            if tag == 'material':
                asset.remove(elem)
                continue
            if tag == 'mesh':
                name = elem.get('name')
                file_attr = elem.get('file', '')
                # Keep only left_* named meshes and tetheria_mount; drop generic arm meshes (link*, base_link, etc.)
                keep = (name is not None and (name.startswith('left_') or name == 'tetheria_mount'))
                if not keep:
                    # Also drop file-based meshes for links
                    if file_attr.startswith('link') or file_attr in { 'base_link.stl', 'link1.stl', 'link2.stl', 'link3.stl', 'link4.stl', 'link5.stl', 'link6.stl' }:
                        asset.remove(elem)
                        continue
                    # Otherwise drop unnamed meshes too
                    if name is None:
                        asset.remove(elem)
                        continue

    # Helper: prefix values that are names (skip already left_/right_)
    def maybe_prefix(val: str) -> str:
        if val is None:
            return val
        if val.startswith('left_') or val.startswith('right_'):
            return val
        return 'left_' + val

    # Attributes that reference names and should be prefixed
    ref_attrs = {
        'name', 'joint', 'tendon', 'site', 'geom', 'body1', 'body2', 'joint1', 'joint2', 'sidesite', 'body'
    }

    # 3) Walk all elements and prefix names/references for the left subtree
    for elem in root.iter():
        for attr, val in list(elem.attrib.items()):
            if not val:  # Skip empty values
                continue
            # Skip class and material to reuse right-side defaults/materials
            if attr in ('class', 'material'):
                continue
            if attr == 'mesh':
                # Do not prefix generic arm meshes like link2_0 etc; keep left_* as-is
                if val.startswith('left_') or val.startswith('right_'):
                    continue
                else:
                    # Reference arm meshes from right include: leave unmodified
                    continue
            if attr in ref_attrs:
                elem.set(attr, maybe_prefix(val))

    # 4) Offset and rename the root left base body (side-by-side along +Y)
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for b in worldbody.findall('body'):
            if b.get('name') == 'left_base_link' or b.get('name') == 'base_link':
                b.set('name', 'left_base_link')
                b.set('pos', '0 0.40 0')
                break

    # 5) Fix the hand orientation in dual scene: flip mount and palm quaternions to match right hand
    def find_body_recursive(parent, name):
        for b in parent.findall('.//body'):
            if b.get('name') == name:
                return b
        return None
    
    # Find and fix tetheria_mount orientation (thumb on top)
    mount = find_body_recursive(root, 'left_tetheria_mount')
    if mount is not None:
        mount.set('quat', '-0.5 0.5 0.5 0.5')  # Match right hand orientation
    
    # Find and fix palm orientation (mount on back of hand, not palm)
    palm = find_body_recursive(root, 'left_palm')
    if palm is not None:
        palm.set('quat', '0.4056064898 -0.5792092674 0.5792092674 -0.4056064898')  # Flip palm

    dst.write_text(ET.tostring(root, encoding='unicode'))

# Always (re)build the left variant to ensure it's up-to-date
def build_right_variant(src: Path, dst: Path) -> None:
    xml_text = src.read_text()
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()
    # Offset only the base body along -Y; keep defaults/materials as-is
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for b in worldbody.findall('body'):
            if b.get('name') == 'right_base_link':
                b.set('pos', '0 -0.40 0')
                break
    dst.write_text(ET.tostring(root, encoding='unicode'))

# (Re)build both variants so positions are side-by-side
if left_src.exists():
    build_left_variant(left_src, left_variant)
if right_src.exists():
    build_right_variant(right_src, right_variant)

# Function to clean up temporary variant files
def cleanup_variants():
    """Remove temporary variant XML files."""
    if left_variant.exists():
        left_variant.unlink()
        print(f"Cleaned up: {left_variant}")
    if right_variant.exists():
        right_variant.unlink()
        print(f"Cleaned up: {right_variant}")

# Load the dual scene (right + left)
model = mujoco.MjModel.from_xml_path("aero_piper/scene_dual.xml")
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

# Use try-finally to ensure cleanup even if interrupted (Ctrl+C)
import atexit
import signal

# Register cleanup to run on exit (handles normal exit and signals)
atexit.register(cleanup_variants)

# Also handle termination signals explicitly
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_variants()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

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
finally:
    # Always clean up temporary variant files, even if interrupted
    print("\nCleaning up temporary files...")
    cleanup_variants()



