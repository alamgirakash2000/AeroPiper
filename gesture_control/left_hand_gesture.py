import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing from gesture_control/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from physical_to_mujoco import physical_to_mujoco  # type: ignore[import]
from hand_tracker import HandGestureController  # type: ignore[import]

HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]


def _load_mujoco_scene():
    model = mujoco.MjModel.from_xml_path("assets/scene_left.xml")
    data = mujoco.MjData(model)
    return model, data


def set_positions(data_obj, positions):
    """Set target positions for the 13 actuators (6 arm + 7 hand)."""
    assert len(positions) == 13
    data_obj.ctrl[:] = positions


def _apply_deadband(values: np.ndarray, previous: np.ndarray | None, threshold: float):
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


def _print_status(
    physical: np.ndarray,
    mujoco_vals: np.ndarray,
    prefix: str = "[Hand DOFs]",
):
    dof_stats = " ".join(f"{name}:{value:5.1f}" for name, value in zip(HAND_LABELS, physical))
    tendon_stats = " ".join(f"T{idx + 1}:{value:6.3f}" for idx, value in enumerate(mujoco_vals))
    lines = f"\r{prefix} {dof_stats}\n[Tendons] {tendon_stats}"
    print(lines, end="\033[F", flush=True)


def run_tracking_only(
    controller,
    update_threshold: float,
    print_interval: float = 0.1,
):
    """Continuously print the 7 physical control values without launching MuJoCo."""
    last_print = 0.0
    last_sent: np.ndarray | None = None
    try:
        while not controller.should_stop():
            physical_hand = controller.update()
            filtered = _apply_deadband(physical_hand, last_sent, update_threshold)
            last_sent = filtered
            hand_targets = physical_to_mujoco(filtered)
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, hand_targets)
                last_print = now
    finally:
        print()


def run_simulation(controller, update_threshold: float, print_interval: float):
    model, data = _load_mujoco_scene()
    arm_targets = np.zeros(6)
    last_sent: np.ndarray | None = None
    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not controller.should_stop():
            physical_hand = controller.update()
            filtered = _apply_deadband(physical_hand, last_sent, update_threshold)
            last_sent = filtered
            hand_targets = physical_to_mujoco(filtered)
            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(filtered, hand_targets, prefix="[Hand DOFs|Sim]")
                last_print = now
            targets = np.concatenate([arm_targets, hand_targets])
            set_positions(data, targets)
            mujoco.mj_step(model, data)
            viewer.sync()
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Control the AeroPiper left hand with real-time webcam-based gestures "
            "while keeping the arm fixed."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window if running headless.",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable mirroring in the preview window.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="EMA weight in [0,1] for new gesture samples (default: 0.35).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="Seconds between CLI print updates.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Skip launching MuJoCo; just print the filtered physical values.",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=5.0,
        help="Minimum change (0-100 scale) required to emit a new value (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = None
    try:
        controller = HandGestureController(
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            smoothing=args.smoothing,
        )
        threshold = max(args.update_threshold, 0.0)
        if args.print_only:
            run_tracking_only(
                controller,
                threshold,
                args.print_interval,
            )
        else:
            run_simulation(controller, threshold, args.print_interval)
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
