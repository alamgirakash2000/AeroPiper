#!/usr/bin/env python3
"""
Launch a Mujoco viewer for an AeroPiper environment and step with random actions
or zero-hold. Choose the task via --env.
"""

import argparse
import sys
import time
from pathlib import Path
import os

import mujoco
import mujoco.viewer
import numpy as np

# Make the script runnable from anywhere (repo root or outside of it).
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # /.../aeropiper_playground
MODULE_DIR = SCRIPT_DIR / "module"
for p in (REPO_ROOT, MODULE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Default to local/empty menagerie cache and disable warp if user didn't set them.
os.environ.setdefault("MUJOCO_MENAGERIE_PATH", "/tmp/mj_empty")
os.makedirs(os.environ["MUJOCO_MENAGERIE_PATH"], exist_ok=True)
os.environ.setdefault("MUJOCO_WARP_DISABLE", "1")

from camera_preview import CameraPreviewer  # type: ignore
from envs import (  # type: ignore  # import after sys.path tweak
    AeroPiperAssembly,
    AeroPiperBase,
    AeroPiperHandover,
    AeroPiperPickPlace,
)


ENV_MAP = {
    "base": AeroPiperBase,
    "pick_place": AeroPiperPickPlace,
    "handover": AeroPiperHandover,
    "assembly": AeroPiperAssembly,
}


def main():
    parser = argparse.ArgumentParser(description="Play with AeroPiper Mujoco environments")
    parser.add_argument(
        "--env",
        default="pick_place",
        choices=ENV_MAP.keys(),
        help="Which AeroPiper task to load",
    )
    parser.add_argument(
        "--mode",
        default="zero",
        choices=["zero", "random"],
        help="zero = hold still; random = random actions each step",
    )
    parser.add_argument("--hz", type=float, default=50.0, help="Control rate (Hz)")
    parser.add_argument("--random-scale", type=float, default=0.2, help="Range for random actions (-s..s)")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Open wrist camera previews in OpenCV windows (right & left). Viewer camera stays unchanged.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help="Optional fixed viewer camera (e.g., right_wrist_cam, left_wrist_cam, agentview). "
             "Short aliases: 'right' -> right_wrist_cam, 'left' -> left_wrist_cam.",
    )
    args = parser.parse_args()

    env_cls = ENV_MAP[args.env]
    env = env_cls()
    env.reset()
    cam_id = None
    cam_name = None
    if args.camera_name:
        cam_name = args.camera_name

    if cam_name:
        alias = cam_name.lower()
        if alias == "right":
            alias = "right_wrist_cam"
        elif alias == "left":
            alias = "left_wrist_cam"
        try:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, alias)
        except KeyError:
            print(f"Warning: camera '{cam_name}' not found; using free camera.")

    previewer = None
    if args.camera:
        # Open both wrist cams in OpenCV popups; viewer stays on default camera.
        previewer = CameraPreviewer(
            env.model,
            camera_names=["right_wrist_cam", "left_wrist_cam"],
            interval=0.05,  # ~20 FPS
            log_prefix="play_env",
            width=320,
            height=240,
        )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        if cam_id is not None:
            viewer.cam.fixedcamid = cam_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        dt = 1.0 / args.hz
        try:
            while viewer.is_running():
                t0 = time.time()
                if args.mode == "random":
                    s = args.random_scale
                    action = np.random.uniform(-s, s, size=env.action_size)
                else:
                    action = np.zeros(env.action_size)
                env.step(action)

                if previewer:
                    previewer.update(env.data)

                viewer.sync()
                # pace loop
                remaining = dt - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if previewer:
                previewer.close()


if __name__ == "__main__":
    main()
