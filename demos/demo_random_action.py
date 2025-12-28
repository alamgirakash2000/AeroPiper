import time

import argparse

import os
import sys

import numpy as np

# Allow running as `python demos/demo_random_action.py` (so repo root is on sys.path)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aeropiper as suite
from aeropiper.robots import MobileRobot
from aeropiper.utils.input_utils import *

MAX_FR = 25  # max frame rate for running simluation

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gr",
        action="store_true",
        help="Use GR1ArmsOnly instead of AeroPiper.",
    )
    args = parser.parse_args()

    # Create dict to hold options that will be passed to env creation call
    options = {}
    robot_name = "GR1ArmsOnly" if args.gr else "AeroPiper"

    # print welcome info
    print("Welcome to aeropiper v{}!".format(suite.__version__))
    print(suite.__logo__)
    print(f"Using robot: {robot_name}\n")

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # For TwoArm tasks, run the bimanual single-robot configuration by default.
        options["env_configuration"] = "single-robot"
        options["robots"] = robot_name
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = robot_name
    else:
        options["robots"] = robot_name

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        renderer="mjviewer",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    # Keep free camera so you can rotate / pan / zoom with the mouse
    env.viewer.set_camera(camera_id=-1)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # do visualization
    for i in range(10000):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)

        obs, reward, done, _ = env.step(action)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
