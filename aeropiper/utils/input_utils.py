"""
Utility functions for grabbing user inputs
"""

import numpy as np
from typing import Optional

import aeropiper as suite
import aeropiper.utils.transform_utils as T

# from aeropiper.devices import *
from aeropiper.models.robots import *
from aeropiper.robots import *


def choose_environment(only_two_arm: bool = False, default_env: Optional[str] = None):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all environments
    envs = sorted(suite.ALL_ENVIRONMENTS)
    if only_two_arm:
        envs = [e for e in envs if "TwoArm" in e]

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()

    # Resolve default index (fallback to 0 if not found / not specified)
    default_idx = 0
    if default_env is not None and default_env in envs:
        default_idx = envs.index(default_env)

    try:
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1)).strip()
        if s == "":
            k = default_idx
        else:
            # parse input into a number within range
            k = min(max(int(s), 0), len(envs) - 1)
    except:
        k = default_idx
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    # Return the chosen environment name
    return envs[k]


def choose_controller(part_controllers=False):
    """
    Prints out controller options, and returns the requested controller name

    Returns:
        str: Chosen controller name
    """
    # get the list of all controllers
    controllers = list(suite.ALL_PART_CONTROLLERS) if part_controllers else list(suite.ALL_COMPOSITE_CONTROLLERS)

    # Select controller to use
    print("Here is a list of controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print("[{}] {}".format(k, controller))
    print()
    try:
        s = input("Choose a controller for the robot " + "(enter a number from 0 to {}): ".format(len(controllers) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(controllers) - 1)
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(controllers)[k])

    # Return chosen controller
    return controllers[k]


def choose_multi_arm_config():
    """
    Prints out multi-arm environment configuration options, and returns the requested config name

    Returns:
        str: Requested multi-arm configuration name
    """
    # Get the list of all multi arm configs
    env_configs = {
        "Opposed": "opposed",
        "Parallel": "parallel",
        "Single-Robot": "single-robot",
    }

    # Select environment configuration
    print("A multi-arm environment was chosen. Here is a list of multi-arm environment configurations:\n")

    for k, env_config in enumerate(list(env_configs)):
        print("[{}] {}".format(k, env_config))
    print()
    try:
        s = input(
            "Choose a configuration for this environment "
            + "(enter a number from 0 to {}): ".format(len(env_configs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(env_configs) - 1)
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(env_configs)[k]))

    # Return requested configuration
    return list(env_configs.values())[k]


def choose_robots(exclude_bimanual=False, use_humanoids=False, exclude_single_arm=False, default_robot: str = "AeroPiper"):
    """
    Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default). Restrict options to humanoids if @use_humanoids is set to True (Flase by default).

    Args:
        exclude_bimanual (bool): If set, excludes bimanual robots from the robot options
        use_humanoids (bool): If set, use humanoid robots

    Returns:
        str: Requested robot name
    """
    # Repo pruned state: only keep AeroPiper + GR1ArmsOnly in the chooser
    robots = {"AeroPiper", "GR1ArmsOnly"}

    # Maintain compatibility with legacy flags
    if exclude_bimanual:
        robots = set()
    # NOTE: In this pruned repo, we keep both robots visible even when humanoids are requested.
    # Some demos call choose_robots(use_humanoids=True) even for non-humanoid envs.

    # Make sure set is deterministically sorted
    robots = sorted(robots)

    # Select robot
    print("Here is a list of available robots:\n")

    for k, robot in enumerate(robots):
        print("[{}] {}".format(k, robot))
    print()

    # Resolve default index (fallback to 0 if not found)
    default_idx = 0
    if default_robot is not None and default_robot in robots:
        default_idx = robots.index(default_robot)
    try:
        s = input("Choose a robot " + "(enter a number from 0 to {}): ".format(len(robots) - 1)).strip()
        if s == "":
            k = default_idx
        else:
            # parse input into a number within range
            k = min(max(int(s), 0), len(robots) - 1)
    except:
        k = default_idx
        print("Input is not valid. Use {} by default.".format(list(robots)[k]))

    # Return requested robot
    return list(robots)[k]
